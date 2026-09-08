import httpx
import json
from typing import List, Dict, Any, AsyncGenerator

HEADERS = {
    "Bypass-Tunnel-Remainder": "true",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class KaggleClient:
    def __init__(self, tunnel_url: str):
        self.tunnel_url = tunnel_url.rstrip("/")

    async def check_health(self) -> Dict[str, Any]:
        url = f"{self.tunnel_url}/health"
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=HEADERS) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {
                    "status": "offline",
                    "error": f"Could not reach Kaggle tunnel: {str(e)}"
                }

    async def load_model(self, repo_id: str, filename: str, n_ctx: int = 4096) -> Dict[str, Any]:
        url = f"{self.tunnel_url}/load"
        payload = {
            "repo_id": repo_id,
            "filename": filename,
            "n_ctx": n_ctx
        }
        # Extended timeout to 600 seconds (10 mins) for downloading GGUF files
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True, headers=HEADERS) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                return {"status": "error", "error": f"Kaggle error {e.response.status_code}: {e.response.text}"}
            except Exception as e:
                return {"status": "error", "error": f"Failed to issue load request to Kaggle: {str(e)}"}

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 1024,
        reasoning: str = "off"
    ) -> AsyncGenerator[str, None]:
        url = f"{self.tunnel_url}/chat/stream"
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "reasoning": reasoning
        }

        try:
            async with httpx.AsyncClient(timeout=600.0, follow_redirects=True, headers=HEADERS) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"data: {json.dumps({'error': f'Kaggle worker error HTTP {response.status_code}: {error_text.decode()}'})}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if line.strip():
                            yield f"{line}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Connection Error: Could not reach Kaggle tunnel ({str(exc)})'})}\n\n"
