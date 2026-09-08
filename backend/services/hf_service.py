import httpx
import re
from typing import List, Dict, Any

def extract_quant_type(filename: str) -> str:
    """Extract quantization label (e.g., Q4_K_M, Q8_0, Q2_K) from filename."""
    fname_upper = filename.upper()
    # Match patterns like Q4_K_M, Q8_0, Q2_K, Q3_K_L, IQ3_XS, Q4_0, Q5_1, etc.
    match = re.search(r'(IQ\d+_[A-Z0-9_]+|Q\d+_[A-Z0-9_]+|Q\d+_\d+|Q\d+)', fname_upper)
    if match:
        return match.group(1)
    return "GGUF"

async def fetch_hf_gguf_files(repo_id: str) -> Dict[str, Any]:
    """Fetch tree structure of Hugging Face repository and filter for .gguf files."""
    repo_id = repo_id.strip()
    if not repo_id:
        return {"repo_id": repo_id, "files": [], "error": "Repo ID cannot be empty"}

    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 404:
                return {"repo_id": repo_id, "files": [], "error": f"Repository '{repo_id}' not found on Hugging Face."}
            resp.raise_for_status()
            data = resp.json()
            
            gguf_files = []
            for item in data:
                path = item.get("path", "")
                size = item.get("size", 0)
                if path.endswith(".gguf"):
                    quant = extract_quant_type(path)
                    gguf_files.append({
                        "filename": path,
                        "quant": quant,
                        "size_bytes": size,
                        "size_gb": round(size / (1024 ** 3), 2)
                    })
            
            # Sort by quant name or file size
            gguf_files.sort(key=lambda x: x["quant"])

            return {
                "repo_id": repo_id,
                "files": gguf_files,
                "total_ggufs": len(gguf_files),
                "error": None
            }
        except httpx.HTTPError as e:
            return {"repo_id": repo_id, "files": [], "error": f"Failed to contact Hugging Face API: {str(e)}"}
