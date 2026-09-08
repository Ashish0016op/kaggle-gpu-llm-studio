import os
import sys
import time
import asyncio
import json
from typing import List, Dict, Any, Optional

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False

def check_gpu():
    """Check PyTorch / CUDA GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return True, f"{gpu_name} ({total_mem:.2f} GB VRAM)"
    except Exception:
        pass
    return False, "CPU Inference (CUDA not detected)"

class LlamaModelManager:
    def __init__(self):
        self.llm = None
        self.repo_id: Optional[str] = None
        self.filename: Optional[str] = None
        self.n_ctx: int = 4096
        self.status: str = "idle"  # idle, loading, ready, error
        self.error: Optional[str] = None
        self.loaded_at: Optional[float] = None

    def get_status(self) -> Dict[str, Any]:
        gpu_ok, gpu_info = check_gpu()
        return {
            "status": self.status,
            "llama_cpp_installed": LLAMA_CPP_AVAILABLE,
            "gpu_available": gpu_ok,
            "gpu_info": gpu_info,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "n_ctx": self.n_ctx,
            "loaded_at": self.loaded_at,
            "error": self.error
        }

    def load_model(self, repo_id: str, filename: str, n_ctx: int = 4096) -> Dict[str, Any]:
        if not LLAMA_CPP_AVAILABLE:
            self.status = "error"
            self.error = "llama-cpp-python package is not installed in your Python environment. Run 'pip install llama-cpp-python' (or prebuilt CUDA wheel)."
            return {"status": "error", "error": self.error}

        self.status = "loading"
        self.error = None

        print(f"⏳ Downloading & loading GGUF from HF: {repo_id} / {filename}")

        try:
            if self.llm is not None:
                del self.llm
                self.llm = None

            gpu_ok, _ = check_gpu()
            gpu_layers = -1 if gpu_ok else 0

            llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=gpu_layers,
                n_ctx=n_ctx or 4096,
                verbose=True
            )

            self.llm = llm
            self.repo_id = repo_id
            self.filename = filename
            self.n_ctx = n_ctx or 4096
            self.loaded_at = time.time()
            self.status = "ready"

            print(f"✓ Successfully loaded {repo_id} ({filename})!")
            return {"status": "success", "message": f"Loaded model {filename} successfully"}
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            print(f"❌ Error loading model: {e}")
            return {"status": "error", "error": f"Failed to load model: {str(e)}"}

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 1024,
        reasoning: str = "off"
    ):
        if not LLAMA_CPP_AVAILABLE:
            yield f"data: {json.dumps({'error': 'llama-cpp-python is not installed.'})}\n\n"
            return

        if self.status != "ready" or self.llm is None:
            yield f"data: {json.dumps({'error': 'No model loaded. Please load a Hugging Face GGUF model first.'})}\n\n"
            return

        REASONING_INSTRUCTIONS = {
            "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            "medium": "Reasoning effort is set to medium. Think step by step before answering.",
            "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
        }

        raw_messages = [dict(msg) for msg in messages]

        if reasoning and reasoning != "off" and reasoning in REASONING_INSTRUCTIONS:
            instruction = REASONING_INSTRUCTIONS[reasoning]
            raw_messages.insert(0, {"role": "system", "content": instruction})

        kwargs = {
            "messages": raw_messages,
            "max_tokens": max_tokens or 1024,
            "stream": True,
        }
        if temperature and temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p or 0.95, top_k=top_k or 40)
        else:
            kwargs.update(temperature=0.0)

        try:
            for chunk in self.llm.create_chat_completion(**kwargs):
                choice = chunk["choices"][0]
                delta = choice.get("delta", {}).get("content", "")
                finish_reason = choice.get("finish_reason", None)
                data = json.dumps({"token": delta, "finish_reason": finish_reason})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.001)
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

# Singleton instance
llama_manager = LlamaModelManager()
