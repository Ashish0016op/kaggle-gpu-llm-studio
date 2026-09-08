import os
import sys
import json
import httpx
from typing import Dict, Any, Optional

KAGGLE_NOTEBOOK_SCRIPT = '''# 🚀 Kaggle T4 GPU Remote Tunnel Worker Script (Clean Output)
# Copy and run this entire cell in a Kaggle GPU Notebook!

import os, sys, time, subprocess, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LLAMA_CPP_LIB_VERBOSE"] = "0"

print("⚡ Step 1/3: Installing GPU Dependencies & Llama CUDA Wheels...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu125"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "sse-starlette", "huggingface_hub", "pydantic", "requests", "torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("🚀 Step 2/3: Creating Kaggle Worker Server...")
worker_code = """import os, sys, time, asyncio, json, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LLAMA_CPP_LIB_VERBOSE"] = "0"

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return True, f"{name} ({vram:.2f} GB VRAM)"
    except Exception:
        pass
    return False, "CPU Only"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

state = {"llm": None, "repo_id": None, "filename": None, "status": "idle", "error": None}

class LoadReq(BaseModel):
    repo_id: str
    filename: str
    n_ctx: Optional[int] = 4096

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[Msg]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 1024

@app.get("/health")
def health():
    gpu_ok, gpu_info = check_gpu()
    return {
        "status": "online", 
        "gpu_available": gpu_ok, 
        "gpu_info": gpu_info, 
        "model_status": state["status"], 
        "loaded_model": {"repo_id": state["repo_id"], "filename": state["filename"]}, 
        "error": state["error"]
    }

@app.post("/load")
async def load_m(req: LoadReq):
    global state
    state["status"] = "loading"
    state["error"] = None
    state["repo_id"] = req.repo_id
    state["filename"] = req.filename

    def _do():
        from llama_cpp import Llama
        if state["llm"]: del state["llm"]
        state["llm"] = Llama.from_pretrained(repo_id=req.repo_id, filename=req.filename, n_gpu_layers=-1, n_ctx=req.n_ctx or 4096, verbose=False)
        state["status"] = "ready"
        return {"status": "success"}

    try:
        return await asyncio.to_thread(_do)
    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        return {"status": "error", "error": str(e)}

@app.post("/chat/stream")
async def chat_s(req: ChatReq):
    if state["status"] != "ready" or not state["llm"]:
        raise HTTPException(status_code=400, detail="Model not loaded on Kaggle GPU")
    raw = [m.dict() for m in req.messages]
    kwargs = {"messages": raw, "max_tokens": req.max_tokens or 1024, "stream": True}
    if req.temperature and req.temperature > 0:
        kwargs.update(temperature=req.temperature, top_p=req.top_p or 0.95, top_k=req.top_k or 40)
    else:
        kwargs.update(temperature=0.0)

    async def gen():
        try:
            for chunk in state["llm"].create_chat_completion(**kwargs):
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                yield f"data: {json.dumps({'token': delta})}\\n\\n"
                await asyncio.sleep(0.001)
            yield f"data: {json.dumps({'done': True})}\\n\\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\\n\\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
"""

with open("kaggle_worker.py", "w") as f:
    f.write(worker_code)

print("🌐 Step 3/3: Launching Worker Server & Cloudflare Tunnel...")
subprocess.Popen([sys.executable, "kaggle_worker.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

# Download cloudflared binary
subprocess.run(["wget", "-q", "-O", "cloudflared", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["chmod", "+x", "cloudflared"])

# Start cloudflared reverse tunnel
print("\\n✨ YOUR KAGGLE GPU REMOTE TUNNEL IS READY! Copy the URL below:")
print("=" * 65)
p = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(p.stdout.readline, ""):
    if "trycloudflare.com" in line:
        print(">>> TUNNEL URL:", line.strip())
'''

class KaggleAutomator:
    def __init__(self):
        self.tunnel_url: Optional[str] = None
        self.status: str = "disconnected"
        self.progress_msg: str = "Waiting for Remote Tunnel URL..."
        self.error: Optional[str] = None

    def register_tunnel(self, tunnel_url: str) -> Dict[str, Any]:
        url = tunnel_url.strip()
        if not url:
            return {"status": "error", "message": "Tunnel URL cannot be empty."}
        
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        self.tunnel_url = url
        self.status = "connected"
        self.progress_msg = f"Kaggle Remote Tunnel Connected: {self.tunnel_url}"
        return {
            "status": "success",
            "tunnel_url": self.tunnel_url,
            "message": "Remote Tunnel registered successfully!"
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "tunnel_url": self.tunnel_url,
            "progress_msg": self.progress_msg,
            "error": self.error
        }

    def get_notebook_script(self) -> str:
        return KAGGLE_NOTEBOOK_SCRIPT

kaggle_automator = KaggleAutomator()
