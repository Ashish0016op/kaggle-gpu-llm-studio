"""
Kaggle T4 GPU Remote Tunnel Worker Server for GGUF LLM & FLUX Image Inference
Runs inside Kaggle notebook environment with GPU accelerator enabled.
Exposes a FastAPI server over Cloudflare reverse tunnel.
"""

import os
import sys
import time
import subprocess
import asyncio
import warnings
import io
import base64
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LLAMA_CPP_LIB_VERBOSE"] = "0"

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return True, f"{gpu_name} ({total_mem:.2f} GB VRAM)"
    except Exception:
        pass
    return False, "CPU Only"

app = FastAPI(title="Kaggle Remote LLM & FLUX GPU Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_state = {
    "llm": None,
    "repo_id": None,
    "filename": None,
    "n_ctx": 4096,
    "loaded_at": None,
    "status": "idle", # idle, loading, ready, error
    "error": None,
    "target_file_size": 0
}

image_state = {
    "pipe": None,
    "model_id": None,
    "status": "idle",
    "error": None
}

class LoadModelRequest(BaseModel):
    repo_id: str
    filename: str
    n_ctx: Optional[int] = 4096

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 1024
    reasoning: Optional[str] = "off"

class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    model_id: Optional[str] = "black-forest-labs/FLUX.1-schnell"
    width: Optional[int] = 512
    height: Optional[int] = 512
    num_inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 3.5

def get_download_progress():
    """Calculate current downloaded GGUF file size from Hugging Face hub cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.exists(cache_dir):
        return None

    downloaded_bytes = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".gguf") or f.endswith(".incomplete") or "blobs" in root:
                try:
                    fpath = os.path.join(root, f)
                    downloaded_bytes += os.path.getsize(fpath)
                except Exception:
                    pass

    total_bytes = model_state["target_file_size"] or (14 * 1024 ** 3)
    downloaded_gb = round(downloaded_bytes / (1024 ** 3), 2)
    total_gb = round(total_bytes / (1024 ** 3), 2)
    percent = min(round((downloaded_bytes / total_bytes) * 100, 1), 99.9) if total_bytes else 0

    return {
        "downloaded_bytes": downloaded_bytes,
        "total_bytes": total_bytes,
        "downloaded_gb": downloaded_gb,
        "total_gb": total_gb,
        "percent": percent
    }

class LoadImageModelRequest(BaseModel):
    model_id: str

@app.get("/health")
def get_health():
    gpu_ok, gpu_info = check_gpu()
    progress = get_download_progress() if model_state["status"] == "loading" else None
    return {
        "status": "online",
        "gpu_available": gpu_ok,
        "gpu_info": gpu_info,
        "model_status": model_state["status"],
        "download_progress": progress,
        "loaded_model": {
            "repo_id": model_state["repo_id"],
            "filename": model_state["filename"],
            "n_ctx": model_state["n_ctx"],
            "loaded_at": model_state["loaded_at"],
        },
        "image_state": {
            "status": image_state["status"],
            "model_id": image_state["model_id"],
            "error": image_state["error"]
        },
        "error": model_state["error"]
    }

@app.post("/load-image-model")
async def load_image_model(req: LoadImageModelRequest):
    global image_state
    
    image_state["status"] = "loading"
    image_state["error"] = None
    image_state["model_id"] = req.model_id
    
    print(f"⏳ Downloading & Loading Image Model onto GPU: {req.model_id}")

    def _do_load():
        import torch
        from diffusers import AutoPipelineForText2Image

        if image_state["pipe"] is not None:
            del image_state["pipe"]
            image_state["pipe"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        pipe = AutoPipelineForText2Image.from_pretrained(
            req.model_id,
            torch_dtype=dtype,
            safety_checker=None
        )
        pipe.to("cuda")
        image_state["pipe"] = pipe
        image_state["status"] = "ready"
        return {"status": "success", "message": f"Loaded Image Model {req.model_id} onto GPU VRAM!"}

    try:
        res = await asyncio.to_thread(_do_load)
        return res
    except Exception as e:
        image_state["status"] = "error"
        image_state["error"] = str(e)
        print(f"❌ Error loading image model: {e}")
        return {"status": "error", "error": f"Failed to load image model: {str(e)}"}

@app.post("/load")
async def load_model(req: LoadModelRequest):
    global model_state

    model_state["status"] = "loading"
    model_state["error"] = None
    model_state["filename"] = req.filename
    model_state["repo_id"] = req.repo_id

    print(f"⏳ Downloading & loading GGUF from HF: {req.repo_id} / {req.filename}")

    def _non_blocking_load():
        from llama_cpp import Llama
        if model_state["llm"] is not None:
            del model_state["llm"]
            model_state["llm"] = None

        llm = Llama.from_pretrained(
            repo_id=req.repo_id,
            filename=req.filename,
            n_gpu_layers=-1,
            n_ctx=req.n_ctx or 4096,
            verbose=False
        )

        model_state["llm"] = llm
        model_state["n_ctx"] = req.n_ctx or 4096
        model_state["loaded_at"] = time.time()
        model_state["status"] = "ready"
        return {"status": "success", "message": f"Loaded model {req.filename} on GPU"}

    try:
        result = await asyncio.to_thread(_non_blocking_load)
        return result
    except Exception as e:
        model_state["status"] = "error"
        model_state["error"] = str(e)
        print(f"❌ Error loading model: {e}")
        return {"status": "error", "error": f"Failed to load model: {str(e)}"}

@app.post("/chat/stream")
async def chat_stream(req: ChatCompletionRequest):
    if model_state["status"] != "ready" or model_state["llm"] is None:
        raise HTTPException(status_code=400, detail="No model loaded on Kaggle GPU. Load a model first.")

    raw_messages = [msg.dict() for msg in req.messages]

    kwargs = {
        "messages": raw_messages,
        "max_tokens": req.max_tokens or 1024,
        "stream": True,
    }
    if req.temperature and req.temperature > 0:
        kwargs.update(temperature=req.temperature, top_p=req.top_p or 0.95, top_k=req.top_k or 40)
    else:
        kwargs.update(temperature=0.0)

    async def token_generator():
        llm = model_state["llm"]
        try:
            for chunk in llm.create_chat_completion(**kwargs):
                choice = chunk["choices"][0]
                delta = choice.get("delta", {}).get("content", "")
                finish_reason = choice.get("finish_reason", None)
                data = json.dumps({"token": delta, "finish_reason": finish_reason})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.001)
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.post("/generate-image")
async def generate_image(req: ImageGenRequest):
    """Generate image using diffusers (FLUX.1 / Stable Diffusion) on Kaggle GPU."""
    global image_state
    
    target_model = req.model_id or "black-forest-labs/FLUX.1-schnell"

    def _generate():
        import torch
        from diffusers import AutoPipelineForText2Image

        if image_state["pipe"] is None or image_state["model_id"] != target_model:
            print(f"🎨 Loading Image Generation Model: {target_model}")
            image_state["status"] = "loading"
            
            # Use bfloat16 or float16 for fast GPU VRAM efficiency
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            pipe = AutoPipelineForText2Image.from_pretrained(
                target_model,
                torch_dtype=dtype,
                safety_checker=None
            )
            pipe.to("cuda")
            image_state["pipe"] = pipe
            image_state["model_id"] = target_model
            image_state["status"] = "ready"

        pipe = image_state["pipe"]
        steps = req.num_inference_steps or (4 if "schnell" in target_model else 25)
        
        generator = torch.Generator("cuda").manual_seed(int(time.time()))
        
        print(f"🖼️ Generating image for prompt: '{req.prompt}'...")
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt if req.negative_prompt else None,
            width=req.width or 512,
            height=req.height or 512,
            num_inference_steps=steps,
            guidance_scale=req.guidance_scale or 3.5,
            generator=generator
        )
        
        img = result.images[0]
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "image_url": f"data:image/png;base64,{img_str}",
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "model_id": target_model
        }

    try:
        res = await asyncio.to_thread(_generate)
        return res
    except Exception as e:
        image_state["status"] = "error"
        image_state["error"] = str(e)
        print(f"❌ Image generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import threading

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    time.sleep(2)
    print("🌐 Launching Cloudflare Remote Tunnel...")

    if not os.path.exists("cloudflared"):
        subprocess.run(["wget", "-q", "-O", "cloudflared", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["chmod", "+x", "cloudflared"])

    print("\n✨ YOUR KAGGLE GPU REMOTE TUNNEL IS READY!")
    print("=" * 60)
    p = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(p.stdout.readline, ""):
        if "trycloudflare.com" in line:
            print(">>> TUNNEL URL:", line.strip())
