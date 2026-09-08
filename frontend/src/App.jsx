import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, 
  Send, 
  Plus, 
  Trash2, 
  Cpu, 
  Globe, 
  CheckCircle2, 
  XCircle, 
  Sparkles, 
  Search, 
  Download, 
  MessageSquare,
  Zap,
  Play,
  Loader2,
  Terminal,
  Copy,
  Check,
  ExternalLink,
  Link2,
  RefreshCw,
  Shield,
  ArrowRight,
  PanelLeftClose,
  PanelLeft,
  Edit2,
  Image,
  Wand2,
  ZoomIn,
  Sliders,
  Eye,
  Maximize2
} from 'lucide-react';
import { marked } from 'marked';

const CLOUDFLARE_SCRIPT = `# 🚀 Kaggle T4 GPU Worker - Clean Output (1-Click Run)
import os, sys, time, subprocess, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LLAMA_CPP_LIB_VERBOSE"] = "0"

print("⚡ Installing CUDA wheels & GPU dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu125"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "sse-starlette", "huggingface_hub", "pydantic", "requests", "torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        raise HTTPException(status_code=400, detail="Model not ready")
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

print("🌐 Starting Worker & Cloudflare Tunnel...")
subprocess.Popen([sys.executable, "kaggle_worker.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

subprocess.run(["wget", "-q", "-O", "cloudflared", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["chmod", "+x", "cloudflared"])

print("\\n✨ COPY YOUR TUNNEL URL BELOW:")
print("=" * 60)
p = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(p.stdout.readline, ""):
    if "trycloudflare.com" in line:
        print(">>> TUNNEL URL:", line.strip())
`;

const PINGGY_SCRIPT = `# 🚀 Kaggle T4 GPU Worker - Pinggy Tunnel
import os, sys, time, subprocess, warnings
warnings.filterwarnings("ignore")

print("⚡ Installing GPU dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu125"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "sse-starlette", "huggingface_hub", "pydantic", "requests", "torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Start pinggy tunnel over SSH
print("\\n✨ Starting Pinggy Tunnel...")
p = subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", "-p", "443", "-R0:localhost:8000", "qr@a.pinggy.link"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(p.stdout.readline, ""):
    if "pinggy.link" in line:
        print(">>> TUNNEL URL:", line.strip())
`;

export default function App() {
  const [viewMode, setViewMode] = useState(() => {
    const savedMode = localStorage.getItem('view_mode');
    const savedUrl = localStorage.getItem('kaggle_tunnel_url');
    if (savedMode === 'studio' && savedUrl) {
      return 'studio';
    }
    return 'welcome';
  });
  
  const updateViewMode = (mode) => {
    setViewMode(mode);
    localStorage.setItem('view_mode', mode);
  };

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const [tunnelUrl, setTunnelUrl] = useState(() => localStorage.getItem('kaggle_tunnel_url') || '');
  const [tunnelStatus, setTunnelStatus] = useState({ online: false, gpu_info: '', error: '' });
  const [isCheckingTunnel, setIsCheckingTunnel] = useState(false);
  const [isEditingUrl, setIsEditingUrl] = useState(false);
  const [activeTunnelTab, setActiveTunnelTab] = useState('cloudflare');
  const [copiedCode, setCopiedCode] = useState(false);

  // Fast 1.1B GGUF Model (637 MB - downloads & loads in 3 seconds!)
  const [repoId, setRepoId] = useState('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF');
  const [discoveredFiles, setDiscoveredFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [isScanningHF, setIsScanningHF] = useState(false);
  const [modelStatus, setModelStatus] = useState({ status: 'idle', message: '', loaded_model: null, progress: null });
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  const [temperature, setTemperature] = useState(0.7);
  const messagesEndRef = useRef(null);

  // 🎨 Dual Studio: FLUX & SD Image Generation State
  const [studioMode, setStudioMode] = useState('llm'); // 'llm' | 'flux'
  const [imagePrompt, setImagePrompt] = useState('');
  const [imageNegativePrompt, setImageNegativePrompt] = useState('');
  const [imageModelId, setImageModelId] = useState('black-forest-labs/FLUX.1-schnell');
  const [imageWidth, setImageWidth] = useState(512);
  const [imageHeight, setImageHeight] = useState(512);
  const [imageSteps, setImageSteps] = useState(4);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [isLoadingImageModel, setIsLoadingImageModel] = useState(false);
  const [imageModelStatus, setImageModelStatus] = useState({ status: 'idle', message: '' });

  const loadImageModelToGPU = async () => {
    if (!imageModelId.trim()) return;
    setIsLoadingImageModel(true);
    setImageModelStatus({ status: 'loading', message: `Downloading & Loading ${imageModelId} onto Kaggle GPU VRAM...` });

    try {
      const resp = await fetch('/api/image/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: imageModelId })
      });
      const data = await resp.json();
      if (resp.ok && data.status === 'success') {
        setImageModelStatus({ status: 'ready', message: `Image Model ${imageModelId} Loaded on Kaggle GPU!` });
      } else {
        setImageModelStatus({ status: 'error', message: data.detail || data.error || 'Failed to load image model.' });
      }
    } catch (err) {
      setImageModelStatus({ status: 'error', message: err.message || 'Error connecting to Kaggle GPU' });
    } finally {
      setIsLoadingImageModel(false);
    }
  };

  const promptPresets = [
    "Cyberpunk neon metropolis with rainy reflections and flying cars, cinematic 8k",
    "Hyperrealistic close-up portrait of an astronaut floating in colorful nebula",
    "Cute 3D Pixar style baby dragon sitting on a pile of glowing crystals",
    "Epic fantasy landscape with floating islands, waterfalls, and sunset sky",
    "Isometric futuristic AI laboratory with glowing holographic displays"
  ];

  const handleGenerateImage = async () => {
    if (!imagePrompt.trim()) return;
    setIsGeneratingImage(true);
    setImageError('');

    try {
      const resp = await fetch('/api/image/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: imagePrompt,
          negative_prompt: imageNegativePrompt,
          model_id: imageModelId,
          width: imageWidth,
          height: imageHeight,
          num_inference_steps: imageSteps
        })
      });
      const data = await resp.json();
      if (resp.ok && data.status === 'success') {
        const newImg = {
          id: Date.now().toString(),
          url: data.image_url,
          prompt: imagePrompt,
          model: imageModelId,
          width: imageWidth,
          height: imageHeight,
          steps: imageSteps,
          timestamp: new Date().toLocaleTimeString()
        };
        setGeneratedImages(prev => [newImg, ...prev]);
      } else {
        setImageError(data.detail || data.error || 'Failed to generate image on Kaggle GPU');
      }
    } catch (err) {
      setImageError(err.message || 'Network error connecting to Kaggle GPU');
    } finally {
      setIsGeneratingImage(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetchSessions();
    if (tunnelUrl) {
      checkTunnelHealth(tunnelUrl);
    }
    scanHFRepo('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF');

    const handleResize = () => {
      if (window.innerWidth < 768) {
        setIsSidebarOpen(false);
      }
    };
    if (window.innerWidth < 768) {
      setIsSidebarOpen(false);
    }
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const fetchSessions = async () => {
    try {
      const resp = await fetch('/api/chat/sessions');
      const data = await resp.json();
      setSessions(data);
      if (data.length > 0 && !currentSessionId) {
        selectSession(data[0].id);
      } else if (data.length === 0) {
        createNewChat();
      }
    } catch (e) {
      console.error("Failed to load sessions:", e);
    }
  };

  const selectSession = async (sessionId) => {
    setCurrentSessionId(sessionId);
    try {
      const resp = await fetch(`/api/chat/sessions/${sessionId}/messages`);
      if (!resp.ok) {
        // Session not found (404), remove stale session and auto-create new conversation
        setSessions(prev => prev.filter(s => s.id !== sessionId));
        createNewChat();
        return;
      }
      const msgs = await resp.json();
      setMessages(msgs);
    } catch (e) {
      console.error("Failed to load messages:", e);
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      createNewChat();
    }
  };

  const createNewChat = async () => {
    try {
      const resp = await fetch('/api/chat/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Conversation' })
      });
      const session = await resp.json();
      setSessions(prev => [session, ...prev]);
      setCurrentSessionId(session.id);
      setMessages([]);
    } catch (e) {
      console.error("Failed to create chat:", e);
    }
  };

  const updateSessionTitle = async (sessionId, newTitle) => {
    try {
      await fetch(`/api/chat/sessions/${sessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle })
      });
      setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, title: newTitle } : s));
    } catch (e) {
      console.error("Failed to update session title:", e);
    }
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await fetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      if (currentSessionId === sessionId) {
        const remaining = sessions.filter(s => s.id !== sessionId);
        if (remaining.length > 0) {
          selectSession(remaining[0].id);
        } else {
          createNewChat();
        }
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const checkTunnelHealth = async (urlToCheck = tunnelUrl) => {
    if (!urlToCheck.trim()) {
      setTunnelStatus({ online: false, gpu_info: '', error: 'Please enter a Tunnel URL' });
      return false;
    }
    
    let cleanUrl = urlToCheck.trim();
    if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
      cleanUrl = `https://${cleanUrl}`;
      setTunnelUrl(cleanUrl);
    }

    setIsCheckingTunnel(true);
    setTunnelStatus({ online: false, gpu_info: '', error: 'Checking connection...' });
    localStorage.setItem('kaggle_tunnel_url', cleanUrl);
    
    try {
      const resp = await fetch('/api/model/kaggle-health', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tunnel_url: cleanUrl })
      });
      const data = await resp.json();
      if (data.status === 'online') {
        setTunnelStatus({ online: true, gpu_info: data.gpu_info || 'Tesla T4 Connected', error: '' });
        setIsEditingUrl(false);
        if (data.loaded_model?.filename) {
          setModelStatus({
            status: data.model_status,
            message: `Loaded: ${data.loaded_model.filename}`,
            loaded_model: data.loaded_model,
            progress: null
          });
        }
        return true;
      } else {
        setTunnelStatus({ online: false, gpu_info: '', error: data.error || 'Offline: Unable to ping Kaggle worker' });
        return false;
      }
    } catch (e) {
      setTunnelStatus({ online: false, gpu_info: '', error: 'Cannot connect to Tunnel URL' });
      return false;
    } finally {
      setIsCheckingTunnel(false);
    }
  };

  const handleConnectTunnel = async () => {
    const isOk = await checkTunnelHealth(tunnelUrl);
    if (isOk) {
      updateViewMode('studio');
    }
  };

  const copyScriptCode = (codeText) => {
    navigator.clipboard.writeText(codeText);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const scanHFRepo = async (targetRepo = repoId) => {
    if (!targetRepo.trim()) return;
    setIsScanningHF(true);
    try {
      const resp = await fetch(`/api/model/hf-files?repo=${encodeURIComponent(targetRepo.trim())}`);
      const data = await resp.json();
      if (data.files && data.files.length > 0) {
        setDiscoveredFiles(data.files);
        const q4 = data.files.find(f => f.quant.includes('Q4')) || data.files[0];
        setSelectedFile(q4.filename);
      }
    } catch (e) {
      console.error("HF scan error:", e);
    } finally {
      setIsScanningHF(false);
    }
  };

  const loadModelToGPU = async () => {
    if (!tunnelUrl.trim()) {
      alert("Please connect your Kaggle Remote Tunnel URL first!");
      return;
    }
    if (!selectedFile) {
      alert("Please select a GGUF quantization file first!");
      return;
    }

    setIsLoadingModel(true);
    setModelStatus({ 
      status: 'loading', 
      message: `⏳ Downloading GGUF file & offloading to Kaggle T4 GPU VRAM (${selectedFile})...`, 
      loaded_model: null,
      progress: null
    });

    const progressInterval = setInterval(async () => {
      try {
        const resp = await fetch('/api/model/kaggle-health', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tunnel_url: tunnelUrl.trim() })
        });
        const data = await resp.json();
        if (data.download_progress) {
          const p = data.download_progress;
          setModelStatus({
            status: 'loading',
            message: `⏳ Downloading GGUF: ${p.downloaded_gb} GB / ${p.total_gb} GB (${p.percent}%)`,
            progress: p,
            loaded_model: null
          });
        }
      } catch (e) {
        // ignore poll
      }
    }, 1500);

    try {
      const resp = await fetch('/api/model/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tunnel_url: tunnelUrl.trim(),
          repo_id: repoId.trim(),
          filename: selectedFile,
          n_ctx: 4096
        })
      });
      const data = await resp.json();
      if (data.status === 'success') {
        setModelStatus({
          status: 'ready',
          message: `✓ Model loaded on T4 GPU VRAM! Ready for chat.`,
          loaded_model: { repo_id: repoId, filename: selectedFile },
          progress: null
        });
      } else {
        const errDetail = data.error || data.detail || "Failed to load model on Kaggle GPU";
        setModelStatus({ 
          status: 'error', 
          message: `❌ ${errDetail}`, 
          loaded_model: null, 
          progress: null 
        });
      }
    } catch (e) {
      setModelStatus({ status: 'error', message: "Error communicating with Kaggle GPU server", loaded_model: null, progress: null });
    } finally {
      clearInterval(progressInterval);
      setIsLoadingModel(false);
    }
  };

  const sendMessage = async () => {
    if (!inputPrompt.trim() || isStreaming) return;
    if (!tunnelUrl.trim()) {
      alert("Please enter your Kaggle Tunnel URL in the sidebar!");
      return;
    }

    const userText = inputPrompt.trim();
    setInputPrompt('');
    setIsStreaming(true);

    // Auto-update conversation title if default
    const currentSession = sessions.find(s => s.id === currentSessionId);
    if (currentSession && (currentSession.title === 'New Conversation' || currentSession.title === 'New Chat Thread')) {
      const newTitle = userText.length > 24 ? userText.slice(0, 24) + '...' : userText;
      updateSessionTitle(currentSessionId, newTitle);
    }

    const userMsg = { role: 'user', content: userText };
    const tempAssistantMsg = { role: 'assistant', content: '' };
    setMessages(prev => [...prev, userMsg, tempAssistantMsg]);

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          content: userText,
          tunnel_url: tunnelUrl.trim(),
          temperature: temperature
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedReply = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value);
        const lines = chunkText.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const dataStr = line.slice(6).trim();
              const parsed = JSON.parse(dataStr);
              if (parsed.token) {
                accumulatedReply += parsed.token;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = { role: 'assistant', content: accumulatedReply };
                  return updated;
                });
              }
            } catch (err) {
              // Ignore partial JSON
            }
          }
        }
      }
    } catch (err) {
      console.error("Streaming error:", err);
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: 'assistant', content: '❌ Connection to Kaggle GPU failed or stream interrupted.' };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  if (viewMode === 'welcome') {
    return (
      <div style={{ display: 'flex', width: '100vw', height: '100vh', background: 'var(--bg-primary)', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ maxWidth: '660px', width: '90%', padding: '40px', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-lg)', textAlign: 'center', boxShadow: '0 12px 40px rgba(0,0,0,0.5)' }}>
          <div style={{ width: '80px', height: '80px', margin: '0 auto 24px auto', borderRadius: 'var(--radius-lg)', background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', boxShadow: '0 8px 24px var(--accent-glow)' }}>
            <Cpu size={42} />
          </div>

          <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: '2.2rem', fontWeight: 800, marginBottom: '12px', background: 'linear-gradient(90deg, #ffffff, #a5b4fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Run LLMs on Kaggle T4 GPU
          </h1>

          <p style={{ color: 'var(--text-muted)', fontSize: '1.05rem', lineHeight: '1.7', marginBottom: '28px' }}>
            Connect directly using <strong>Remote Tunneling</strong>. No Kaggle API secret key required!
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '32px', textAlign: 'left' }}>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <Shield size={20} style={{ color: '#10b981', marginBottom: '6px' }} />
              <div style={{ fontSize: '0.88rem', fontWeight: 700 }}>Zero Credentials</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>No API key secret needed.</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <Link2 size={20} style={{ color: '#a5b4fc', marginBottom: '6px' }} />
              <div style={{ fontSize: '0.88rem', fontWeight: 700 }}>Remote Tunneling</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Secure public URL bridge.</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <Zap size={20} style={{ color: '#06b6d4', marginBottom: '6px' }} />
              <div style={{ fontSize: '0.88rem', fontWeight: 700 }}>Free T4 GPU VRAM</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Runs Hugging Face GGUF.</div>
            </div>
          </div>

          <button 
            className="action-btn primary" 
            style={{ margin: '0 auto', padding: '14px 32px', fontSize: '1.05rem', borderRadius: 'var(--radius-md)' }}
            onClick={() => updateViewMode('tunnel_modal')}
          >
            <Play size={20} />
            Connect Kaggle Remote Tunnel
          </button>
        </div>
      </div>
    );
  }

  if (viewMode === 'tunnel_modal') {
    const currentScript = activeTunnelTab === 'cloudflare' ? CLOUDFLARE_SCRIPT : PINGGY_SCRIPT;

    return (
      <div style={{ display: 'flex', width: '100vw', height: '100vh', background: 'var(--bg-primary)', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ maxWidth: '680px', width: '92%', padding: '32px', background: 'var(--bg-card)', border: '1px solid var(--border-glow)', borderRadius: 'var(--radius-lg)', boxShadow: '0 12px 40px rgba(0,0,0,0.6)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Link2 size={24} style={{ color: '#a5b4fc' }} />
              <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', fontWeight: 700 }}>Connect Kaggle Remote Tunnel</h2>
            </div>
            <button className="action-btn" onClick={() => updateViewMode('welcome')} style={{ padding: '4px 10px', fontSize: '0.78rem' }}>
              Back
            </button>
          </div>

          <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
            Enter your public Cloudflare Tunnel URL (generated by running your Kaggle GPU worker notebook) to connect to your Kaggle T4 VRAM.
          </p>

          <div style={{ marginBottom: '24px' }}>
            <label style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
              Remote Tunnel URL
            </label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input 
                type="text" 
                className="hf-input" 
                placeholder="e.g. https://xxx.trycloudflare.com"
                value={tunnelUrl}
                onChange={(e) => setTunnelUrl(e.target.value)}
                style={{ flex: 1 }}
              />
              <button 
                className="action-btn primary" 
                onClick={handleConnectTunnel}
                disabled={isCheckingTunnel || !tunnelUrl.trim()}
              >
                {isCheckingTunnel ? (
                  <>
                    <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                    Verifying...
                  </>
                ) : (
                  <>
                    <Zap size={16} />
                    Connect & Open Studio
                  </>
                )}
              </button>
            </div>
          </div>

          {tunnelStatus.gpu_info && (
            <div style={{ fontSize: '0.85rem', padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'rgba(16, 185, 129, 0.15)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle2 size={16} /> Connected: {tunnelStatus.gpu_info}
            </div>
          )}
          {tunnelStatus.error && !isCheckingTunnel && (
            <div style={{ fontSize: '0.85rem', padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <XCircle size={16} /> {tunnelStatus.error}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {isSidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setIsSidebarOpen(false)} />
      )}
      <aside className={`sidebar ${!isSidebarOpen ? 'closed' : ''}`}>
        <div className="sidebar-header">
          <div className="logo-badge">
            <Cpu size={22} />
          </div>
          <div className="logo-text" style={{ flex: 1 }}>
            <h1>Kaggle GGUF Studio</h1>
            <span>Remote Tunneling</span>
          </div>
          <button 
            className="sidebar-toggle-btn" 
            onClick={() => setIsSidebarOpen(false)}
            title="Close sidebar"
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        <button className="new-chat-btn" onClick={createNewChat}>
          <Plus size={18} />
          New Chat Thread
        </button>

        <div className="session-list">
          {sessions.map(s => (
            <div
              key={s.id}
              className={`session-item ${currentSessionId === s.id ? 'active' : ''}`}
              onClick={() => selectSession(s.id)}
            >
              <div className="session-info">
                <MessageSquare size={16} />
                <span className="session-title">{s.title || 'Conversation'}</span>
              </div>
              <button className="delete-session-btn" onClick={(e) => deleteSession(s.id, e)}>
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        {/* Clean minimal Kaggle GPU status card */}
        <div className="tunnel-card">
          <div className="tunnel-header">
            <span>Kaggle GPU</span>
            <div className={`status-badge ${tunnelStatus.online ? 'online' : 'offline'}`}>
              <span className="status-dot"></span>
              {tunnelStatus.online ? 'Connected' : 'Offline'}
            </div>
          </div>

          {tunnelStatus.online && !isEditingUrl ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '2px' }}>
              <div style={{ fontSize: '0.78rem', color: '#10b981', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                <CheckCircle2 size={12} /> {tunnelStatus.gpu_info || 'Tesla T4 Connected'}
              </div>
              <button 
                onClick={() => setIsEditingUrl(true)} 
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.72rem', textDecoration: 'underline' }}
              >
                Change
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px' }}>
              <div className="tunnel-input-group">
                <input
                  type="text"
                  className="tunnel-input"
                  placeholder="https://xxx.trycloudflare.com"
                  value={tunnelUrl}
                  onChange={(e) => setTunnelUrl(e.target.value)}
                />
                <button className="tunnel-btn" onClick={() => checkTunnelHealth()} disabled={isCheckingTunnel}>
                  {isCheckingTunnel ? '...' : 'Check'}
                </button>
              </div>
              {tunnelStatus.online && (
                <button 
                  onClick={() => setIsEditingUrl(false)} 
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.72rem', alignSelf: 'flex-end' }}
                >
                  Cancel
                </button>
              )}
            </div>
          )}

          {isCheckingTunnel && (
            <div style={{ fontSize: '0.75rem', color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} /> Pinging Kaggle GPU...
            </div>
          )}

          {tunnelStatus.error && !tunnelStatus.online && !isCheckingTunnel && (
            <div style={{ fontSize: '0.75rem', color: '#ef4444', display: 'flex', alignItems: 'center', gap: '4px', wordBreak: 'break-word' }}>
              <XCircle size={12} /> {tunnelStatus.error}
            </div>
          )}
        </div>
      </aside>

      <main className="main-content">
        <header className="top-bar">
          <div className="model-bar-row">
            {!isSidebarOpen && (
              <button 
                className="topbar-toggle-btn" 
                onClick={() => setIsSidebarOpen(true)}
                title="Open Sidebar"
              >
                <PanelLeft size={20} />
              </button>
            )}

            {/* 🎨 Dual Studio Mode Switcher */}
            <div className="studio-mode-switcher">
              <button 
                className={`mode-btn ${studioMode === 'llm' ? 'active' : ''}`}
                onClick={() => setStudioMode('llm')}
              >
                <MessageSquare size={16} />
                <span>GGUF LLM Studio</span>
              </button>
              <button 
                className={`mode-btn ${studioMode === 'flux' ? 'active' : ''}`}
                onClick={() => setStudioMode('flux')}
              >
                <Wand2 size={16} />
                <span>FLUX Image Studio</span>
              </button>
            </div>

            {studioMode === 'llm' ? (
              <>
                <div className="input-container">
                  <Search className="input-icon" size={18} />
                  <input
                    type="text"
                    className="hf-input"
                    placeholder="Hugging Face Repo (e.g. TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)"
                    value={repoId}
                    onChange={(e) => setRepoId(e.target.value)}
                  />
                </div>

                <button className="action-btn" onClick={() => scanHFRepo(repoId)} disabled={isScanningHF}>
                  <Globe size={16} />
                  {isScanningHF ? 'Scanning HF...' : 'Scan GGUF Quants'}
                </button>

                <button 
                  className="action-btn primary" 
                  onClick={loadModelToGPU} 
                  disabled={isLoadingModel || !selectedFile}
                  title={!selectedFile ? "Please click 'Scan GGUF Quants' first to select a quantization level." : "Load model onto Kaggle GPU VRAM"}
                >
                  <Download size={16} />
                  {isLoadingModel ? 'Loading to GPU...' : 'Load to Kaggle GPU'}
                </button>
              </>
            ) : (
              <>
                <div className="input-container">
                  <Search className="input-icon" size={18} />
                  <input
                    type="text"
                    className="hf-input"
                    placeholder="Hugging Face Image Model (e.g. black-forest-labs/FLUX.1-schnell or stabilityai/stable-diffusion-xl-base-1.0)"
                    value={imageModelId}
                    onChange={(e) => setImageModelId(e.target.value)}
                  />
                </div>

                <button 
                  className="action-btn primary" 
                  onClick={loadImageModelToGPU} 
                  disabled={isLoadingImageModel || !imageModelId.trim()}
                  title="Load Image Model onto Kaggle GPU VRAM"
                >
                  <Download size={16} />
                  {isLoadingImageModel ? 'Loading to GPU...' : 'Load Image Model to Kaggle GPU'}
                </button>
              </>
            )}
          </div>

          {studioMode === 'llm' && discoveredFiles.length > 0 && (
            <div className="quants-pill-container">
              <span style={{ fontSize: '0.78rem', color: '#9ca3af', fontWeight: 600 }}>Quantizations:</span>
              {discoveredFiles.map(f => (
                <button
                  key={f.filename}
                  className={`quant-pill ${selectedFile === f.filename ? 'selected' : ''}`}
                  onClick={() => setSelectedFile(f.filename)}
                >
                  {f.quant} ({f.size_gb} GB)
                </button>
              ))}
            </div>
          )}

          {studioMode === 'llm' && modelStatus.message && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', padding: '8px 12px', background: 'rgba(0,0,0,0.4)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', marginTop: '4px' }}>
              <div style={{ fontSize: '0.85rem', color: modelStatus.status === 'ready' ? '#10b981' : modelStatus.status === 'error' ? '#ef4444' : '#a5b4fc', display: 'flex', alignItems: 'center', gap: '8px' }}>
                {isLoadingModel ? (
                  <Loader2 size={16} style={{ animation: 'spin 1s linear infinite', color: '#a5b4fc' }} />
                ) : modelStatus.status === 'ready' ? (
                  <CheckCircle2 size={16} style={{ color: '#10b981' }} />
                ) : (
                  <Sparkles size={16} />
                )}
                <span style={{ fontWeight: 600 }}>{modelStatus.message}</span>
              </div>

              {isLoadingModel && modelStatus.progress && (
                <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
                  <div style={{ width: `${modelStatus.progress.percent}%`, height: '100%', background: 'linear-gradient(90deg, #6366f1, #10b981)', transition: 'width 0.4s ease-in-out' }} />
                </div>
              )}
            </div>
          )}

          {studioMode === 'flux' && imageModelStatus.message && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', padding: '8px 12px', background: 'rgba(0,0,0,0.4)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', marginTop: '4px' }}>
              <div style={{ fontSize: '0.85rem', color: imageModelStatus.status === 'ready' ? '#10b981' : imageModelStatus.status === 'error' ? '#ef4444' : '#a5b4fc', display: 'flex', alignItems: 'center', gap: '8px' }}>
                {isLoadingImageModel ? (
                  <Loader2 size={16} style={{ animation: 'spin 1s linear infinite', color: '#a5b4fc' }} />
                ) : imageModelStatus.status === 'ready' ? (
                  <CheckCircle2 size={16} style={{ color: '#10b981' }} />
                ) : (
                  <Sparkles size={16} />
                )}
                <span style={{ fontWeight: 600 }}>{imageModelStatus.message}</span>
              </div>
            </div>
          )}
        </header>

        {studioMode === 'llm' ? (
          <>
            <div className="chat-canvas">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <div className="welcome-icon">
                    <Bot size={32} />
                  </div>
                  <h2>Run any GGUF Model on Kaggle T4 GPU</h2>
                  <p>
                    Select your GGUF quantization (`Q1`, `Q2_K`, `Q3_K_M`, `Q4_K_M`, `Q8_0`), load the model onto your Kaggle GPU, and enjoy multi-turn conversation memory!
                  </p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className={`msg-row ${msg.role}`}>
                    <div className="avatar">
                      {msg.role === 'user' ? 'U' : <Bot size={18} />}
                    </div>
                    <div className="msg-bubble">
                      <div 
                        className="msg-text"
                        dangerouslySetInnerHTML={{ __html: marked.parse(msg.content || '') }}
                      />
                      {msg.role === 'assistant' && !msg.content && isStreaming && idx === messages.length - 1 && (
                        <div style={{ display: 'flex', gap: '4px', padding: '4px 0' }}>
                          <span className="typing-dot"></span>
                          <span className="typing-dot" style={{ animationDelay: '0.2s' }}></span>
                          <span className="typing-dot" style={{ animationDelay: '0.4s' }}></span>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            <footer className="input-area">
              <div className="input-box-wrapper">
                <textarea
                  className="chat-input"
                  rows={1}
                  placeholder="Ask anything... (Multi-turn chat memory active)"
                  value={inputPrompt}
                  onChange={(e) => setInputPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                />
                <button className="send-btn" onClick={sendMessage} disabled={isStreaming || !inputPrompt.trim()}>
                  <Send size={18} />
                </button>
              </div>
            </footer>
          </>
        ) : (
          /* 🎨 FLUX & Stable Diffusion Image Studio Canvas */
          <div className="flux-studio-container">
            <div className="flux-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <Wand2 size={22} style={{ color: '#a5b4fc' }} />
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.4rem', fontWeight: 700 }}>
                  FLUX.1 & SD Image Generator
                </h2>
                <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: 'var(--radius-full)', background: 'rgba(16,185,129,0.15)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)', fontWeight: 600 }}>
                  Kaggle T4 GPU Accelerated
                </span>
              </div>

              <div style={{ display: 'flex', gap: '10px', marginBottom: '14px' }}>
                <input
                  type="text"
                  className="hf-input"
                  placeholder="Describe the image you want to generate... (e.g. Cyberpunk neon city in rain)"
                  value={imagePrompt}
                  onChange={(e) => setImagePrompt(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleGenerateImage(); }}
                  style={{ flex: 1, padding: '12px 16px', fontSize: '0.95rem' }}
                />
                <button 
                  className="action-btn primary"
                  onClick={handleGenerateImage}
                  disabled={isGeneratingImage || !imagePrompt.trim()}
                  style={{ padding: '12px 24px', fontSize: '0.95rem', borderRadius: 'var(--radius-md)' }}
                >
                  {isGeneratingImage ? (
                    <>
                      <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      Generate Image
                    </>
                  )}
                </button>
              </div>

              {/* Preset Prompts */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>Ideas:</span>
                {promptPresets.map((preset, pIdx) => (
                  <button 
                    key={pIdx} 
                    className="flux-preset-pill"
                    onClick={() => setImagePrompt(preset)}
                  >
                    {preset.slice(0, 42)}...
                  </button>
                ))}
              </div>

              {/* FLUX Controls Row */}
              <div className="flux-controls-row">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Model:</label>
                  <select 
                    className="flux-select" 
                    value={imageModelId}
                    onChange={(e) => setImageModelId(e.target.value)}
                  >
                    <option value="black-forest-labs/FLUX.1-schnell">FLUX.1 schnell (Fast 4-Step)</option>
                    <option value="stabilityai/stable-diffusion-xl-base-1.0">Stable Diffusion XL 1.0</option>
                    <option value="runwayml/stable-diffusion-v1-5">Stable Diffusion 1.5</option>
                  </select>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Resolution:</label>
                  <select 
                    className="flux-select"
                    value={`${imageWidth}x${imageHeight}`}
                    onChange={(e) => {
                      const [w, h] = e.target.value.split('x').map(Number);
                      setImageWidth(w);
                      setImageHeight(h);
                    }}
                  >
                    <option value="512x512">512 x 512 (Square Fast)</option>
                    <option value="768x768">768 x 768 (HD Square)</option>
                    <option value="1024x1024">1024 x 1024 (Ultra HD)</option>
                  </select>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Inference Steps:</label>
                  <input
                    type="number"
                    className="flux-input-num"
                    min={1}
                    max={50}
                    value={imageSteps}
                    onChange={(e) => setImageSteps(Number(e.target.value))}
                    style={{ width: '70px' }}
                  />
                </div>
              </div>

              {imageError && (
                <div style={{ marginTop: '12px', padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', color: '#ef4444', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <XCircle size={16} /> {imageError}
                </div>
              )}
            </div>

            {/* Generated Image Gallery */}
            <div style={{ marginTop: '10px' }}>
              <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.15rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Image size={18} style={{ color: '#a5b4fc' }} />
                Generated Artwork Gallery ({generatedImages.length})
              </h3>

              {generatedImages.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '60px 20px', border: '2px dashed var(--border-subtle)', borderRadius: 'var(--radius-lg)', color: 'var(--text-muted)' }}>
                  <Wand2 size={40} style={{ margin: '0 auto 12px auto', opacity: 0.4 }} />
                  <p style={{ fontSize: '1rem', fontWeight: 600 }}>No artwork generated yet</p>
                  <p style={{ fontSize: '0.85rem' }}>Enter a prompt above and click "Generate Image" to create artwork on Kaggle GPU!</p>
                </div>
              ) : (
                <div className="image-gallery-grid">
                  {generatedImages.map((img) => (
                    <div key={img.id} className="image-card">
                      <img src={img.url} alt={img.prompt} />
                      <div className="image-card-overlay">
                        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'white', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          "{img.prompt}"
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          <span>{img.width}x{img.height} • {img.steps} steps</span>
                          <span>{img.timestamp}</span>
                        </div>
                        <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
                          <button 
                            className="action-btn"
                            style={{ flex: 1, padding: '5px 8px', fontSize: '0.75rem' }}
                            onClick={() => setActiveZoomImage(img)}
                          >
                            <ZoomIn size={14} /> Zoom
                          </button>
                          <a 
                            href={img.url} 
                            download={`flux-art-${img.id}.png`}
                            className="action-btn primary"
                            style={{ flex: 1, padding: '5px 8px', fontSize: '0.75rem', textDecoration: 'none', textAlign: 'center' }}
                          >
                            <Download size={14} /> Download
                          </a>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Lightbox Zoom Modal */}
            {activeZoomImage && (
              <div className="lightbox-overlay" onClick={() => setActiveZoomImage(null)}>
                <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
                  <img 
                    src={activeZoomImage.url} 
                    alt={activeZoomImage.prompt} 
                    style={{ maxWidth: '100%', maxHeight: '75vh', objectFit: 'contain', background: '#000' }} 
                  />
                  <div style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--border-subtle)' }}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'white' }}>"{activeZoomImage.prompt}"</div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                        Model: {activeZoomImage.model} • Resolution: {activeZoomImage.width}x{activeZoomImage.height}
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: '10px' }}>
                      <a 
                        href={activeZoomImage.url} 
                        download={`flux-art-${activeZoomImage.id}.png`}
                        className="action-btn primary"
                        style={{ padding: '8px 16px', fontSize: '0.85rem', textDecoration: 'none' }}
                      >
                        <Download size={16} /> Download PNG
                      </a>
                      <button className="action-btn" onClick={() => setActiveZoomImage(null)}>
                        Close
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
