# 🚀 Kaggle T4 GPU Dual Studio (LLMs + FLUX.1 Image AI)

> **Run GGUF Text LLMs & FLUX.1 AI Image Generation on this UI — powered by free Kaggle T4 GPU (16GB VRAM)!**  
> *Run any Hugging Face GGUF text model or generate FLUX.1 / Stable Diffusion AI images on the web UI without extra local GPU setup or API key secrets.*

---

## 🌟 Why Kaggle GPU Dual Studio?

Running open-source LLMs and AI Image Generators locally often requires expensive high-VRAM NVIDIA GPUs. **Kaggle GPU Dual Studio** allows you to run GGUF text models and FLUX.1 image generation on a sleek Web UI while offloading all GPU computation to Kaggle's free **Tesla T4 GPU (16GB VRAM)** via secure remote tunneling. 

---

## ✨ Key Features

- 💬 **GGUF Text LLM Studio**: Stream LLM chat responses (Llama 3, Qwen 2.5, DeepSeek R1, Mistral) locally in your browser.
- 🎨 **FLUX.1 & SD Image Studio**: Generate high-resolution AI artwork using `FLUX.1-schnell` and `Stable Diffusion XL` directly on Kaggle GPU.
- ⚡ **Free Kaggle T4 GPU Acceleration**: Offload heavy LLM loading and FLUX image diffusion onto free 16GB Kaggle GPU VRAM.
- 🧠 **Auto Hugging Face Quant Discovery**: Type any Hugging Face GGUF repo ID (e.g. `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`) and automatically discover all available quantization options.
- 🚀 **Zero Credentials / Secret Keys**: Connect securely via **Cloudflare Remote Tunneling** without managing API keys or secrets.
- 🖼️ **Image Lightbox & 1-Click HD Download**: Fullscreen image preview modal with prompt metadata and direct PNG downloads.
- 💬 **Multi-Turn Conversation Memory**: Persistent SQLite chat history with dynamic thread titles and session management.

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────┐         ┌─────────────────────────┐
│                         │  HTTP   │                         │
│  ReactJS Local Web UI   │ ──────> │  FastAPI Central Server │
│  (Port 3000)            │         │  (Port 5000)            │
└─────────────────────────┘         └────────────┬────────────┘
                                                 │
                                                 │ HTTPS Tunnel
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 Cloudflare Remote Tunnel                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Kaggle T4 GPU Remote Worker                 │
│                 (llama-cpp-python + CUDA)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Setup Guide

### 1️⃣ Start Local Web Studio

#### **Terminal 1: Start Central Backend**
```bash
cd llm_gguf
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --port 5000 --reload
```

#### **Terminal 2: Start React Frontend**
```bash
cd llm_gguf/frontend
npm install
npm run dev
```

Open your browser at **`http://localhost:3000`**.

---

### 2️⃣ Kaggle Notebook Execution (2 Minutes)

1. Open [Kaggle.com](https://www.kaggle.com) -> Create **New Dataset** -> Upload `kaggle_worker.py`.
2. Create a **New Notebook** on Kaggle, attach your `kaggle-worker` dataset, and in **Settings** set:
   - **Accelerator**: `GPU T4 x2` (or P100)
   - **Internet**: `On`
3. Run the following cells in your notebook:

#### **Cell 1: Copy file to working directory**
```bash
!cp /kaggle/input/datasets/username/folder_name/kaggle_worker.py -d /kaggle/working
```

#### **Cell 2: Change working directory**
```bash
%cd /kaggle/working
```

#### **Cell 3: Install CUDA Llama Wheels & Worker Dependencies**
```bash
!pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
!pip install -q fastapi uvicorn sse-starlette huggingface_hub pydantic requests torch
```

#### **Cell 4: Launch Worker Server & Tunnel**
```bash
!python kaggle_worker.py
```

---

### 3️⃣ Connect & Chat!

1. Copy the generated Tunnel URL from Cell 4 output:
   ```text
   ✨ YOUR KAGGLE GPU REMOTE TUNNEL IS READY!
   ============================================================
   >>> TUNNEL URL: https://random-name.trycloudflare.com
   ```
2. Open `http://localhost:3000`, click **Connect Kaggle Remote Tunnel**, paste your URL, and click **Connect & Open Studio**.
3. Pick any Hugging Face GGUF repo, select a quantization pill (`Q4_K_M`), click **Load to Kaggle GPU**, and start chatting locally!

---

## 📁 Repository Structure

```text
llm_gguf/
├── kaggle_worker.py               # Standalone Kaggle T4 GPU Worker & Cloudflare Tunnel
├── README.md                      # Complete setup & user guide
├── backend/
│   ├── main.py                    # FastAPI server entry point
│   ├── database.py                # SQLite chat memory DB
│   ├── requirements.txt           # Python dependencies
│   ├── routes/
│   │   ├── chat_routes.py         # Session CRUD & streaming endpoints
│   │   ├── model_routes.py        # HF discovery & model loading endpoints
│   │   └── kaggle_routes.py       # Tunnel status endpoints
│   └── services/
│       ├── hf_service.py          # Hugging Face GGUF quant scanner
│       ├── kaggle_api_service.py  # Tunnel status & script generator
│       └── kaggle_client.py       # Tunnel HTTP client & SSE streaming
└── frontend/
    ├── package.json               # React dependencies
    ├── vite.config.js             # Vite configuration
    ├── index.html                 # HTML shell
    └── src/
        ├── App.jsx                # Interactive Studio & Chat UI
        ├── main.jsx               # React entry point
        └── index.css              # Glassmorphism styling system
```

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><strong>Q: Do I need a local NVIDIA GPU?</strong></summary>
No! All heavy matrix calculations and GGUF CUDA inference run on free Kaggle T4 GPUs, while you stream the responses locally on your UI.
</details>

<details>
<summary><strong>Q: Is an API Secret Key required?</strong></summary>
No credentials or secret API keys are stored or needed. The connection is established via secure Cloudflare HTTPS tunneling.
</details>

<details>
<summary><strong>Q: Which GGUF models are supported?</strong></summary>
Any GGUF repo hosted on Hugging Face! (Examples: <code>TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF</code>, <code>bartowski/Llama-3.2-3B-Instruct-GGUF</code>, <code>TheBloke/Mistral-7B-Instruct-v0.2-GGUF</code>, <code>Qwen/Qwen2.5-7B-Instruct-GGUF</code>, etc.).
</details>

---

## 📜 License

Distributed under the **MIT License**. Free to use, modify, and distribute for personal or commercial projects.
