# Airgap Web Copilot 🛡️⚡

An air-gapped, zero-data-egress hybrid AI assistant for analyzing web pages, articles, and documents completely offline.

It combines:
1. **Client-Side WebGPU (WebLLM)**: Runs language models (Gemma 2, Qwen 2.5, Llama 3) directly inside your browser tab on your GPU with zero server compute.
2. **Localhost Ollama Backend**: Seamless fallback to a local Ollama daemon on your PC when documents exceed browser context windows or for headless execution.
3. **Local Python RAG Engine**: Cleans noisy web pages using **Trafilatura**, generates dense vector embeddings via **SentenceTransformers**, and indexes chunks into an in-memory **FAISS** index.

---

## Quick Start

### 1. Run the Web Copilot
```powershell
# Windows (PowerShell)
.\.venv\Scripts\python.exe main.py
```
```bash
# macOS / Linux
./.venv/bin/python main.py
```

Once started, open your browser to:
- **Web App**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Getting and Running Ollama (Step-by-Step)

If you do not have Ollama installed or want to use the local background engine:

### 1. Download & Install Ollama

- **Windows**:
  1. Download the Windows installer from [https://ollama.com/download/OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe).
  2. Run `OllamaSetup.exe` and follow the on-screen installer.
  3. Once installed, Ollama starts automatically and runs in your Windows System Tray (near the clock).

- **macOS**:
  1. Download the Mac zip from [https://ollama.com/download](https://ollama.com/download) or run:
     ```bash
     brew install ollama
     ```
  2. Move `Ollama.app` to your Applications folder and launch it.

- **Linux**:
  Run the official one-line install script:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

---

### 2. Verify Ollama is Running

Open a terminal and check:
```bash
ollama --version
```
By default, the Ollama daemon listens on `http://localhost:11434`. You can test connectivity by running:
```bash
curl http://localhost:11434/api/tags
```
If you receive a JSON response (e.g. `{"models": [...]}`), Ollama is ready!

---

### 3. Download (Pull) a Model

Ollama needs at least one model downloaded to generate responses. Run any of the following commands in your terminal:

```bash
# Recommended: Fast, high-quality, lightweight 3B model
ollama pull llama3.2

# Google Gemma 2 (2B parameters)
ollama pull gemma2:2b

# Qwen 2.5 (7B parameters, higher reasoning capability)
ollama pull qwen2.5:7b
```

The copilot automatically detects whichever model is installed in Ollama. If you download `llama3.2`, the copilot will automatically select it!

---

### 4. Running the Ollama Server Manually

If Ollama is ever stopped or not running in your system tray:
```bash
ollama serve
```

---

## In-Browser WebGPU (Zero-Install Alternative)

If you don't want to install Ollama at all, you can run entirely within your web browser using **WebGPU**:

1. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in **Google Chrome** or **Microsoft Edge**.
2. Make sure hardware acceleration is enabled (`Settings -> System -> Use graphics acceleration when available`).
3. The app will detect your GPU (e.g., `WebGPU: blackwell` or `WebGPU: Ready`).
4. Click **`🌐 Force WebGPU`** in the header.
5. Ask any question. The browser will download the model weights (e.g., `gemma-2-2b-it-q4f16_1-MLC`) directly into your browser's local cache on the first run and execute 100% on your GPU shaders.

---

## Configuring Models

### In-Browser WebGPU Model
Edit `static/index.html` around line 995:
```javascript
const state = {
  webllmModel: "gemma-2-2b-it-q4f16_1-MLC", // Change to Qwen2.5-1.5B-Instruct-q4f16_1-MLC, Llama-3.2-1B-Instruct-q4f16_1-MLC, etc.
};
```

### Backend Ollama Model
Pull any model with `ollama pull <model-name>`. The backend automatically selects your active installed model.

---

## Agentic Mesh: LLM-as-a-Judge Architecture 🛡️⚖️

AirGap Web Copilot features a dual-layer **Agentic Consensus Mesh** that detects hallucinations and verifies factual grounding without introducing user-facing latency:

1. **Layer 1 (Immediate Drafter)**:
   - Inference runs via in-browser **WebGPU** or local **Ollama** and streams tokens to the chat interface with instantaneous First-Token Latency (TTFT) and high throughput (tok/s).
2. **Layer 2 (Asynchronous Critic / Judge)**:
   - Immediately upon stream completion, the drafted answer and retrieved FAISS document context are dispatched to `POST /api/verify`.
   - A local Judge model audits factual claims against the original document context, outputting structured JSON with:
     - **Status**: `verified` (score ≥ 0.75), `flagged` (score < 0.75), or `inconclusive`.
     - **Score**: 0.0 to 1.0 groundedness index.
     - **Verdict**: Concise 1-sentence audit summary.
     - **Supported Claims**: Contextually verified statements.
     - **Unsupported Claims**: Extrapolations or hallucinations detected.
3. **Interactive UI Badge & Drawer**:
   - Live pulsating mesh badge (`🛡️ Mesh Judge: Auditing groundedness...`) automatically resolves to a green verified badge (`🛡️ Grounded: Verified (95%) • llama3.2`) or an alert badge (`⚠️ Hallucination Alert: Flagged`).
   - Clicking the badge opens an interactive drawer breaking down supported vs unverified claims.

---

## Deployment Guide 🚀

### Method 1: Docker Compose (Recommended)

Run both the AirGap Copilot and an Ollama instance with a single command:

```bash
docker compose up -d --build
```

- Access the Copilot UI at: `http://localhost:8000`
- The `ollama_models` Docker volume persists downloaded weights across container restarts.
- To download a model into the containerized Ollama:
  ```bash
  docker exec -it airgap-ollama ollama pull llama3.2
  ```

---

### Method 2: Local Workstation / Air-Gapped Machine

If deploying directly on an air-gapped machine without Docker:

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e .
   ```
2. Run in production mode with Uvicorn:
   ```powershell
   uvicorn airgap_web_copilot.main:app --host 0.0.0.0 --port 8000 --workers 1
   ```

#### Linux Systemd Service:
Create `/etc/systemd/system/airgap-copilot.service`:
```ini
[Unit]
Description=AirGap Web Copilot Service
After=network.target

[Service]
Type=simple
User=copilot
WorkingDirectory=/opt/airgap-web-copilot
ExecStart=/opt/airgap-web-copilot/.venv/bin/uvicorn airgap_web_copilot.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=OLLAMA_HOST=http://localhost:11434

[Install]
WantedBy=multi-user.target
```
Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now airgap-copilot
```

---

### Critical: WebGPU & HTTPS Requirement for Remote Clients

> [!IMPORTANT]
> **WebGPU requires a Secure Context (`https://` or `http://localhost`)**.
> - If users access the copilot on `http://localhost:8000`, in-browser WebGPU works out of the box.
> - If users access the copilot across a Local Area Network (e.g. `http://192.168.1.50:8000`) or over the internet, browsers disable WebGPU unless the site is served over **HTTPS**.
> - To enable WebGPU for remote network clients, place a reverse proxy like **Caddy** (automatic HTTPS) or **Nginx** in front of port 8000.

#### Example Caddyfile (Automatic HTTPS):
```caddy
copilot.yourdomain.com {
    reverse_proxy localhost:8000
}
```

---

## Running Automated Tests

To run the complete test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest
```
