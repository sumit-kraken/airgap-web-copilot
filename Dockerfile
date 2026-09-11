# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (build tools, curl, and OpenMP for FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        trafilatura \
        sentence-transformers \
        faiss-cpu \
        ollama \
        pydantic \
        httpx

# Pre-cache SentenceTransformers weights for 100% offline air-gap deployment
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application source and static UI files
COPY src/ ./src/
COPY static/ ./static/
COPY main.py copilot.py extractor.py rag_engine.py ./

# Set Python path so airgap_web_copilot module is importable
ENV PYTHONPATH="/app/src:/app"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Production server entrypoint (binds to Render's dynamic $PORT, defaulting to 8000)
CMD ["sh", "-c", "uvicorn airgap_web_copilot.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
