"""FastAPI application for AirGap Hybrid Copilot."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from airgap_web_copilot.copilot import (
    check_ollama_status,
    evaluate_groundedness,
    stream_rag_response,
)
from airgap_web_copilot.extractor import extract_content
from airgap_web_copilot.rag_engine import RAGEngine

# Initialize FastAPI application
app = FastAPI(
    title="AirGap Hybrid Copilot API",
    description="Offline Hybrid WebGPU & Localhost Ollama API for web document analysis",
    version="0.1.0",
)

# Enable CORS for local client-side WebGPU operations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory RAG engine instance
rag_engine = RAGEngine()


class ExtractContextRequest(BaseModel):
    raw_html: Optional[str] = Field(default=None, description="Raw HTML markup to extract")
    html: Optional[str] = Field(default=None, description="Alternative raw HTML input")
    url: Optional[str] = Field(default=None, description="URL to fetch and extract")
    query: Optional[str] = Field(default=None, description="Optional search query for chunk ranking")
    top_k: int = Field(default=4, ge=1, le=20, description="Top-k chunks to return")


class AnalyzeRequest(BaseModel):
    query: str = Field(description="User prompt or question to answer")
    raw_html: Optional[str] = Field(default=None, description="Raw HTML markup to analyze")
    html: Optional[str] = Field(default=None, description="Alternative raw HTML input")
    url: Optional[str] = Field(default=None, description="URL to fetch and extract")
    context: Optional[str] = Field(default=None, description="Pre-extracted plain text context")
    model: str = Field(default="qwen2.5:7b", description="Ollama model identifier")
    host: Optional[str] = Field(default=None, description="Optional Ollama daemon host URL")
    top_k: int = Field(default=4, ge=1, le=20, description="Top-k chunks to retrieve")


class VerifyRequest(BaseModel):
    query: str = Field(description="Original user prompt or question")
    context: str = Field(description="Document context or retrieved chunks used for the answer")
    answer: str = Field(description="Answer drafted by Layer 1 model to evaluate")
    model: Optional[str] = Field(default=None, description="Optional Ollama judge model")
    host: Optional[str] = Field(default=None, description="Optional Ollama daemon host URL")


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health status check and Ollama daemon connectivity reporting."""
    ollama_info = await check_ollama_status()
    return {
        "status": "ok",
        "service": "AirGap Hybrid Copilot",
        "ollama": ollama_info,
    }


@app.post("/api/extract-context")
async def extract_context_endpoint(payload: ExtractContextRequest) -> Dict[str, Any]:
    """Extract clean text and generate vector chunks for WebGPU or offline analysis."""
    input_source = payload.raw_html or payload.html or payload.url

    # If new input is provided, extract and rebuild the FAISS index
    if input_source and input_source.strip():
        try:
            extracted = extract_content(input_source)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

        extracted_text = extracted["text"]
        title = extracted.get("title", "Web Document")

        # Index document in the in-memory FAISS engine
        indexed_chunks = rag_engine.build_index(extracted_text)
        rag_engine.active_title = title
        rag_engine.active_text = extracted_text
    elif rag_engine.chunks:
        # Re-use existing active document
        extracted_text = getattr(rag_engine, "active_text", "\n\n".join(rag_engine.chunks))
        title = getattr(rag_engine, "active_title", "Web Document")
        indexed_chunks = rag_engine.chunks
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing required content. Provide 'html', 'raw_html', or 'url'.",
        )

    # If a query is provided, retrieve top-k relevant chunks
    if payload.query and payload.query.strip():
        retrieved_chunks = rag_engine.retrieve(payload.query, top_k=payload.top_k)
        if retrieved_chunks:
            context_str = "\n\n".join(c["text"] for c in retrieved_chunks)
            response_chunks = retrieved_chunks
        else:
            fallback_chunks = indexed_chunks[:payload.top_k] if indexed_chunks else []
            context_str = "\n\n".join(fallback_chunks) if fallback_chunks else extracted_text[:4000]
            response_chunks = [{"text": c, "score": 1.0, "chunk_index": i} for i, c in enumerate(fallback_chunks)]
    else:
        sample_chunks = indexed_chunks[:payload.top_k] if indexed_chunks else []
        context_str = "\n\n".join(sample_chunks) if sample_chunks else extracted_text[:4000]
        response_chunks = [
            {"text": c, "score": 1.0, "chunk_index": i}
            for i, c in enumerate(sample_chunks)
        ]

    # Context window guard (max 8000 chars)
    if len(context_str) > 8000:
        context_str = context_str[:8000]

    return {
        "status": "success",
        "context": context_str,
        "chunks": response_chunks,
        "char_count": len(extracted_text),
        "title": title,
    }


@app.post("/api/analyze")
async def analyze_endpoint(payload: AnalyzeRequest) -> StreamingResponse:
    """Analyze webpage content with local Ollama fallback using Server-Sent Events (SSE)."""
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string is required.")

    # Determine context and chunks
    context_str = ""
    chunks_to_cite: List[Dict[str, Any]] = []

    if payload.context and payload.context.strip():
        context_str = payload.context.strip()
    elif rag_engine.chunks and not (payload.raw_html or payload.html or payload.url):
        # Query active indexed document directly
        retrieved = rag_engine.retrieve(payload.query, top_k=payload.top_k)
        if retrieved:
            context_str = "\n\n".join(c["text"] for c in retrieved)
            chunks_to_cite = retrieved
        else:
            fallback = rag_engine.chunks[:payload.top_k]
            context_str = "\n\n".join(fallback) if fallback else getattr(rag_engine, "active_text", "")[:4000]
    else:
        input_source = payload.raw_html or payload.html or payload.url
        if input_source and input_source.strip():
            try:
                extracted = extract_content(input_source)
                rag_engine.build_index(extracted["text"])
                rag_engine.active_title = extracted.get("title", "Web Document")
                rag_engine.active_text = extracted["text"]
                retrieved = rag_engine.retrieve(payload.query, top_k=payload.top_k)
                if retrieved:
                    context_str = "\n\n".join(c["text"] for c in retrieved)
                    chunks_to_cite = retrieved
                else:
                    fallback = rag_engine.chunks[:payload.top_k]
                    context_str = "\n\n".join(fallback) if fallback else extracted["text"][:4000]
                    chunks_to_cite = [{"text": context_str, "score": 1.0, "chunk_index": 0}]
            except ValueError as err:
                raise HTTPException(status_code=422, detail=str(err)) from err
        elif rag_engine.chunks:
            # Re-use existing indexed document
            retrieved = rag_engine.retrieve(payload.query, top_k=payload.top_k)
            if retrieved:
                context_str = "\n\n".join(c["text"] for c in retrieved)
                chunks_to_cite = retrieved
            else:
                fallback = rag_engine.chunks[:payload.top_k]
                context_str = "\n\n".join(fallback) if fallback else getattr(rag_engine, "active_text", "")[:4000]

    # Guard maximum context length for LLM prompt
    if len(context_str) > 8000:
        context_str = context_str[:8000]

    async def event_generator():
        # First send citation chunks if available
        if chunks_to_cite:
            yield f"data: {json.dumps({'chunks': chunks_to_cite})}\n\n"

        # Stream LLM tokens from Ollama
        async for token in stream_rag_response(
            context=context_str,
            user_prompt=payload.query,
            model=payload.model,
            host=payload.host,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Signal completion
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/verify")
async def verify_endpoint(payload: VerifyRequest) -> Dict[str, Any]:
    """Agentic Mesh Layer 2: Asynchronously evaluate groundedness of a drafted answer."""
    return await evaluate_groundedness(
        context=payload.context,
        user_prompt=payload.query,
        answer=payload.answer,
        model=payload.model,
        host=payload.host,
    )


# Mount static files directory if it exists
static_path = Path(__file__).resolve().parent.parent.parent / "static"
if not static_path.exists():
    static_path = Path("./static").resolve()

if static_path.is_dir():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


def main() -> None:
    """Application entrypoint for local execution."""
    import uvicorn
    uvicorn.run("airgap_web_copilot.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
