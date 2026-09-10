"""Unit and integration tests for AirGap Hybrid Copilot core services and API routes."""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from airgap_web_copilot.copilot import (
    check_ollama_status,
    evaluate_groundedness,
    stream_rag_response,
)
from airgap_web_copilot.extractor import extract_content
from airgap_web_copilot.main import app
from airgap_web_copilot.rag_engine import RAGEngine, chunk_text


# ---------------------------------------------------------------------------
# 1. Extractor Tests
# ---------------------------------------------------------------------------

def test_extract_content_valid_html():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Article Title</title></head>
    <body>
      <nav><a href="/home">Home</a><a href="/ad">Ad Link</a></nav>
      <script>var trackingCode = 12345;</script>
      <main>
        <h1>Main Heading</h1>
        <p>This is a high signal-to-noise sentence about air-gapped web copilot architectures.</p>
        <p>It processes everything locally without sending tokens to cloud providers.</p>
      </main>
      <footer><p>Copyright 2026</p></footer>
    </body>
    </html>
    """
    res = extract_content(html)
    assert isinstance(res, dict)
    assert "text" in res
    assert "char_count" in res
    assert "title" in res

    assert res["title"] == "Test Article Title"
    assert "high signal-to-noise sentence" in res["text"]
    assert "air-gapped web copilot" in res["text"]
    # Verify scripts and boilerplate are stripped
    assert "trackingCode" not in res["text"]
    assert res["char_count"] == len(res["text"])


def test_extract_content_empty_input():
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        extract_content("")

    with pytest.raises(ValueError, match="Input data cannot be empty"):
        extract_content("   \n\t  ")


def test_extract_content_unextractable_markup():
    with pytest.raises(ValueError, match="Failed to extract readable content"):
        extract_content("<html><body><script>console.log('only js');</script></body></html>")


def test_extract_content_invalid_url():
    with pytest.raises(ValueError, match="Failed to fetch content from URL"):
        extract_content("http://127.0.0.1:54321/nonexistent-page-url")


# ---------------------------------------------------------------------------
# 2. RAG Engine & Chunking Tests
# ---------------------------------------------------------------------------

def test_chunk_text_sliding_window():
    # Test short text (< 400 words) produces 1 chunk
    short_text = "This is a short test document containing only ten words here."
    short_chunks = chunk_text(short_text, chunk_size=400, overlap=50)
    assert len(short_chunks) == 1
    assert short_chunks[0] == short_text

    # Test multi-chunk sliding window
    words = [f"word{i}" for i in range(500)]
    long_text = " ".join(words)
    chunks = chunk_text(long_text, chunk_size=400, overlap=50)
    # 500 words with chunk_size 400 and overlap 50:
    # Chunk 0: words[0:400] (400 words)
    # Chunk 1: words[350:500] (150 words)
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 400
    assert len(chunks[1].split()) == 150

    # Verify 50 words overlap between chunk 0 and chunk 1
    c0_words = chunks[0].split()
    c1_words = chunks[1].split()
    assert c0_words[350:] == c1_words[:50]


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("    ") == []


def test_rag_engine_index_and_retrieve():
    engine = RAGEngine(chunk_size=400, overlap=50)
    
    docs = [
        "WebGPU executes WGSL compute shaders directly on the user's GPU hardware for fast token generation.",
        "FAISS provides efficient vector similarity search using in-memory indices like IndexFlatIP.",
        "Strawberry and watermelon recipes for summer refreshing fruit salads and smoothies.",
    ]
    indexed = engine.build_index(docs)
    assert len(indexed) == 3

    # Query about WebGPU
    results = engine.retrieve("How does WebGPU run shaders on local GPU?", top_k=2)
    assert len(results) == 2
    assert "WebGPU executes WGSL compute shaders" in results[0]["text"]
    assert results[0]["score"] > results[1]["score"]
    assert "score" in results[0]
    assert "chunk_index" in results[0]

    # Query about vector similarity
    results_vec = engine.retrieve("What algorithm is used for vector search?", top_k=1)
    assert len(results_vec) == 1
    assert "FAISS" in results_vec[0]["text"]


def test_rag_engine_empty_query_or_index():
    engine = RAGEngine()
    assert engine.retrieve("some query") == []

    engine.build_index(["Sample text."])
    assert engine.retrieve("") == []
    assert engine.retrieve("   ") == []


# ---------------------------------------------------------------------------
# 3. Copilot Service Tests
# ---------------------------------------------------------------------------

def test_check_ollama_status():
    import asyncio
    status = asyncio.run(check_ollama_status())
    assert isinstance(status, dict)
    assert "connected" in status
    assert "models" in status


def test_stream_rag_response_offline_fallback():
    import asyncio

    async def run():
        offline_stream = stream_rag_response(
            context="Context for offline test",
            user_prompt="What happens offline?",
            host="http://localhost:59999",
        )
        tokens = []
        async for token in offline_stream:
            tokens.append(token)
        return tokens

    tokens = asyncio.run(run())
    assert len(tokens) >= 1
    assert "offline or unreachable" in tokens[0].lower()


def test_evaluate_groundedness_offline():
    import asyncio
    result = asyncio.run(
        evaluate_groundedness(
            context="FAISS is a vector search library.",
            user_prompt="What is FAISS?",
            answer="FAISS is a vector library.",
            host="http://localhost:59999",
        )
    )
    assert isinstance(result, dict)
    assert result["status"] == "inconclusive"
    assert result["score"] == 0.0
    assert "offline or unreachable" in result["verdict"].lower()


def test_evaluate_groundedness_mock_verified():
    import asyncio
    mock_client = AsyncMock()
    mock_client.chat.return_value = {
        "message": {
            "content": '{"status": "verified", "score": 0.95, "verdict": "Strictly grounded in context.", "supported_claims": ["FAISS is a vector library"], "unsupported_claims": []}'
        }
    }
    with patch("airgap_web_copilot.copilot.check_ollama_status", return_value={"connected": True, "models": ["llama3.2"]}):
        result = asyncio.run(
            evaluate_groundedness(
                context="FAISS is a vector search library.",
                user_prompt="What is FAISS?",
                answer="FAISS is a vector library.",
                client=mock_client,
            )
        )
    assert result["status"] == "verified"
    assert result["score"] == 0.95
    assert "Strictly grounded" in result["verdict"]
    assert len(result["supported_claims"]) == 1
    assert len(result["unsupported_claims"]) == 0


# ---------------------------------------------------------------------------
# 4. FastAPI Endpoints Integration Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ollama" in data


def test_api_extract_context_missing_input(client):
    response = client.post("/api/extract-context", json={})
    assert response.status_code == 400


def test_api_extract_context_valid_html(client):
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Extract API Test</title></head>
    <body>
      <article>
        <h1>AirGap Copilot Architecture</h1>
        <p>WebLLM runs in the client browser using WebGPU acceleration.</p>
        <p>The backend Python server handles Trafilatura DOM extraction and FAISS indexing.</p>
      </article>
    </body>
    </html>
    """
    response = client.post(
        "/api/extract-context",
        json={"html": sample_html, "query": "WebGPU"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "context" in data
    assert "chunks" in data
    assert len(data["chunks"]) > 0
    assert data["title"] == "Extract API Test"
    assert data["char_count"] > 0


def test_api_analyze_missing_query(client):
    response = client.post("/api/analyze", json={})
    assert response.status_code == 422 or response.status_code == 400


def test_api_analyze_streaming_endpoint(client):
    sample_html = "<html><body><article><p>FastAPI serves SSE streams to frontend.</p></article></body></html>"
    with client.stream(
        "POST",
        "/api/analyze",
        json={"html": sample_html, "query": "What does FastAPI serve?"},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        lines = [line if isinstance(line, str) else line.decode("utf-8") for line in response.iter_lines() if line]
        assert len(lines) > 0
        # Should contain SSE format
        has_data_prefix = any(line.startswith("data:") for line in lines)
        assert has_data_prefix
        # Should contain [DONE] sentinel
        assert any("[DONE]" in line for line in lines)


def test_api_verify_endpoint(client):
    response = client.post(
        "/api/verify",
        json={
            "query": "What is WebGPU?",
            "context": "WebGPU is a graphics and compute API.",
            "answer": "WebGPU executes shaders directly on the GPU.",
            "host": "http://127.0.0.1:59999",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "score" in data
    assert "verdict" in data
    assert "supported_claims" in data
    assert "unsupported_claims" in data


def test_api_verify_missing_fields(client):
    response = client.post("/api/verify", json={"query": "test"})
    assert response.status_code == 422

