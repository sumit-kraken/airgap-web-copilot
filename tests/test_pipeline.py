"""End-to-end pipeline verification test suite for AirGap Hybrid Copilot.

Verifies:
1. Extractor HTML fixture cleaning (script, nav, and noise removal).
2. RAG engine sliding-window chunking and semantic vector retrieval.
3. Async integration tests via httpx.AsyncClient for /api/extract-context and /api/analyze SSE error fallback.
"""

import json
import pytest
from httpx import ASGITransport, AsyncClient

from airgap_web_copilot.extractor import extract_content
from airgap_web_copilot.main import app
from airgap_web_copilot.rag_engine import RAGEngine, chunk_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_html_document() -> str:
    """Mock web document fixture containing noisy scripts, navigation, and core article text."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Air-Gapped Sovereign AI Architecture</title>
        <script>
            window.__ANALYTICS_TRACKER__ = function() { return "tracking_telemetry_id_999"; };
        </script>
        <style>
            .ad-banner { display: block; color: red; }
        </style>
    </head>
    <body>
        <header>
            <nav>
                <ul>
                    <li><a href="/home">Home Link</a></li>
                    <li><a href="/products">Products Link</a></li>
                    <li><a href="/sponsors">Sponsors Link</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <article>
                <h1>Local Intelligence Without Cloud Egress</h1>
                <p>
                    The AirGap Hybrid Copilot executes user inference within the client hardware boundaries.
                    By running WebLLM on WebGPU compute shaders, user data never touches a remote server.
                </p>
                <p>
                    For heavier reasoning tasks or hardware without WebGPU, the system seamlessly redirects
                    the prompt to a localhost Ollama daemon via server-sent events.
                </p>
            </article>
        </main>

        <aside class="sidebar-ads">
            <p>Click here for discounted cloud GPUs and commercial subscriptions.</p>
        </aside>

        <footer>
            <p>&copy; 2026 Sovereign Systems. All rights reserved.</p>
            <script>console.log("Footer script execution");</script>
        </footer>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# 1. Extractor Unit Tests
# ---------------------------------------------------------------------------

def test_extractor_strips_scripts_and_navigation(mock_html_document: str):
    """Verify Trafilatura extracts core content while stripping scripts, navigation, and noise."""
    result = extract_content(mock_html_document)

    assert isinstance(result, dict)
    assert result["title"] == "Air-Gapped Sovereign AI Architecture"

    text = result["text"]
    assert "AirGap Hybrid Copilot executes user inference" in text
    assert "WebLLM on WebGPU compute shaders" in text
    assert "localhost Ollama daemon" in text

    # Verify script content is completely removed
    assert "__ANALYTICS_TRACKER__" not in text
    assert "tracking_telemetry_id_999" not in text
    assert "Footer script execution" not in text

    # Verify char count accuracy
    assert result["char_count"] == len(text)
    assert result["char_count"] > 100


def test_extractor_error_on_empty_and_corrupt():
    """Verify extractor raises informative ValueError on unextractable or empty input."""
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        extract_content("   ")

    with pytest.raises(ValueError, match="Failed to extract readable content"):
        extract_content("<html><body><div></div><script>var x = 1;</script></body></html>")


# ---------------------------------------------------------------------------
# 2. RAG Engine Unit Tests
# ---------------------------------------------------------------------------

def test_rag_engine_chunk_sliding_window_boundaries():
    """Verify dynamic sliding window chunks adhere to chunk size and overlap constraints."""
    total_words = 900
    words = [f"token_{i}" for i in range(total_words)]
    text = " ".join(words)

    chunk_size = 400
    overlap = 50
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # Calculation: step = 350.
    # Chunk 0: 0..400
    # Chunk 1: 350..750
    # Chunk 2: 700..900 (200 words)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 400
    assert len(chunks[1].split()) == 400
    assert len(chunks[2].split()) == 200

    # Overlap validation between chunk 0 and chunk 1
    c0 = chunks[0].split()
    c1 = chunks[1].split()
    assert c0[350:] == c1[:50]


def test_rag_engine_semantic_search_retrieval():
    """Verify in-memory FAISS returns semantically relevant chunks ranked by similarity score."""
    engine = RAGEngine(chunk_size=400, overlap=50)

    corpus = [
        "WebGPU enables in-browser hardware acceleration using WGSL shaders for client LLM execution.",
        "FAISS indexes dense embedding vectors using Inner Product similarity for sub-millisecond search.",
        "A recipe for chocolate chip cookies requires flour, sugar, butter, and vanilla extract.",
        "Deep learning model quantization reduces 16-bit floats to 4-bit integers with minimal loss.",
    ]
    engine.build_index(corpus)

    # Query 1: Vector indexing
    results_rag = engine.retrieve("How are vector embeddings indexed and searched rapidly?", top_k=2)
    assert len(results_rag) == 2
    assert "FAISS indexes dense embedding vectors" in results_rag[0]["text"]
    assert results_rag[0]["score"] > results_rag[1]["score"]

    # Query 2: In-browser GPU shaders
    results_gpu = engine.retrieve("Can WebGPU run shaders directly in client web browsers?", top_k=1)
    assert len(results_gpu) == 1
    assert "WebGPU enables in-browser hardware acceleration" in results_gpu[0]["text"]


# ---------------------------------------------------------------------------
# 3. FastAPI Async Integration Tests (httpx.AsyncClient)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_api_extract_context_async_integration(mock_html_document: str):
    """Test POST /api/extract-context returns valid JSON with context and indexed chunks."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/extract-context",
            json={
                "html": mock_html_document,
                "query": "What hardware runs WebLLM?",
                "top_k": 3,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "WebLLM" in data["context"]
        assert isinstance(data["chunks"], list)
        assert len(data["chunks"]) > 0
        assert data["title"] == "Air-Gapped Sovereign AI Architecture"
        assert data["char_count"] > 0


@pytest.mark.anyio
async def test_api_analyze_sse_offline_fallback_integration(mock_html_document: str):
    """Test POST /api/analyze streams SSE events and falls back gracefully when Ollama is unreachable."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Pass an unreachable Ollama port (e.g. 59999) to trigger the graceful fallback generator
        async with client.stream(
            "POST",
            "/api/analyze",
            json={
                "html": mock_html_document,
                "query": "What are the air-gap guarantees?",
                "host": "http://127.0.0.1:59999",
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            lines = []
            async for line in response.aiter_lines():
                if line:
                    lines.append(line)

            assert len(lines) > 0

            # Parse SSE lines
            tokens = []
            has_done_sentinel = False
            for line in lines:
                if line.startswith("data: "):
                    payload_str = line[6:].strip()
                    if payload_str == "[DONE]":
                        has_done_sentinel = True
                    else:
                        parsed = json.loads(payload_str)
                        if "token" in parsed:
                            tokens.append(parsed["token"])

            assert has_done_sentinel, "Stream did not end with [DONE] sentinel"
            assert len(tokens) > 0, "No tokens yielded in event stream"

            combined_tokens = "".join(tokens)
            # Verify graceful error notification was yielded without server crash
            assert "offline or unreachable" in combined_tokens.lower()


@pytest.mark.anyio
async def test_api_verify_pipeline_integration():
    """Test POST /api/verify endpoint integration for Agentic Mesh LLM-as-a-Judge."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/verify",
            json={
                "query": "Does WebLLM make cloud calls?",
                "context": "WebLLM runs entirely inside the browser using WebGPU with zero cloud telemetry.",
                "answer": "No, WebLLM executes locally inside the browser on WebGPU.",
                "host": "http://127.0.0.1:59999",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["verified", "flagged", "inconclusive"]
        assert "score" in data
        assert "verdict" in data
        assert "judge_model" in data
        assert isinstance(data["supported_claims"], list)
        assert isinstance(data["unsupported_claims"], list)

