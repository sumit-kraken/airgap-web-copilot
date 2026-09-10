"""In-memory RAG engine utilizing SentenceTransformers, sliding-window chunking, and FAISS."""

import re
from typing import Any, Dict, List, Union
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into dynamic sliding-window chunks based on word count.

    Args:
        text: Raw or cleaned text to chunk.
        chunk_size: Maximum words per chunk (default: 400).
        overlap: Number of overlapping words between consecutive chunks (default: 50).

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    words = re.findall(r"\S+", text)
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if i + chunk_size >= len(words):
            break

    return chunks


class RAGEngine:
    """In-memory FAISS vector indexing and retrieval engine using SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[str] = []

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk_document(self, text: str) -> List[str]:
        """Generate sliding window chunks for a given document text."""
        return chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)

    def build_index(self, document_or_chunks: Union[str, List[str]]) -> List[str]:
        """Index a raw document text or pre-split chunks into the in-memory FAISS index.

        Args:
            document_or_chunks: Either raw string text or list of chunk strings.

        Returns:
            The list of indexed chunks.
        """
        if isinstance(document_or_chunks, str):
            self.chunks = self.chunk_document(document_or_chunks)
        elif isinstance(document_or_chunks, list):
            self.chunks = [c.strip() for c in document_or_chunks if c and c.strip()]
        else:
            self.chunks = []

        if not self.chunks:
            self.index = None
            return []

        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return self.chunks

    def index_document(self, text: str) -> List[str]:
        """Alias for build_index with raw text."""
        return self.build_index(text)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Extract top relevant chunks for a user query.

        Args:
            query: The question or search query string.
            top_k: Number of most relevant chunks to return (default: 3).

        Returns:
            List of dicts containing 'text', 'score', and 'chunk_index'.
        """
        if not self.chunks or self.index is None or not query or not query.strip():
            return []

        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []

        query_vector = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(
                {
                    "text": self.chunks[idx],
                    "score": float(dist),
                    "chunk_index": int(idx),
                }
            )

        return results
