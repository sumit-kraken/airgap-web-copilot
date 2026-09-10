"""Root export for rag_engine module."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from airgap_web_copilot.rag_engine import RAGEngine, chunk_text

__all__ = ["RAGEngine", "chunk_text"]
