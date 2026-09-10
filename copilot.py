"""Root export for copilot module."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from airgap_web_copilot.copilot import (
    check_ollama_status,
    evaluate_groundedness,
    get_ollama_client,
    stream_rag_response,
)

__all__ = [
    "check_ollama_status",
    "evaluate_groundedness",
    "get_ollama_client",
    "stream_rag_response",
]
