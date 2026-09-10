"""Root export for extractor module."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from airgap_web_copilot.extractor import extract_content, extract

__all__ = ["extract_content", "extract"]
