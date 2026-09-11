"""DOM extraction and content cleaning module using Trafilatura for AirGap Hybrid Copilot."""

import re
from typing import Any, Dict
import trafilatura


def extract_content(input_data: str) -> Dict[str, Any]:
    """Extract readable markdown content from raw HTML strings or URLs.

    Args:
        input_data: A raw HTML string or an HTTP/HTTPS URL.

    Returns:
        A dict containing:
            - 'text': Extracted readable markdown text.
            - 'char_count': Total characters in extracted text.
            - 'title': Document title if available.

    Raises:
        ValueError: If input is empty, URL fetch fails, or text extraction fails.
    """
    if not input_data or not input_data.strip():
        raise ValueError("Input data cannot be empty.")

    target = input_data.strip()
    is_url = target.startswith(("http://", "https://"))
    raw_html: str | None = None

    if is_url:
        try:
            downloaded = trafilatura.fetch_url(target)
            if not downloaded:
                raise ValueError(f"Failed to fetch content from URL: '{target}'")
            raw_html = downloaded
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error fetching URL '{target}': {str(e)}") from e
    else:
        raw_html = target

    # Extract title if present in raw HTML
    title = "Web Document"
    if raw_html:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
        if title_match:
            clean_title = re.sub(r"\s+", " ", title_match.group(1)).strip()
            if clean_title:
                title = clean_title

    # Extract clean markdown text using trafilatura
    extracted_text = trafilatura.extract(
        raw_html,
        output_format="markdown",
        include_formatting=True,
        include_links=True,
        favor_precision=True,
    )

    # Fallback to standard extraction without precision constraint if None
    if not extracted_text:
        extracted_text = trafilatura.extract(raw_html, output_format="markdown")

    if not extracted_text:
        extracted_text = trafilatura.extract(raw_html, favor_recall=True)

    # Resilient fallback: clean HTML tags if trafilatura heuristics filtered everything out
    if not extracted_text or not extracted_text.strip():
        cleaned = re.sub(r"<(script|style|svg|noscript)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            extracted_text = cleaned

    if not extracted_text or not extracted_text.strip():
        raise ValueError(
            "Failed to extract readable content from the provided input. "
            "The document contains insufficient textual body content or only unsupported elements."
        )

    text = extracted_text.strip()
    return {
        "text": text,
        "char_count": len(text),
        "title": title,
    }


# Convenience alias
extract = extract_content
