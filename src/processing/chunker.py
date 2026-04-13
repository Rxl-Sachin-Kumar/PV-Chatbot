"""
Chunker — splits long texts into overlapping token-approximate windows.

Uses a word-based approximation (1 word ≈ 1.3 tokens on average for English)
so we avoid a hard tiktoken dependency while remaining close to token limits.

For PDFs : chunk_size=400, overlap=80 (config defaults)
For FAERS: records are pre-chunked → pass-through only
"""
from __future__ import annotations

from src.utils.config import PDF_CHUNK_OVERLAP, PDF_CHUNK_SIZE
from src.utils.logger import get_logger

log = get_logger(__name__)

# Approx tokens per word for biomedical English text
_TOKENS_PER_WORD = 1.35


def _word_count_to_chars(n_words: int, sample: str) -> int:
    """Heuristic: average chars per word from a sample."""
    words = sample.split()
    if not words:
        return n_words * 6
    return int((len(sample) / len(words)) * n_words)


def chunk_text(
    text: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = PDF_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping windows measured in approximate tokens.

    Strategy
    --------
    • Convert token sizes → word counts
    • Split on word boundaries
    • Slide a window with *overlap* words of context
    """
    chunk_words = max(1, int(chunk_size / _TOKENS_PER_WORD))
    overlap_words = max(0, int(overlap / _TOKENS_PER_WORD))

    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_words:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_words - overlap_words   # slide forward

    return chunks


def chunk_pdf_pages(
    pages: list[dict],
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = PDF_CHUNK_OVERLAP,
) -> tuple[list[str], list[dict]]:
    """
    Chunk a list of PDF page dicts (output of pdf_loader.load_pdfs).

    Returns
    -------
    texts    : list[str]
    metadata : list[dict]   {"source": "book", "file": ..., "page": ..., "chunk": ...}
    """
    texts: list[str] = []
    metadata: list[dict] = []

    for page in pages:
        raw_text = page["text"]
        file_name = page["file"]
        page_num = page["page"]

        chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            metadata.append(
                {
                    "source": "book",
                    "file": file_name,
                    "page": page_num,
                    "chunk": idx,
                }
            )

    log.info(
        "Chunked %d PDF pages → %d chunks (size≈%d tok, overlap≈%d tok)",
        len(pages),
        len(texts),
        chunk_size,
        overlap,
    )
    return texts, metadata