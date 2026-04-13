"""
PDF Loader — extracts raw text from every PDF in a directory.
Skips sub-directories automatically so FAERS PDFs (ASC_NTS, README etc.)
inside data/FAERS/ are ignored — only top-level PDFs are loaded.

Returns a list of dicts:
    {"text": str, "page": int, "file": str}
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from src.utils.config import PDF_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

# PDFs inside these subdirectories are skipped (FAERS docs, not knowledge base)
_SKIP_SUBDIRS = {"FAERS", "raw", "processed"}


def _clean_page_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned: list[str] = []
    prev_blank = False
    for ln in lines:
        if ln == "":
            if not prev_blank:
                cleaned.append(ln)
            prev_blank = True
        else:
            cleaned.append(ln)
            prev_blank = False
    return "\n".join(cleaned).strip()


def load_pdfs(pdf_dir: str | Path | None = None) -> list[dict]:
    """
    Load all top-level PDFs from *pdf_dir* (default: config.PDF_DIR).
    Skips PDFs inside FAERS/, raw/, processed/ subdirectories.

    Returns
    -------
    list[dict]  keys: text, page, file
    """
    pdf_dir = Path(pdf_dir or PDF_DIR)
    if not pdf_dir.exists():
        log.warning("PDF directory not found: %s — skipping.", pdf_dir)
        return []

    # Only grab PDFs directly in pdf_dir, not nested subdirs
    pdf_files = sorted(
        p for p in pdf_dir.glob("*.pdf")
        if p.parent.name not in _SKIP_SUBDIRS
    )

    if not pdf_files:
        log.warning("No PDF files found in %s", pdf_dir)
        return []

    log.info("Found %d PDF file(s) in %s", len(pdf_files), pdf_dir)
    pages: list[dict] = []

    for pdf_path in pdf_files:
        log.info("Loading: %s", pdf_path.name)
        page_count_before = len(pages)
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, start=1):
                raw = page.get_text("text")
                text = _clean_page_text(raw)
                if len(text) < 50:
                    continue
                pages.append({"text": text, "page": page_num, "file": pdf_path.name})
            doc.close()
            log.info("  → %d usable pages extracted from %s",
                     len(pages) - page_count_before, pdf_path.name)
        except Exception as exc:
            log.error("Failed to load %s: %s", pdf_path.name, exc)

    log.info("Total pages loaded from all PDFs: %d", len(pages))
    return pages