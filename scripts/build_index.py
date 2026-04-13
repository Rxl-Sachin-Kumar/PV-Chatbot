"""
scripts/build_index.py

Run ONCE to build the FAISS vector store.

Sources
-------
  1. PDFs   : data/Pharmacovigilance.pdf + data/Pharmacovigilance-A-Practical-Approach.pdf
  2. FAERS  : data/processed/faers_rag.json (pre-chunked)

Usage
-----
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loaders.pdf_loader import load_pdfs
from src.loaders.json_loader import load_faers
from src.processing.chunker import chunk_pdf_pages
from src.processing.cleaner import batch_clean
from src.embeddings.hf_embedder import embed_documents
from src.vectorstore.faiss_store import build_and_save
from src.utils.logger import get_logger

log = get_logger("build_index")


def main() -> None:
    t0 = time.perf_counter()
    log.info("=" * 60)
    log.info("  Pharmacovigilance RAG — Index Builder")
    log.info("  Sources: PDFs + FAERS JSON")
    log.info("=" * 60)

    # ── 1. Load PDFs ──────────────────────────────────────────────
    log.info("\n[1/5] Loading PDFs ...")
    pdf_pages = load_pdfs()

    # ── 2. Chunk PDFs ─────────────────────────────────────────────
    log.info("\n[2/5] Chunking PDF pages ...")
    pdf_texts, pdf_meta = chunk_pdf_pages(pdf_pages)
    log.info("  PDF chunks: %d", len(pdf_texts))

    # ── 3. Load FAERS ─────────────────────────────────────────────
    log.info("\n[3/5] Loading FAERS JSON ...")
    faers_texts, faers_meta = load_faers()
    log.info("  FAERS records: %d", len(faers_texts))

    # ── 4. Merge + clean ──────────────────────────────────────────
    log.info("\n[4/5] Merging and cleaning ...")
    all_texts = batch_clean(pdf_texts + faers_texts)
    all_meta  = pdf_meta + faers_meta
    log.info("  Total documents to embed: %d", len(all_texts))

    if not all_texts:
        log.error("No documents found. Check data/ directory. Aborting.")
        sys.exit(1)

    # ── 5. Embed + save ───────────────────────────────────────────
    log.info("\n[5/5] Generating embeddings ...")
    embeddings = embed_documents(all_texts)
    build_and_save(embeddings, all_texts, all_meta)

    elapsed = time.perf_counter() - t0
    log.info("\n✓ Index built in %.1f seconds.", elapsed)
    log.info("  Total vectors : %d  (PDF: %d + FAERS: %d)",
             len(all_texts), len(pdf_texts), len(faers_texts))
    log.info("  Run UI with  : streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()