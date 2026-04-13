"""
Centralized configuration — all paths and hyperparams live here.
Never hardcode anything outside this file.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root — this file lives at src/utils/config.py
# parents[0]=utils, parents[1]=src, parents[2]=project root
ROOT = Path(__file__).resolve().parents[2]

# Raw data — PDFs are in data/ root (not data/raw/pdfs/)
PDF_DIR         = ROOT / "data"
FAERS_JSON_PATH = ROOT / "data" / "processed" / "faers_rag.json"

# Vector store
VECTORSTORE_DIR  = ROOT / "vectorstore" / "faiss_index"
FAISS_INDEX_FILE = VECTORSTORE_DIR / "index.faiss"
METADATA_FILE    = VECTORSTORE_DIR / "metadata.pkl"
TEXTS_FILE       = VECTORSTORE_DIR / "texts.pkl"

# Hugging Face — fast lightweight model
HF_API_KEY    = os.getenv("HF_API_KEY", "")
HF_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
HF_ENDPOINT   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
EMBEDDING_DIM = 384

# Chunking
PDF_CHUNK_SIZE    = 400
PDF_CHUNK_OVERLAP = 80

# Retrieval
DEFAULT_TOP_K = 5

# API
API_BATCH_SIZE    = 64
API_MAX_RETRIES   = 5
API_RETRY_BACKOFF = 2.0
API_TIMEOUT       = 60

# Query prefix
BGE_QUERY_PREFIX = ""