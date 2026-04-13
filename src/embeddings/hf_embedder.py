"""
Embedder — uses sentence-transformers locally.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 80MB download (vs 1.3GB for BGE-large)
  - ~5x faster on CPU
  - 384-dim embeddings
  - Excellent semantic similarity quality
"""
from __future__ import annotations

import numpy as np
from src.utils.config import (
    API_BATCH_SIZE,
    BGE_QUERY_PREFIX,
    EMBEDDING_DIM,
    HF_MODEL,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Singleton model
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        log.info("Loading embedding model: %s", HF_MODEL)
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(HF_MODEL)
        log.info("Model loaded. Embedding dim: %d", _MODEL.get_sentence_embedding_dimension())
    return _MODEL


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def embed_documents(texts: list[str], batch_size: int = API_BATCH_SIZE) -> np.ndarray:
    """
    Embed a list of document texts locally.

    Returns
    -------
    np.ndarray  shape (N, 384), float32, L2-normalised
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    log.info("Embedding %d documents ...", len(texts))

    matrix = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    matrix = matrix.astype(np.float32)
    log.info("Done. Shape: %s", matrix.shape)
    return matrix


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns
    -------
    np.ndarray  shape (1, 384), float32, L2-normalised
    """
    q = BGE_QUERY_PREFIX + query if BGE_QUERY_PREFIX else query
    model = _get_model()
    vec = model.encode(
        [q],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)