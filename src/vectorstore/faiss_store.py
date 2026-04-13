"""
FAISS Vector Store

Responsibilities
----------------
• Build an IndexFlatIP index from embedding matrix
• Persist: index.faiss + metadata.pkl + texts.pkl
• Load (singleton-cached) for query time — zero recomputation
• Search returning (text, score, metadata) triples

IndexFlatIP + L2-normalised vectors → cosine similarity search.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.utils.config import (
    EMBEDDING_DIM,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    TEXTS_FILE,
    VECTORSTORE_DIR,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Build & Save ──────────────────────────────────────────────────────────────

def build_and_save(
    embeddings: np.ndarray,
    texts: list[str],
    metadata: list[dict],
    vectorstore_dir: str | Path | None = None,
) -> None:
    """
    Build a FAISS IndexFlatIP from *embeddings* and persist all artefacts.

    Parameters
    ----------
    embeddings     : (N, D) float32 array, must be L2-normalised
    texts          : raw text strings (parallel to embeddings)
    metadata       : dicts with source / file / drug / reaction info
    vectorstore_dir: override for config.VECTORSTORE_DIR
    """
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided — cannot build index.")

    assert embeddings.ndim == 2, "embeddings must be 2-D"
    assert embeddings.shape[0] == len(texts) == len(metadata), (
        "embeddings / texts / metadata must have the same length"
    )

    save_dir = Path(vectorstore_dir or VECTORSTORE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    log.info("Building IndexFlatIP  dim=%d  vectors=%d", dim, len(texts))

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    log.info("Saving index → %s", save_dir)
    faiss.write_index(index, str(save_dir / "index.faiss"))

    with open(save_dir / "metadata.pkl", "wb") as fh:
        pickle.dump(metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_dir / "texts.pkl", "wb") as fh:
        pickle.dump(texts, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("Vector store saved (%d vectors).", index.ntotal)


# ── Load (cached singleton) ───────────────────────────────────────────────────

class VectorStore:
    """
    Lightweight wrapper holding the loaded FAISS index + parallel arrays.

    Use ``VectorStore.load()`` rather than the constructor directly.
    """

    def __init__(
        self,
        index: faiss.Index,
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        self.index = index
        self.texts = texts
        self.metadata = metadata

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        vectorstore_dir: str | Path | None = None,
    ) -> "VectorStore":
        """Load from disk. Repeated calls within a process return the same object."""
        return _load_cached(str(vectorstore_dir or VECTORSTORE_DIR))

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Nearest-neighbour search.

        Parameters
        ----------
        query_embedding : shape (1, D) or (D,), float32, L2-normalised
        top_k           : number of results

        Returns
        -------
        list of dicts:
            text, score, source, file, page, chunk, drug, reaction (when present)
        """
        vec = np.atleast_2d(query_embedding).astype(np.float32)

        scores, indices = self.index.search(vec, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:        # FAISS uses -1 for padding
                continue
            result = {
                "text": self.texts[idx],
                "score": float(score),
                **self.metadata[idx],
            }
            results.append(result)

        return results

    @property
    def num_vectors(self) -> int:
        return self.index.ntotal


# ── Private cached loader ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_cached(vectorstore_dir: str) -> VectorStore:
    """Cached so the index is loaded at most once per process."""
    save_dir = Path(vectorstore_dir)
    index_path    = save_dir / "index.faiss"
    metadata_path = save_dir / "metadata.pkl"
    texts_path    = save_dir / "texts.pkl"

    for p in (index_path, metadata_path, texts_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Vector store artefact missing: {p}\n"
                "Run `python scripts/build_index.py` first."
            )

    log.info("Loading FAISS index from %s ...", save_dir)
    index = faiss.read_index(str(index_path))

    with open(metadata_path, "rb") as fh:
        metadata: list[dict] = pickle.load(fh)

    with open(texts_path, "rb") as fh:
        texts: list[str] = pickle.load(fh)

    log.info("FAISS index loaded: %d vectors, dim=%d", index.ntotal, index.d)
    return VectorStore(index=index, texts=texts, metadata=metadata)