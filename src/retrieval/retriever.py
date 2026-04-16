from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings.hf_embedder import embed_documents, embed_query
from src.utils.config import DEFAULT_TOP_K
from src.utils.logger import get_logger
from src.vectorstore.faiss_store import VectorStore

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONFIG  (tune without touching function signatures)
# ─────────────────────────────────────────────────────────────────────────────
BM25_TOP_N       = 10   # Stage 1: BM25 candidates-50
DENSE_TOP_N      = 6    # Stage 2: dense re-rank candidates-15
FAERS_BOOST      = 0.08  # additive score bonus for FAERS when boost_faers=True

# Hybrid fusion weights  (alpha * bm25_norm + beta * cosine)
ALPHA            = 0.30  # BM25 weight
BETA             = 0.70  # dense weight

# Cross-encoder model
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN DRUGS  (keyword drug-intent detection)
# ─────────────────────────────────────────────────────────────────────────────
_COMMON_DRUGS: frozenset[str] = frozenset({
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "naproxen",
    "metformin", "atorvastatin", "lisinopril", "amoxicillin", "warfarin",
    "clopidogrel", "omeprazole", "metoprolol", "amlodipine", "gabapentin",
    "sertraline", "fluoxetine", "prednisone", "levothyroxine", "albuterol",
    "diclofenac", "methotrexate", "adalimumab", "rituximab", "cetirizine",
})


def detect_drug(query: str) -> Optional[str]:
    lower = query.lower()
    for drug in _COMMON_DRUGS:
        if re.search(rf"\b{re.escape(drug)}\b", lower):
            return drug
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BM25 INDEX  (built once, cached for lifetime of process)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_bm25():
    """Build BM25 index from the FAISS vectorstore corpus (cached singleton)."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("Install rank_bm25: pip install rank-bm25")

    store = VectorStore.load()
    tokenized = [_tokenize(t) for t in store.texts]
    bm25 = BM25Okapi(tokenized)
    log.info("BM25 index built over %d documents.", len(store.texts))
    return bm25, store


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ENCODER  (loaded once, cached)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load CrossEncoder model (cached singleton)."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    log.info("Loading CrossEncoder: %s", CROSS_ENCODER_MODEL)
    model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    log.info("CrossEncoder loaded.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BM25 SPARSE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def bm25_search(query: str, top_n: int = BM25_TOP_N) -> list[dict]:
    """
    BM25 sparse retrieval over the full corpus.

    Returns top_n documents as dicts with an added 'bm25_score' field.
    Scores are raw BM25 values (not normalised yet).
    """
    try:
        bm25, store = _get_bm25()
        tokens      = _tokenize(query)
        scores      = bm25.get_scores(tokens)          # shape (N,)

        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = {
                "text"       : store.texts[idx],
                "bm25_score" : float(scores[idx]),
                "score"      : float(scores[idx]),   # placeholder, overwritten later
                "_idx"       : int(idx),
                **store.metadata[idx],
            }
            results.append(doc)

        log.debug("BM25 returned %d candidates (top score=%.4f)",
                  len(results), results[0]["bm25_score"] if results else 0)
        return results

    except Exception as exc:
        log.error("BM25 search failed: %s — falling back to empty.", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DENSE RETRIEVAL (cosine similarity on BM25 candidates)
# ─────────────────────────────────────────────────────────────────────────────

def dense_search(
    query: str,
    candidates: list[dict],
    top_n: int = DENSE_TOP_N,
) -> list[dict]:
    """
    Re-rank BM25 candidates using dense cosine similarity.

    Embeds query + all candidate texts in one batch call,
    computes cosine similarity, fuses with BM25 score, returns top_n.
    """
    if not candidates:
        return []

    try:
        # Embed query
        query_vec = embed_query(query)                     # (1, D)

        # Batch embed all candidate texts (single API/model call)
        texts     = [c["text"] for c in candidates]
        doc_vecs  = embed_documents(texts)                 # (N, D)

        # Cosine similarity  (both already L2-normalised → dot product = cosine)
        cos_scores = cosine_similarity(query_vec, doc_vecs)[0]  # (N,)

        # Normalise BM25 scores to [0, 1] for hybrid fusion
        bm25_scores = np.array([c["bm25_score"] for c in candidates], dtype=np.float32)
        bm25_max    = bm25_scores.max()
        bm25_norm   = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores

        # Hybrid score
        hybrid = ALPHA * bm25_norm + BETA * cos_scores

        # Attach scores and sort
        for i, doc in enumerate(candidates):
            doc["cosine_score"] = float(cos_scores[i])
            doc["hybrid_score"] = float(hybrid[i])
            doc["score"]        = float(hybrid[i])   # canonical score field

        ranked = sorted(candidates, key=lambda d: d["hybrid_score"], reverse=True)
        top    = ranked[:top_n]

        log.debug(
            "Dense re-rank: %d → %d candidates (top hybrid=%.4f)",
            len(candidates), len(top),
            top[0]["hybrid_score"] if top else 0,
        )
        return top

    except Exception as exc:
        log.error("Dense search failed: %s — returning BM25 candidates.", exc)
        return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CROSS-ENCODER RE-RANKING
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = DEFAULT_TOP_K,
    boost_faers: bool = False,
    detected_drug: Optional[str] = None,
) -> list[dict]:
    """
    Cross-encoder re-ranking of dense candidates.

    Optionally applies FAERS boost before final sort.
    Returns final top_k results.
    """
    if not candidates:
        return []

    try:
        cross_encoder = _get_cross_encoder()
        pairs         = [(query, doc["text"]) for doc in candidates]
        ce_scores     = cross_encoder.predict(pairs, show_progress_bar=False)

        for i, doc in enumerate(candidates):
            doc["ce_score"] = float(ce_scores[i])
            doc["score"]    = float(ce_scores[i])

        # FAERS boost — applied AFTER cross-encoder scoring
        if boost_faers and detected_drug:
            for doc in candidates:
                if doc.get("source") == "faers":
                    drug_in_meta = str(doc.get("drug", "")).lower()
                    if detected_drug in drug_in_meta:
                        doc["score"] += FAERS_BOOST
                        doc["ce_score"] += FAERS_BOOST

        candidates.sort(key=lambda d: d["score"], reverse=True)
        final = candidates[:top_k]

        log.debug(
            "Cross-encoder re-rank: %d → %d (top score=%.4f)",
            len(candidates), len(final),
            final[0]["score"] if final else 0,
        )
        return final

    except Exception as exc:
        log.error("Cross-encoder failed: %s — using hybrid scores.", exc)
        # Graceful fallback: use hybrid scores with optional boost
        if boost_faers and detected_drug:
            for doc in candidates:
                if doc.get("source") == "faers":
                    if detected_drug in str(doc.get("drug", "")).lower():
                        doc["score"] += FAERS_BOOST
        candidates.sort(key=lambda d: d["score"], reverse=True)
        return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  — unchanged signature
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    boost_faers: bool = True,
) -> list[dict]:
    """
    Multi-stage hybrid retrieval pipeline.

    Stage 1  BM25 sparse retrieval        → top 50 candidates
    Stage 2  Dense cosine + hybrid fusion → top 15 candidates
    Stage 3  Cross-encoder re-ranking     → final top_k results

    Parameters
    ----------
    query       : natural language question
    top_k       : number of final results (applied only at stage 3)
    boost_faers : boost FAERS documents matching detected drug

    Returns
    -------
    list[dict] sorted by descending relevance, each dict contains:
        text, score, source, file, page, drug, reaction,
        bm25_score, cosine_score, hybrid_score, ce_score
    """
    t0 = time.perf_counter()
    log.info("retrieve() | query='%s' top_k=%d boost_faers=%s",
             query[:80], top_k, boost_faers)

    # Drug intent detection (used for boost, not filtering)
    detected_drug = detect_drug(query) if boost_faers else None
    if detected_drug:
        log.info("Drug intent detected: %s", detected_drug)

    # ── Stage 1: BM25 ────────────────────────────────────────────────────────
    bm25_candidates = bm25_search(query, top_n=BM25_TOP_N)

    if not bm25_candidates:
        log.warning("BM25 returned 0 results — falling back to pure FAISS.")
        bm25_candidates = _faiss_fallback(query, top_n=BM25_TOP_N)

    # ── Stage 2: Dense re-rank ────────────────────────────────────────────────
    dense_candidates = dense_search(query, bm25_candidates, top_n=DENSE_TOP_N)

    if not dense_candidates:
        log.warning("Dense search returned 0 results — using BM25 top results.")
        dense_candidates = bm25_candidates[:DENSE_TOP_N]

    # ── Stage 3: Cross-encoder re-rank ────────────────────────────────────────
    final = rerank(
        query        = query,
        candidates   = dense_candidates,
        top_k        = top_k,
        boost_faers  = boost_faers,
        detected_drug= detected_drug,
    )

    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "retrieve() done | results=%d | scores=%.4f…%.4f | elapsed=%.0fms",
        len(final),
        final[0]["score"] if final else 0,
        final[-1]["score"] if final else 0,
        elapsed,
    )
    return final


# ─────────────────────────────────────────────────────────────────────────────
# FAISS FALLBACK  (pure dense when BM25 fails)
# ─────────────────────────────────────────────────────────────────────────────

def _faiss_fallback(query: str, top_n: int) -> list[dict]:
    """Pure FAISS retrieval used as fallback when BM25 yields nothing."""
    try:
        store     = VectorStore.load()
        query_vec = embed_query(query)
        results   = store.search(query_vec, top_k=min(top_n, store.num_vectors))
        for r in results:
            r["bm25_score"]   = 0.0
            r["cosine_score"] = r.get("score", 0.0)
            r["hybrid_score"] = r.get("score", 0.0)
        return results
    except Exception as exc:
        log.error("FAISS fallback also failed: %s", exc)
        return []