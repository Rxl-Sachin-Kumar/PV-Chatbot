"""
Retriever — the single entry-point for querying the RAG system.

Pipeline
--------
1. Detect drug entities in the query (optional boost)
2. Prepend BGE query instruction
3. Embed via HF API
4. Normalise
5. Search FAISS
6. Optionally re-score FAERS results when a drug is detected
7. Return top-k results

Designed so an LLM generation step can be appended easily:
    results = retrieve(query)
    context = "\n\n".join(r["text"] for r in results)
    answer  = llm.generate(context, query)   # future integration
"""
from __future__ import annotations

import re

from src.embeddings.hf_embedder import embed_query
from src.utils.config import DEFAULT_TOP_K
from src.utils.logger import get_logger
from src.vectorstore.faiss_store import VectorStore

log = get_logger(__name__)

# ── Known drugs for intent detection (extend / replace with NER later) ────────
_COMMON_DRUGS: frozenset[str] = frozenset(
    {
        "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "naproxen",
        "metformin", "atorvastatin", "lisinopril", "amoxicillin", "warfarin",
        "clopidogrel", "omeprazole", "metoprolol", "amlodipine", "gabapentin",
        "sertraline", "fluoxetine", "prednisone", "levothyroxine", "albuterol",
    }
)

# Boost factor applied to FAERS results when a drug match is found
_FAERS_BOOST = 0.05   # additive score bonus


def detect_drug(query: str) -> str | None:
    """
    Simple keyword-based drug detection.

    Returns the matched drug name (lower-case) or None.
    Future: replace with a proper NER model (scispaCy, medspaCy, etc.)
    """
    lower = query.lower()
    for drug in _COMMON_DRUGS:
        # whole-word match
        if re.search(rf"\b{re.escape(drug)}\b", lower):
            log.debug("Drug detected in query: %s", drug)
            return drug
    return None


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    boost_faers: bool = True,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for *query*.

    Parameters
    ----------
    query        : natural language question
    top_k        : number of results to return
    boost_faers  : apply drug-intent boosting when a drug is detected

    Returns
    -------
    list[dict] sorted by descending relevance score:
        {text, score, source, file?, page?, drug?, reaction?, ...}
    """
    log.info("Retrieving top-%d results for query: %s", top_k, query[:80])

    # 1. Drug intent detection
    detected_drug: str | None = None
    if boost_faers:
        detected_drug = detect_drug(query)
        if detected_drug:
            log.info("Drug intent detected: %s → boosting FAERS results.", detected_drug)

    # 2-4. Embed query (includes BGE prefix + L2 norm)
    query_vec = embed_query(query)

    # 5. Search FAISS — fetch extra candidates when boosting
    fetch_k = top_k * 3 if detected_drug else top_k
    store = VectorStore.load()
    candidates = store.search(query_vec, top_k=min(fetch_k, store.num_vectors))

    # 6. Optional re-scoring
    if detected_drug:
        for res in candidates:
            if res.get("source") == "faers":
                drug_in_meta = str(res.get("drug", "")).lower()
                if detected_drug in drug_in_meta:
                    res["score"] += _FAERS_BOOST

        candidates.sort(key=lambda r: r["score"], reverse=True)

    results = candidates[:top_k]
    log.info(
        "Retrieved %d results (scores: %.4f … %.4f)",
        len(results),
        results[0]["score"] if results else 0,
        results[-1]["score"] if results else 0,
    )
    return results