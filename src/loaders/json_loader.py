"""
FAERS JSON Loader — reads the pre-processed FAERS RAG JSON.

Expected schema (each element):
    {
        "text": "...",          # required — the chunk text
        "drug": "ASPIRIN",      # optional
        "reaction": "headache", # optional
        ...                     # any additional fields preserved in metadata
    }

Returns:
    texts    : list[str]
    metadata : list[dict]   {"source": "faers", **original_fields_minus_text}
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import FAERS_JSON_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_faers(
    json_path: str | Path | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Load FAERS records.

    Parameters
    ----------
    json_path : optional override for config.FAERS_JSON_PATH

    Returns
    -------
    texts    : list[str]
    metadata : list[dict]
    """
    json_path = Path(json_path or FAERS_JSON_PATH)

    if not json_path.exists():
        log.warning("FAERS JSON not found at %s — skipping.", json_path)
        return [], []

    log.info("Loading FAERS JSON from %s", json_path)
    with json_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        raise ValueError(f"FAERS JSON must be a list of objects, got {type(records)}")

    texts: list[str] = []
    metadata: list[dict] = []

    for i, rec in enumerate(records):
        text = rec.get("text", "").strip()
        if not text:
            log.debug("Record %d has no 'text' field — skipping.", i)
            continue

        texts.append(text)

        meta = {k: v for k, v in rec.items() if k != "text"}
        meta["source"] = "faers"
        metadata.append(meta)

    log.info("Loaded %d FAERS records.", len(texts))
    return texts, metadata