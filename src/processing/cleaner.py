"""
Text Cleaner — normalises raw strings before chunking or embedding.

Keeps medical symbols intact; focuses on whitespace / encoding artefacts.
"""
from __future__ import annotations

import re
import unicodedata


# ─── Public API ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lightweight, reversible cleaning.

    1. Normalise unicode (NFKC)
    2. Replace common ligatures and fancy quotes
    3. Collapse whitespace while preserving paragraph breaks
    4. Strip leading/trailing whitespace
    """
    # Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # Fancy quotes → plain
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Remove null bytes and other control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse spaces/tabs within a line
    text = re.sub(r"[ \t]+", " ", text)

    # Max two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def batch_clean(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]