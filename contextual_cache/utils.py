"""
Shared utilities used across multiple modules.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Canonical text normalization: lowercase, strip punctuation, collapse whitespace.

    Used by both the lookup engine (exact-hash keys) and the embedding
    service (cache keys) to ensure consistent text representation.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()
