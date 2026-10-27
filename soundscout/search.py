from __future__ import annotations

from typing import Any

import numpy as np


def rank_by_text(query: str, records: list[dict[str, Any]], vectors: np.ndarray, embedder: Any,
                 top_k: int = 20) -> list[dict[str, Any]]:
    """Return records ranked by cosine similarity (not calibrated probabilities)."""
    if not query.strip() or not records or vectors.size == 0:
        return []
    text_vector = embedder.embed_text(query.strip())
    matrix = np.asarray(vectors, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(text_vector)), 1e-12)
    scores = (matrix @ text_vector) / np.maximum(denom, 1e-12)
    order = np.argsort(scores)[::-1]
    results = []
    for idx in order[:max(0, top_k)]:
        item = dict(records[int(idx)])
        item["similarity"] = float(scores[int(idx)])
        results.append(item)
    return results


def filter_results(results: list[dict[str, Any]], *, min_duration: float = 0.0,
                   max_duration: float = float("inf"), sample_rates: list[int] | None = None,
                   extensions: list[str] | None = None, channels: list[int] | None = None,
                   categories: list[str] | None = None) -> list[dict[str, Any]]:
    filtered = []
    for item in results:
        duration = float(item.get("duration_seconds", 0.0))
        if not (min_duration <= duration <= max_duration):
            continue
        if sample_rates and int(item.get("sample_rate", 0)) not in sample_rates:
            continue
        if extensions and item.get("extension", "") not in extensions:
            continue
        if channels and int(item.get("channels", 0)) not in channels:
            continue
        if categories and item.get("category", "Uncategorized") not in categories:
            continue
        filtered.append(item)
    return filtered
