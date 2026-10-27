from __future__ import annotations

from typing import Any

import numpy as np


def rank_by_text(query: str, records: list[dict[str, Any]], vectors: np.ndarray, embedder: Any,
                 top_k: int = 20) -> list[dict[str, Any]]:
    """Return records ranked by maximum segment cosine similarity with matching timestamp (not calibrated probabilities)."""
    if not query.strip() or not records or vectors.size == 0:
        return []

    text_vector = embedder.embed_text(query.strip())
    matrix = np.asarray(vectors, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(text_vector)), 1e-12)
    scores = (matrix @ text_vector) / np.maximum(denom, 1e-12)

    has_segments = "segment_offset" in records[0] and "segment_count" in records[0]
    results: list[dict[str, Any]] = []

    if has_segments:
        for item_raw in records:
            item = dict(item_raw)
            offset = int(item.get("segment_offset", 0))
            count = int(item.get("segment_count", 1))
            if count <= 0 or offset + count > len(scores):
                continue
            file_seg_scores = scores[offset : offset + count]
            best_rel_idx = int(np.argmax(file_seg_scores))
            best_score = float(file_seg_scores[best_rel_idx])

            seg_info_list = item.get("segments", [])
            if seg_info_list and best_rel_idx < len(seg_info_list):
                best_seg = seg_info_list[best_rel_idx]
                item["match_start_seconds"] = float(best_seg.get("start_seconds", 0.0))
                item["match_end_seconds"] = float(best_seg.get("end_seconds", item.get("duration_seconds", 0.0)))
            else:
                item["match_start_seconds"] = 0.0
                item["match_end_seconds"] = float(item.get("duration_seconds", 0.0))

            item["match_segment_index"] = best_rel_idx
            item["total_segments"] = count
            item["similarity"] = best_score
            results.append(item)
    else:
        # Fallback for flat 1:1 records/vectors
        for idx, item_raw in enumerate(records):
            if idx >= len(scores):
                break
            item = dict(item_raw)
            item["similarity"] = float(scores[idx])
            item["match_start_seconds"] = 0.0
            item["match_end_seconds"] = float(item.get("duration_seconds", 0.0))
            item["match_segment_index"] = 0
            item["total_segments"] = 1
            results.append(item)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[: max(0, top_k)]


def filter_results(results: list[dict[str, Any]], *, min_duration: float = 0.0,
                   max_duration: float = float("inf"), sample_rates: list[int] | None = None,
                   extensions: list[str] | None = None, channels: list[int] | None = None,
                   categories: list[str] | None = None, keyword: str | None = None) -> list[dict[str, Any]]:
    filtered = []
    kw = keyword.strip().lower() if keyword and keyword.strip() else None
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
        if kw:
            name = str(item.get("name", "")).lower()
            rel = str(item.get("relative_path", "")).lower()
            if kw not in name and kw not in rel:
                continue
        filtered.append(item)
    return filtered
