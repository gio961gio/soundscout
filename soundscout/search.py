from __future__ import annotations

from typing import Any

import numpy as np


def rank_by_text(query: str, records: list[dict[str, Any]], vectors: np.ndarray, embedder: Any,
                 top_k: int = 20) -> list[dict[str, Any]]:
    """Return records ranked by maximum segment cosine similarity with matching timestamp (not calibrated probabilities)."""
    if not query.strip() or not records or vectors.size == 0 or top_k <= 0:
        return []

    text_vector = np.asarray(embedder.embed_text(query.strip()), dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)

    t_norm = float(np.linalg.norm(text_vector))
    norm_text = text_vector / max(t_norm, 1e-12)

    # Check if vectors are unit-norm; if so, skip redundant row-norm calculation
    sample_norms = np.linalg.norm(matrix[: min(5, len(matrix))], axis=1)
    if np.all(np.abs(sample_norms - 1.0) < 0.05):
        scores = matrix @ norm_text
    else:
        denom = np.linalg.norm(matrix, axis=1)
        scores = (matrix @ norm_text) / np.maximum(denom, 1e-12)

    has_segments = "segment_offset" in records[0] and "segment_count" in records[0]
    num_records = len(records)
    effective_k = min(top_k, num_records)

    if has_segments:
        # Check if segment offsets are monotonic and contiguous for vectorized reduceat
        offsets = np.fromiter((int(r.get("segment_offset", 0)) for r in records), dtype=np.int64, count=num_records)
        counts = np.fromiter((int(r.get("segment_count", 1)) for r in records), dtype=np.int64, count=num_records)

        is_contiguous = (
            num_records > 0
            and offsets[0] == 0
            and (num_records == 1 or np.all(offsets[1:] == offsets[:-1] + counts[:-1]))
            and (offsets[-1] + counts[-1] <= len(scores))
        )

        if is_contiguous:
            max_scores = np.maximum.reduceat(scores, offsets)
            if effective_k < num_records:
                top_indices = np.argpartition(-max_scores, effective_k)[:effective_k]
                top_indices = top_indices[np.argsort(-max_scores[top_indices])]
            else:
                top_indices = np.argsort(-max_scores)

            results: list[dict[str, Any]] = []
            for f_idx in top_indices:
                item = dict(records[f_idx])
                offset = int(offsets[f_idx])
                count = int(counts[f_idx])
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
            return results

        # Fallback for non-contiguous segment offsets
        results = []
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

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:effective_k]

    # Fallback for flat 1:1 records/vectors
    valid_len = min(num_records, len(scores))
    flat_scores = scores[:valid_len]
    if effective_k < valid_len:
        top_indices = np.argpartition(-flat_scores, effective_k)[:effective_k]
        top_indices = top_indices[np.argsort(-flat_scores[top_indices])]
    else:
        top_indices = np.argsort(-flat_scores)

    results = []
    for idx in top_indices:
        item = dict(records[idx])
        item["similarity"] = float(flat_scores[idx])
        item["match_start_seconds"] = 0.0
        item["match_end_seconds"] = float(item.get("duration_seconds", 0.0))
        item["match_segment_index"] = 0
        item["total_segments"] = 1
        results.append(item)

    return results


def create_faiss_index(vectors: np.ndarray) -> Any:
    """Create a FAISS IndexFlatIP (Inner Product) for accelerated vector search."""
    try:
        import faiss
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = matrix.shape[1] if matrix.ndim == 2 else 512
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        return index
    except (ImportError, Exception):
        return None


def search_faiss_index(index: Any, query_vector: np.ndarray, top_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Execute accelerated top-K search on a FAISS index, returning (scores, indices)."""
    q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
    scores, indices = index.search(q, top_k)
    return scores[0], indices[0]


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
