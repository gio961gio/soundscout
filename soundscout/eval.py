from __future__ import annotations

from typing import Any, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    """Calculate Recall@K: fraction of relevant items retrieved in top-K results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k.intersection(relevant_set))
    return float(hits / len(relevant_set))


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    """Calculate Precision@K: fraction of top-K results that are relevant."""
    if k <= 0:
        return 0.0
    effective_k = min(k, len(retrieved_ids))
    if effective_k == 0:
        return 0.0
    top_k = set(retrieved_ids[:effective_k])
    relevant_set = set(relevant_ids)
    hits = len(top_k.intersection(relevant_set))
    return float(hits / k)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Calculate Reciprocal Rank (RR): 1 / rank of the first relevant item (1-based), or 0.0."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_set:
            return float(1.0 / rank)
    return 0.0


def evaluate_query(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Compute all standard retrieval metrics for a single query."""
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k=k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k=k)
    metrics["mrr"] = reciprocal_rank(retrieved_ids, relevant_ids)
    return metrics


def aggregate_retrieval_metrics(query_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Average metrics across all evaluated queries."""
    if not query_metrics:
        return {}
    all_numeric_keys: set[str] = set()
    for m in query_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                all_numeric_keys.add(k)
    averages: dict[str, float] = {}
    total = len(query_metrics)
    for k in sorted(all_numeric_keys):
        averages[k] = round(sum(float(m.get(k, 0.0)) for m in query_metrics) / total, 4)
    return averages
