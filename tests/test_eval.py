import pytest

from soundscout.eval import (
    aggregate_retrieval_metrics,
    evaluate_query,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_normal_and_edge_cases():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["b", "e", "z"]

    # In top 3: ['a', 'b', 'c'] -> hits = {'b'} (1 out of 3)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    # In top 5: ['a', 'b', 'c', 'd', 'e'] -> hits = {'b', 'e'} (2 out of 3)
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)

    # In top 1: ['a'] -> hits = 0
    assert recall_at_k(retrieved, relevant, k=1) == 0.0

    # Empty relevant
    assert recall_at_k(retrieved, [], k=5) == 0.0

    # Empty retrieved
    assert recall_at_k([], relevant, k=5) == 0.0


def test_precision_at_k_normal_and_edge_cases():
    retrieved = ["hit1", "miss1", "hit2", "miss2", "miss3"]
    relevant = ["hit1", "hit2"]

    # Precision@1: 1 hit in 1 result = 1.0
    assert precision_at_k(retrieved, relevant, k=1) == 1.0

    # Precision@3: 2 hits in 3 results = 2/3
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    # Precision@5: 2 hits in 5 results = 2/5 = 0.4
    assert precision_at_k(retrieved, relevant, k=5) == 0.4

    # Empty retrieved
    assert precision_at_k([], relevant, k=5) == 0.0

    # Zero k
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_reciprocal_rank_calculation():
    # First item is relevant -> 1/1 = 1.0
    assert reciprocal_rank(["hit", "miss"], ["hit"]) == 1.0

    # Second item is relevant -> 1/2 = 0.5
    assert reciprocal_rank(["miss", "hit"], ["hit"]) == 0.5

    # Third item is relevant -> 1/3
    assert reciprocal_rank(["m1", "m2", "hit", "m3"], ["hit"]) == pytest.approx(1 / 3)

    # No item relevant -> 0.0
    assert reciprocal_rank(["m1", "m2"], ["hit"]) == 0.0

    # Empty list
    assert reciprocal_rank([], ["hit"]) == 0.0
    assert reciprocal_rank(["hit"], []) == 0.0


def test_evaluate_query_and_aggregation():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["a"]

    metrics = evaluate_query(retrieved, relevant, k_values=(1, 3, 5))
    assert metrics["recall@1"] == 1.0
    assert metrics["precision@1"] == 1.0
    assert metrics["mrr"] == 1.0

    # Test aggregate
    q1 = {"recall@5": 1.0, "precision@5": 0.2, "mrr": 1.0}
    q2 = {"recall@5": 0.5, "precision@5": 0.1, "mrr": 0.5}
    avg = aggregate_retrieval_metrics([q1, q2])

    assert avg["recall@5"] == 0.75
    assert avg["precision@5"] == 0.15
    assert avg["mrr"] == 0.75
