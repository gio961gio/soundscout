from __future__ import annotations

import numpy as np
import pytest

from soundscout.search import (
    create_faiss_index,
    rank_by_audio_example,
    rank_by_text,
    search_faiss_index,
)


class FakeEmbedder:
    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec

    def embed_text(self, text: str) -> np.ndarray:
        return self.vec


def test_rank_by_text_contiguous_vs_fallback() -> None:
    rng = np.random.default_rng(123)
    num_files = 10
    records = []
    offset = 0
    for i in range(num_files):
        cnt = int(rng.integers(1, 4))
        records.append({
            "name": f"sample_{i}.wav",
            "duration_seconds": 10.0,
            "segment_offset": offset,
            "segment_count": cnt,
            "segments": [{"start_seconds": s * 5.0, "end_seconds": s * 5.0 + 10.0} for s in range(cnt)],
        })
        offset += cnt

    total_segments = offset
    vectors = rng.standard_normal((total_segments, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    query_vec = rng.standard_normal(512).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    embedder = FakeEmbedder(query_vec)

    # 1. Contiguous call (uses fast reduceat)
    res_fast = rank_by_text("foley impact", records, vectors, embedder, top_k=5)
    assert len(res_fast) == 5

    # 2. Modify records to be non-contiguous (triggering fallback)
    non_contig_records = [dict(r) for r in records]
    non_contig_records[1]["segment_offset"] = 999  # deliberate gap/invalid

    # Make valid non-contiguous
    non_contig_records = [dict(r) for r in records]
    # Reverse record order so offsets are not monotonically increasing from 0
    non_contig_records.reverse()
    res_fallback = rank_by_text("foley impact", non_contig_records, vectors, embedder, top_k=5)

    # The top result in res_fast should match the top result in res_fallback
    assert res_fast[0]["name"] == res_fallback[0]["name"]
    assert pytest.approx(res_fast[0]["similarity"], abs=1e-5) == res_fallback[0]["similarity"]


def test_rank_by_text_flat_records() -> None:
    rng = np.random.default_rng(42)
    records = [{"name": f"one_shot_{i}.wav", "duration_seconds": 1.0} for i in range(5)]
    vectors = rng.standard_normal((5, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    query_vec = vectors[2].copy()  # Query exactly matches record 2
    embedder = FakeEmbedder(query_vec)

    results = rank_by_text("kick", records, vectors, embedder, top_k=3)
    assert len(results) == 3
    assert results[0]["name"] == "one_shot_2.wav"
    assert pytest.approx(results[0]["similarity"], abs=1e-5) == 1.0


def test_rank_by_text_edge_cases() -> None:
    embedder = FakeEmbedder(np.ones(512, dtype=np.float32))
    assert rank_by_text("", [{"name": "test.wav"}], np.ones((1, 512)), embedder) == []
    assert rank_by_text("query", [], np.ones((1, 512)), embedder) == []
    assert rank_by_text("query", [{"name": "test.wav"}], np.array([]), embedder) == []
    assert rank_by_text("query", [{"name": "test.wav"}], np.ones((1, 512)), embedder, top_k=0) == []
    assert rank_by_text("query", [{"name": "test.wav"}], np.ones((1, 512)), embedder, top_k=-5) == []


def test_faiss_index_creation_and_search() -> None:
    rng = np.random.default_rng(777)
    vectors = rng.standard_normal((20, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = create_faiss_index(vectors)
    assert index is not None

    query = vectors[5].copy()  # Query exactly matches vector 5
    scores, indices = search_faiss_index(index, query, top_k=3)

    assert len(scores) == 3
    assert len(indices) == 3
    assert indices[0] == 5
    assert pytest.approx(scores[0], abs=1e-5) == 1.0


def test_rank_by_audio_example_finds_similar_and_excludes_self() -> None:
    rng = np.random.default_rng(999)
    records = [
        {
            "name": f"sfx_{i}.wav",
            "path": f"/samples/sfx_{i}.wav",
            "segment_offset": i * 2,
            "segment_count": 2,
            "duration_seconds": 10.0,
            "segments": [{"start_seconds": 0.0, "end_seconds": 5.0}, {"start_seconds": 5.0, "end_seconds": 10.0}],
        }
        for i in range(5)
    ]
    vectors = rng.standard_normal((10, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # Make file 3 almost identical to file 1's first segment
    vectors[6] = vectors[2].copy()

    source = records[1]
    similar = rank_by_audio_example(source, records, vectors, top_k=3, exclude_self=True)

    assert len(similar) == 3
    # Self (sfx_1.wav) must be excluded
    assert all(r["name"] != "sfx_1.wav" for r in similar)
    # Most similar should be sfx_3.wav with similarity ~ 1.0
    assert similar[0]["name"] == "sfx_3.wav"
    assert pytest.approx(similar[0]["similarity"], abs=1e-5) == 1.0


def test_rank_by_audio_example_edge_cases() -> None:
    assert rank_by_audio_example({}, [{"name": "a.wav"}], np.ones((1, 512))) == []
    assert rank_by_audio_example({"name": "a.wav"}, [], np.ones((1, 512))) == []
    assert rank_by_audio_example({"name": "a.wav"}, [{"name": "a.wav"}], np.array([])) == []
    assert rank_by_audio_example({"name": "a.wav"}, [{"name": "a.wav"}], np.ones((1, 512)), top_k=0) == []
    assert rank_by_audio_example({"name": "a.wav"}, [{"name": "a.wav"}], np.ones((1, 512)), top_k=-2) == []

