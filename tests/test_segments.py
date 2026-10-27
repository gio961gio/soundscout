import numpy as np
import pytest
import soundfile as sf

from soundscout.audio import audio_segments_timed
from soundscout.search import rank_by_text


def test_audio_segments_timed_short_file():
    sr = 48_000
    # 3 seconds audio
    audio = np.zeros(sr * 3, dtype=np.float32)
    segments = audio_segments_timed(audio, sample_rate=sr, segment_seconds=10.0)

    assert len(segments) == 1
    assert segments[0].start_seconds == 0.0
    assert segments[0].end_seconds == 3.0
    assert segments[0].segment_index == 0
    assert len(segments[0].audio) == sr * 3


def test_audio_segments_timed_long_file():
    sr = 48_000
    # 22 seconds audio with 10s window and 5s hop
    audio = np.zeros(sr * 22, dtype=np.float32)
    segments = audio_segments_timed(audio, sample_rate=sr, segment_seconds=10.0, hop_seconds=5.0)

    # Starts: 0s, 5s, 10s, 15s (15 to 22 is 7s <= window)
    assert len(segments) == 4
    assert segments[0].start_seconds == 0.0
    assert segments[0].end_seconds == 10.0

    assert segments[1].start_seconds == 5.0
    assert segments[1].end_seconds == 15.0

    assert segments[2].start_seconds == 10.0
    assert segments[2].end_seconds == 20.0

    assert segments[3].start_seconds == 15.0
    assert segments[3].end_seconds == 22.0


def test_rank_by_text_pinpoints_exact_segment_timestamp():
    class ToyEmbedder:
        def embed_text(self, _text: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    records = [
        {
            "name": "long_recording.wav",
            "path": "/audio/long_recording.wav",
            "duration_seconds": 30.0,
            "segment_offset": 0,
            "segment_count": 3,
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 10.0, "segment_index": 0},
                {"start_seconds": 10.0, "end_seconds": 20.0, "segment_index": 1},
                {"start_seconds": 20.0, "end_seconds": 30.0, "segment_index": 2},
            ],
        },
        {
            "name": "unrelated_noise.wav",
            "path": "/audio/unrelated_noise.wav",
            "duration_seconds": 10.0,
            "segment_offset": 3,
            "segment_count": 1,
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 10.0, "segment_index": 0},
            ],
        },
    ]

    # Vectors:
    # long_recording seg 0: [0.0, 1.0] (mismatch)
    # long_recording seg 1: [1.0, 0.0] (PERFECT MATCH at 10s-20s!)
    # long_recording seg 2: [0.0, 1.0] (mismatch)
    # unrelated_noise seg 0: [0.2, 0.8] (low similarity)
    vectors = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.8],
    ], dtype=np.float32)

    results = rank_by_text("target event", records, vectors, ToyEmbedder())

    assert len(results) == 2
    top = results[0]
    assert top["name"] == "long_recording.wav"
    assert top["similarity"] == pytest.approx(1.0)
    assert top["match_segment_index"] == 1
    assert top["match_start_seconds"] == 10.0
    assert top["match_end_seconds"] == 20.0
    assert top["total_segments"] == 3


def test_index_with_segments_and_incremental_reuse(tmp_path, monkeypatch):
    from soundscout import index as index_module

    class DummySegmentEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_audio_segments(self, segments, sample_rate):
            self.calls += 1
            # Return distinct vector per segment
            return np.ones((len(segments), 4), dtype=np.float32)

    library = tmp_path / "lib"
    library.mkdir()
    sr = 48_000
    # 22s audio file -> multiple segments
    signal = np.zeros(sr * 22, dtype=np.float32)
    sf.write(library / "long_sample.wav", signal, sr)

    monkeypatch.setattr(index_module, "CACHE_ROOT", tmp_path / "cache")
    embedder = DummySegmentEmbedder()

    # First pass
    stats1 = index_module.build_or_update_index(library, embedder)
    assert stats1["indexed"] == 1
    assert stats1["embedded_now"] == 1
    assert stats1["reused"] == 0
    assert embedder.calls == 1

    # Verify loaded index
    saved = index_module.load_saved_index(library)
    assert saved is not None
    records, vectors = saved
    assert len(records) == 1
    assert records[0]["segment_count"] == 4
    assert vectors.shape == (4, 4)

    # Second pass: incremental reuse
    stats2 = index_module.build_or_update_index(library, embedder)
    assert stats2["indexed"] == 1
    assert stats2["embedded_now"] == 0
    assert stats2["reused"] == 1
    assert embedder.calls == 1  # No re-embedding!


def test_index_heterogeneous_durations_multisegment(tmp_path, monkeypatch):
    """Explicitly verify indexing files with different durations (e.g. 3s and 22s) yields stacked 2D matrix."""
    from soundscout import index as index_module

    class DummySegmentEmbedder:
        def embed_audio_segments(self, segments, sample_rate):
            return np.ones((len(segments), 8), dtype=np.float32)

    library = tmp_path / "mixed_durations"
    library.mkdir()
    sr = 48_000

    # File 1: 3 seconds -> 1 segment
    sf.write(library / "short.wav", np.zeros(sr * 3, dtype=np.float32), sr)
    # File 2: 22 seconds -> 4 segments
    sf.write(library / "long.wav", np.zeros(sr * 22, dtype=np.float32), sr)

    monkeypatch.setattr(index_module, "CACHE_ROOT", tmp_path / "cache")
    embedder = DummySegmentEmbedder()

    stats = index_module.build_or_update_index(library, embedder)
    assert stats["indexed"] == 2
    assert stats["embedded_now"] == 2

    saved = index_module.load_saved_index(library)
    assert saved is not None
    records, vectors = saved

    assert len(records) == 2
    # Verify segment counts
    counts = {r["name"]: r["segment_count"] for r in records}
    assert counts["short.wav"] == 1
    assert counts["long.wav"] == 4
    # Total segments = 1 + 4 = 5
    assert vectors.shape == (5, 8)

