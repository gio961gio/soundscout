import numpy as np

from soundscout.audio import audio_segments
from soundscout.search import filter_results, rank_by_text


def test_audio_segments_short_signal_returns_single_segment():
    x = np.zeros(8_000, dtype=np.float32)
    segments = audio_segments(x, sample_rate=1_000, segment_seconds=10)
    assert len(segments) == 1
    assert len(segments[0]) == 8_000


def test_audio_segments_long_signal_caps_number_and_length():
    x = np.arange(60_000, dtype=np.float32)
    segments = audio_segments(x, sample_rate=1_000, segment_seconds=10, max_segments=3)
    assert len(segments) == 3
    assert all(len(segment) == 10_000 for segment in segments)
    assert segments[0][0] == 0
    assert segments[-1][-1] == x[-1]


def test_filter_results_uses_metadata_filters():
    data = [
        {"duration_seconds": 1.0, "sample_rate": 44100, "extension": "wav", "channels": 2, "category": "Foley"},
        {"duration_seconds": 7.0, "sample_rate": 48000, "extension": "flac", "channels": 1, "category": "Ambience"},
    ]
    result = filter_results(data, min_duration=0.5, max_duration=2, sample_rates=[44100])
    assert len(result) == 1
    assert result[0]["category"] == "Foley"


class FakeEmbedder:
    def embed_text(self, _text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_rank_by_text_orders_by_cosine_similarity():
    records = [{"name": "near"}, {"name": "far"}]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    results = rank_by_text("metal impact", records, vectors, FakeEmbedder())
    assert [r["name"] for r in results] == ["near", "far"]
    assert results[0]["similarity"] > results[1]["similarity"]


def test_index_reuses_embeddings_for_unchanged_files(tmp_path, monkeypatch):
    import soundfile as sf
    from soundscout import index as index_module

    class FakeAudioEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_audio(self, segments, sample_rate):
            self.calls += 1
            return np.array([1.0, 0.5, 0.25], dtype=np.float32)

    library = tmp_path / "library"
    library.mkdir()
    signal = (0.15 * np.sin(2 * np.pi * 330 * np.arange(4800) / 48000)).astype(np.float32)
    sf.write(library / "tone.wav", signal, 48000)
    monkeypatch.setattr(index_module, "CACHE_ROOT", tmp_path / "cache")
    embedder = FakeAudioEmbedder()

    first = index_module.build_or_update_index(library, embedder)
    assert first["indexed"] == 1
    assert first["embedded_now"] == 1
    assert first["reused"] == 0
    assert embedder.calls == 1

    second = index_module.build_or_update_index(library, embedder)
    assert second["indexed"] == 1
    assert second["embedded_now"] == 0
    assert second["reused"] == 1
    assert embedder.calls == 1

    saved = index_module.load_saved_index(library)
    assert saved is not None
    records, vectors = saved
    assert records[0]["name"] == "tone.wav"
    assert vectors.shape == (1, 3)


def test_audio_segments_empty_raises_value_error():
    import pytest
    with pytest.raises(ValueError, match="vuoto"):
        audio_segments(np.array([], dtype=np.float32))


def test_rank_by_text_empty_query_or_empty_records():
    embedder = FakeEmbedder()
    assert rank_by_text("", [{"name": "a"}], np.array([[1.0, 0.0]]), embedder) == []
    assert rank_by_text("   ", [{"name": "a"}], np.array([[1.0, 0.0]]), embedder) == []
    assert rank_by_text("query", [], np.array([]), embedder) == []


def test_index_raises_on_nonexistent_or_empty_dir(tmp_path):
    from soundscout import index as index_module
    import pytest

    # Nonexistent
    with pytest.raises(ValueError, match="non esiste o non è una directory"):
        index_module.build_or_update_index(tmp_path / "does_not_exist", FakeEmbedder())

    # Empty
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="Non ho trovato file supportati"):
        index_module.build_or_update_index(empty_dir, FakeEmbedder())


def test_index_handles_corrupted_file_gracefully(tmp_path, monkeypatch):
    import soundfile as sf
    from soundscout import index as index_module

    class DummyAudioEmbedder:
        def embed_audio(self, segments, sample_rate):
            return np.array([0.5, 0.5], dtype=np.float32)

    library = tmp_path / "mixed_library"
    library.mkdir()

    # Valid audio
    signal = (0.1 * np.sin(2 * np.pi * 440 * np.arange(4800) / 48000)).astype(np.float32)
    sf.write(library / "valid.wav", signal, 48000)

    # Corrupted audio file with .wav extension
    corrupt_file = library / "broken.wav"
    corrupt_file.write_bytes(b"CORRUPTED_NOT_A_VALID_RIFF_HEADER")

    monkeypatch.setattr(index_module, "CACHE_ROOT", tmp_path / "cache")
    stats = index_module.build_or_update_index(library, DummyAudioEmbedder())

    assert stats["indexed"] == 1
    assert stats["embedded_now"] == 1
    assert len(stats["errors"]) == 1
    assert "broken.wav" in stats["errors"][0]["file"]

