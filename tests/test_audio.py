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
