from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 48_000
MAX_CLAP_SECONDS = 10
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}


@dataclass
class AudioData:
    mono: np.ndarray
    original_sample_rate: int
    channels: int
    duration_seconds: float
    frames: int
    file_format: str


@dataclass
class TimedSegment:
    audio: np.ndarray
    start_seconds: float
    end_seconds: float
    segment_index: int


def _ffprobe_metadata(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        raise RuntimeError(
            f"Formato non leggibile da libsndfile: {path.suffix}. "
            "Per MP3 installa FFmpeg (ffmpeg + ffprobe) oppure converti il file in WAV."
        )
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,duration,duration_ts,time_base,nb_frames",
            "-show_entries", "format=duration,format_name",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Impossibile leggere i metadati con ffprobe.")
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams", [])
    fmt = payload.get("format", {})
    stream = streams[0] if streams else {}
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    return {
        "sample_rate": int(stream.get("sample_rate") or TARGET_SAMPLE_RATE),
        "channels": int(stream.get("channels") or 1),
        "duration": max(0.0, duration),
        "format": str(fmt.get("format_name") or path.suffix.lstrip(".")).split(",")[0],
    }


def load_audio(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> AudioData:
    """Read a local audio file, convert to mono float32 and resample for CLAP."""
    path = Path(path)
    try:
        audio, source_sr = sf.read(path, dtype="float32", always_2d=True)
        channels = int(audio.shape[1])
        frames = int(audio.shape[0])
        duration = frames / float(source_sr) if source_sr else 0.0
        fmt = path.suffix.lstrip(".").lower()
        mono = audio.mean(axis=1, dtype=np.float32)
        if source_sr != target_sample_rate:
            from math import gcd
            divisor = gcd(int(source_sr), int(target_sample_rate))
            mono = resample_poly(
                mono,
                target_sample_rate // divisor,
                int(source_sr) // divisor,
            ).astype(np.float32, copy=False)
    except (RuntimeError, sf.LibsndfileError) as exc:
        if path.suffix.lower() == ".mp3" and shutil.which("ffmpeg"):
            metadata = _ffprobe_metadata(path)
            proc = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                    "-acodec", "pcm_f32le", "-ar", str(target_sample_rate), "-ac", "1", "pipe:1",
                ],
                capture_output=True, check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore") or "Decodifica MP3 fallita.")
            mono = np.frombuffer(proc.stdout, dtype="<f4").copy()
            source_sr = metadata["sample_rate"]
            channels = metadata["channels"]
            duration = metadata["duration"]
            frames = int(round(duration * source_sr))
            fmt = "mp3"
        elif path.suffix.lower() == ".mp3":
            raise RuntimeError(
                f"Non riesco a decodificare {path.name}. "
                "Per MP3 installa FFmpeg, oppure converti il file in WAV/FLAC."
            ) from exc
        else:
            raise RuntimeError(
                f"Non riesco a decodificare {path.name}: file corrotto o formato non supportato da libsndfile."
            ) from exc

    if mono.size == 0:
        raise ValueError("Il file audio è vuoto.")
    mono = np.nan_to_num(mono, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return AudioData(
        mono=mono,
        original_sample_rate=int(source_sr),
        channels=channels,
        duration_seconds=float(duration),
        frames=frames,
        file_format=fmt,
    )


def audio_segments_timed(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    segment_seconds: float = MAX_CLAP_SECONDS,
    hop_seconds: float = 5.0,
    max_segments: int = 12,
) -> list[TimedSegment]:
    """Return sequential windows with exact start and end timestamps (seconds)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Il file audio è vuoto.")
    total_samples = len(audio)
    duration = total_samples / float(sample_rate)
    window_samples = max(1, int(sample_rate * segment_seconds))

    if total_samples <= window_samples:
        return [
            TimedSegment(
                audio=audio,
                start_seconds=0.0,
                end_seconds=round(duration, 3),
                segment_index=0,
            )
        ]

    hop_samples = max(1, int(sample_rate * hop_seconds))
    segments: list[TimedSegment] = []
    start = 0
    seg_idx = 0

    while start < total_samples and seg_idx < max_segments:
        end = min(start + window_samples, total_samples)
        seg_audio = audio[start:end]
        start_sec = round(start / float(sample_rate), 3)
        end_sec = round(end / float(sample_rate), 3)
        segments.append(
            TimedSegment(
                audio=seg_audio,
                start_seconds=start_sec,
                end_seconds=end_sec,
                segment_index=seg_idx,
            )
        )
        seg_idx += 1
        if end >= total_samples:
            break
        start += hop_samples

    return segments


def audio_segments(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE,
                   segment_seconds: int = MAX_CLAP_SECONDS, max_segments: int = 3) -> list[np.ndarray]:
    """Return up to three 10-second windows across the file for robust indexing (legacy compatibility)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Il file audio è vuoto.")
    window = max(1, int(sample_rate * segment_seconds))
    if len(audio) <= window:
        return [audio]
    starts = [0, max(0, (len(audio) - window) // 2), max(0, len(audio) - window)]
    unique_starts: list[int] = []
    for start in starts:
        if start not in unique_starts:
            unique_starts.append(start)
    if len(unique_starts) > max_segments:
        unique_starts = unique_starts[:max_segments]
    return [audio[start:start + window] for start in unique_starts]


def waveform_minmax(path: str | Path, bins: int = 500, max_seconds: int = 120) -> tuple[np.ndarray, np.ndarray] | None:
    """Create a compact min/max waveform preview without loading an unlimited duration."""
    path = Path(path)
    try:
        info = sf.info(path)
        frames_to_read = min(int(info.frames), int(info.samplerate * max_seconds))
        signal, _ = sf.read(path, dtype="float32", start=0, frames=frames_to_read, always_2d=True)
        if signal.size == 0:
            return None
        mono = signal.mean(axis=1)
        step = max(1, int(np.ceil(len(mono) / bins)))
        mins, maxs = [], []
        for start in range(0, len(mono), step):
            block = mono[start:start + step]
            mins.append(float(np.min(block)))
            maxs.append(float(np.max(block)))
        x = np.linspace(0, len(mono) / float(info.samplerate), num=len(mins), dtype=np.float32)
        return x, np.vstack([mins, maxs]).T
    except Exception:
        return None
