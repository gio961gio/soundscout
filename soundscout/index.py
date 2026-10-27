from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Any

import numpy as np

from .audio import SUPPORTED_EXTENSIONS, TARGET_SAMPLE_RATE, audio_segments_timed, load_audio
MODEL_ID = "laion/clap-htsat-unfused"

CACHE_ROOT = Path.home() / ".soundscout" / "indexes"
INDEX_VERSION = 2


def cache_dir_for(root: str | Path) -> Path:
    canonical = str(Path(root).expanduser().resolve())
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return CACHE_ROOT / digest


def load_saved_index(root: str | Path) -> tuple[list[dict[str, Any]], np.ndarray] | None:
    cache_dir = cache_dir_for(root)
    manifest_path = cache_dir / "manifest.json"
    vectors_path = cache_dir / "embeddings.npy"
    if not manifest_path.exists() or not vectors_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_id") != MODEL_ID or manifest.get("index_version") != INDEX_VERSION:
            return None
        vectors = np.load(vectors_path, allow_pickle=False)
        records = manifest.get("records", [])
        if records and all("segment_count" in r and "segment_offset" in r for r in records):
            expected_segs = sum(int(r.get("segment_count", 1)) for r in records)
            if len(vectors) != expected_segs:
                return None
        elif len(records) != len(vectors):
            return None
        return records, vectors.astype(np.float32, copy=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def build_or_update_index(
    root: str | Path,
    embedder: Any,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"La cartella non esiste o non è una directory: {root}")

    paths = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if not paths:
        raise ValueError(
            "Non ho trovato file supportati. Per il primo MVP usa WAV, FLAC, OGG o AIFF; "
            "gli MP3 richiedono FFmpeg se non sono leggibili direttamente."
        )

    cache_dir = cache_dir_for(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous = load_saved_index(root)
    old_records, old_vectors = previous if previous else ([], np.empty((0, 0), dtype=np.float32))
    old_by_path = {str(item["path"]): (item, i) for i, item in enumerate(old_records)}

    new_records: list[dict[str, Any]] = []
    new_vectors: list[np.ndarray] = []
    errors: list[dict[str, str]] = []
    reused = 0
    embedded = 0
    total = len(paths)
    current_segment_offset = 0

    for i, path in enumerate(paths, start=1):
        path_key = str(path.resolve())
        try:
            stat = path.stat()
            old_item = old_by_path.get(path_key)
            if old_item and old_vectors.ndim == 2:
                old_record, _ = old_item
                has_seg_meta = "segment_offset" in old_record and "segment_count" in old_record
                unchanged = (
                    has_seg_meta
                    and int(old_record.get("size_bytes", -1)) == int(stat.st_size)
                    and int(old_record.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
                    and old_vectors.shape[1] > 0
                )
            else:
                unchanged = False

            if unchanged:
                old_offset = int(old_item[0]["segment_offset"])
                old_count = int(old_item[0]["segment_count"])
                file_vectors = np.atleast_2d(old_vectors[old_offset : old_offset + old_count]).astype(np.float32, copy=False)
                record = dict(old_item[0])
                record["segment_offset"] = current_segment_offset
                record["segment_count"] = old_count
                reused += 1
            else:
                decoded = load_audio(path)
                timed_segs = audio_segments_timed(decoded.mono, TARGET_SAMPLE_RATE)
                seg_audios = [s.audio for s in timed_segs]
                if hasattr(embedder, "embed_audio_segments"):
                    raw_vectors = embedder.embed_audio_segments(seg_audios, TARGET_SAMPLE_RATE)
                else:
                    raw_vectors = embedder.embed_audio(seg_audios, TARGET_SAMPLE_RATE)
                file_vectors = np.atleast_2d(raw_vectors).astype(np.float32, copy=False)

                relative = path.relative_to(root)
                category = relative.parts[0] if len(relative.parts) > 1 else "Uncategorized"
                record = {
                    "path": path_key,
                    "name": path.name,
                    "relative_path": relative.as_posix(),
                    "category": category,
                    "extension": path.suffix.lower().lstrip("."),
                    "duration_seconds": round(decoded.duration_seconds, 4),
                    "sample_rate": int(decoded.original_sample_rate),
                    "channels": int(decoded.channels),
                    "frames": int(decoded.frames),
                    "file_format": decoded.file_format,
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "segment_offset": current_segment_offset,
                    "segment_count": len(file_vectors),
                    "segments": [
                        {
                            "start_seconds": s.start_seconds,
                            "end_seconds": s.end_seconds,
                            "segment_index": s.segment_index,
                        }
                        for s in timed_segs[:len(file_vectors)]
                    ],
                }
                embedded += 1

            new_records.append(record)
            new_vectors.append(file_vectors)
            current_segment_offset += len(file_vectors)
        except Exception as exc:  # keep indexing the rest of the user's library
            errors.append({"file": str(path), "error": str(exc)[:350]})
        if progress_callback:
            progress_callback(i, total, path.name)

    if new_vectors:
        matrix = np.vstack(new_vectors).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, 512), dtype=np.float32)

    manifest = {
        "index_version": INDEX_VERSION,
        "model_id": MODEL_ID,
        "root": str(root),
        "created_at_epoch": int(__import__("time").time()),
        "records": new_records,
        "errors": errors,
    }
    # Atomic replacement reduces the chance of leaving a half-written index on interruption.
    vectors_tmp = cache_dir / "embeddings.tmp.npy"
    manifest_tmp = cache_dir / "manifest.tmp.json"
    np.save(vectors_tmp, matrix, allow_pickle=False)
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(vectors_tmp, cache_dir / "embeddings.npy")
    os.replace(manifest_tmp, cache_dir / "manifest.json")

    return {
        "root": str(root),
        "total_found": total,
        "indexed": len(new_records),
        "embedded_now": embedded,
        "reused": reused,
        "errors": errors,
        "cache_dir": str(cache_dir),
    }
