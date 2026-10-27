"""Fetch curated percussive one-shot sounds from Freesound One-Shot Percussive Sounds Dataset (UPF MTG / Zenodo 3665275).

Downloads a lightweight subset (40 sounds: 8 kick, 8 snare, 8 hihat, 8 clap, 8 cymbal)
using selective HTTP Range requests without downloading the entire 112 MB archive.
Includes exact licenses, author attribution, and timbral beatmaker/producer queries.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf

ZENODO_LICENSES_URL = "https://zenodo.org/api/records/3665275/files/licenses.txt/content"
ZENODO_ZIP_URL = "https://zenodo.org/api/records/3665275/files/one_shot_percussive_sounds.zip/content"

TARGET_CATEGORIES = ["kick", "snare", "hihat", "clap", "cymbal"]
SOUNDS_PER_CATEGORY = 8


class RemoteZipFile(io.RawIOBase):
    """File-like object that reads arbitrary byte ranges over HTTP without full download."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, headers={"User-Agent": "SoundScout-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            self.length = int(resp.headers.get("Content-Length", 0))
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": "SoundScout-Benchmark/1.0",
                "Range": f"bytes={self.pos}-{self.pos + size - 1}",
            },
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def fetch_percussive_subset(base_dir: Path) -> tuple[Path, Path, Path]:
    library_dir = base_dir / "percussive_evaluation_library"
    eval_dir = base_dir / "percussive_evaluation"
    library_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("--- 1. Fetching Metadata and Licenses from Zenodo 3665275 ---")
    req = urllib.request.Request(ZENODO_LICENSES_URL, headers={"User-Agent": "SoundScout/1.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        licenses_data = json.loads(resp.read().decode("utf-8"))
    print(f"Loaded metadata for {len(licenses_data)} percussive one-shot sounds.")

    print("\n--- 2. Inspecting Remote ZIP Central Directory ---")
    remote_f = RemoteZipFile(ZENODO_ZIP_URL)
    z = zipfile.ZipFile(remote_f)
    wav_entries = [name for name in z.namelist() if name.endswith(".wav")]
    id_to_zip_path: dict[str, str] = {}
    for entry in wav_entries:
        sid = entry.split("/")[-1].replace(".wav", "")
        id_to_zip_path[sid] = entry
    print(f"Indexed {len(id_to_zip_path)} audio files inside remote archive.")

    manifest_records: list[dict] = []
    category_file_ids: dict[str, list[str]] = {c: [] for c in TARGET_CATEGORIES}

    file_counter = 1
    print("\n--- 3. Extracting Curated One-Shot Transients ---")
    for cat in TARGET_CATEGORIES:
        cat_dir = library_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Filter candidate sound IDs
        candidates: list[tuple[str, dict]] = []
        for sid, meta in licenses_data.items():
            name_lower = meta.get("name", "").lower()
            if cat in name_lower and sid in id_to_zip_path:
                # avoid ambiguous cross-category matches
                if not any(other in name_lower for other in TARGET_CATEGORIES if other != cat):
                    candidates.append((sid, meta))

        selected = candidates[:SOUNDS_PER_CATEGORY]
        print(f"  Category [{cat.upper()}]: selected {len(selected)} sounds")

        for sid, meta in selected:
            file_id = f"PERC_{file_counter:02d}"
            category_file_ids[cat].append(file_id)

            orig_name = meta.get("name", f"sound_{sid}.wav")
            stem = Path(orig_name).stem
            clean_stem = "".join(c for c in stem if c.isalnum() or c in (" ", "_", "-")).strip()
            dest_filename = f"{sid}_{clean_stem}.wav"
            rel_path = f"{cat}/{dest_filename}"
            dest_path = cat_dir / dest_filename

            if not dest_path.exists() or dest_path.stat().st_size == 0:
                zip_path = id_to_zip_path[sid]
                wav_bytes = z.read(zip_path)
                dest_path.write_bytes(wav_bytes)
                print(f"    Extracted [{file_id}] {rel_path} ({len(wav_bytes)} bytes)")
            else:
                print(f"    Cached [{file_id}] {rel_path}")

            info = sf.info(dest_path)
            manifest_records.append({
                "file_id": file_id,
                "relative_path": rel_path,
                "name": dest_filename,
                "category": cat,
                "dataset": "Freesound One-Shot Percussive Sounds (UPF MTG)",
                "freesound_id": sid,
                "original_filename": orig_name,
                "author": meta.get("username", "unknown"),
                "license": meta.get("license", "Creative Commons"),
                "duration_seconds": round(float(info.duration), 2),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "description": f"Acoustic/electronic one-shot {cat}: {clean_stem}",
            })
            file_counter += 1

    # --- 4. Formulate Beatmaker / Producer Evaluation Queries ---
    queries = [
        # Broad category queries (targeting all 8 samples of the instrument)
        {
            "query_id": "Q_PERC_01",
            "query_text": "punchy acoustic kick drum transient with deep low end",
            "query_type": "transient_drum",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["kick"],
        },
        {
            "query_id": "Q_PERC_02",
            "query_text": "crisp acoustic snare drum crack with snappy wire attack",
            "query_type": "transient_drum",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["snare"],
        },
        {
            "query_id": "Q_PERC_03",
            "query_text": "metallic closed hi-hat cymbal tick transient",
            "query_type": "transient_drum",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["hihat"],
        },
        {
            "query_id": "Q_PERC_04",
            "query_text": "dry hand clap percussion snap one-shot",
            "query_type": "transient_drum",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["clap"],
        },
        {
            "query_id": "Q_PERC_05",
            "query_text": "bright metallic crash cymbal ring and wash",
            "query_type": "transient_drum",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["cymbal"],
        },
        # Fine-grained producer prompts
        {
            "query_id": "Q_PERC_06",
            "query_text": "deep sub bass kick drum one-shot",
            "query_type": "timbral_producer",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["kick"],
        },
        {
            "query_id": "Q_PERC_07",
            "query_text": "tight sharp snare attack transient",
            "query_type": "timbral_producer",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["snare"],
        },
        {
            "query_id": "Q_PERC_08",
            "query_text": "open hi-hat acoustic splash shimmer",
            "query_type": "timbral_producer",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["hihat"],
        },
        {
            "query_id": "Q_PERC_09",
            "query_text": "electronic drum machine hand clap",
            "query_type": "timbral_producer",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["clap"],
        },
        {
            "query_id": "Q_PERC_10",
            "query_text": "finger cymbal high frequency metallic choke",
            "query_type": "timbral_producer",
            "source_dataset": "Freesound-Percussive",
            "relevant_file_ids": category_file_ids["cymbal"],
        },
    ]

    manifest_file = eval_dir / "manifest.json"
    manifest_file.write_text(json.dumps({
        "version": 1,
        "total_files": len(manifest_records),
        "source": "Freesound One-Shot Percussive Sounds Dataset (UPF MTG / Zenodo 3665275)",
        "records": manifest_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    queries_file = eval_dir / "queries.json"
    queries_file.write_text(json.dumps({
        "version": 1,
        "total_queries": len(queries),
        "queries": queries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Fetched {len(manifest_records)} percussive one-shot sounds into: {library_dir}")
    print(f"[OK] Saved manifest: {manifest_file}")
    print(f"[OK] Saved queries: {queries_file}")
    return library_dir, manifest_file, queries_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=".", help="Base project directory")
    args = parser.parse_args()
    fetch_percussive_subset(Path(args.base_dir).resolve())
