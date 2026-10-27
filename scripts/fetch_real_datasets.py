"""Fetch controlled real-world audio benchmark subsets from ESC-50 and Clotho.

Total size: ~45 MB (50 real audio files: 30 from ESC-50, 20 from Clotho).
Includes authentic crowdsourced captions from DCASE Clotho and environmental audio from ESC-50.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import soundfile as sf

ESC50_CSV_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO_BASE = "https://github.com/karolpiczak/ESC-50/raw/master/audio/"

CLOTHO_CSV_URL = "https://zenodo.org/api/records/4783391/files/clotho_captions_development.csv/content"
CLOTHO_HF_BASE = "https://huggingface.co/datasets/AdoptedIrelia/clotho/resolve/main/clotho/development/"

ESC50_CATEGORIES = {
    "glass_breaking": "shattering glass breaking into sharp pieces",
    "footsteps": "human footsteps walking and stepping on hard floor",
    "door_wood_knock": "knocking loudly on a solid wooden door",
    "rain": "outdoor heavy rain downpour falling",
    "thunderstorm": "loud thunderstorm with rolling thunder rumble",
    "wind": "blowing howling gusts of cold wind",
    "clock_tick": "analog clock ticking rhythmic mechanism",
    "engine": "motor vehicle car engine running and idling",
    "chainsaw": "buzzing electric chainsaw cutting",
    "siren": "emergency police or ambulance siren wailing",
}


def download_url(url: str, output_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "SoundScout-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)


def fetch_datasets(base_dir: Path) -> tuple[Path, Path, Path]:
    library_dir = base_dir / "real_evaluation_library"
    eval_dir = base_dir / "real_evaluation"
    library_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict] = []
    queries: list[dict] = []

    # --- 1. ESC-50 SUBSET ---
    print("\n--- 1. Fetching ESC-50 Subset (10 categories x 3 files = 30 files) ---")
    req = urllib.request.Request(ESC50_CSV_URL, headers={"User-Agent": "SoundScout/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        esc_csv_text = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(esc_csv_text))
    esc_by_cat: dict[str, list[str]] = {}
    for row in reader:
        c = row["category"]
        if c in ESC50_CATEGORIES:
            esc_by_cat.setdefault(c, []).append(row["filename"])

    esc_idx = 1
    for cat_idx, (cat_name, cat_query) in enumerate(ESC50_CATEGORIES.items(), start=1):
        filenames = esc_by_cat.get(cat_name, [])[:3]
        cat_file_ids = []
        for fname in filenames:
            file_id = f"ESC_{esc_idx:02d}"
            cat_file_ids.append(file_id)
            rel_path = f"ESC50/{cat_name}/{fname}"
            dest = library_dir / "ESC50" / cat_name / fname

            if not dest.exists() or dest.stat().st_size == 0:
                url = f"{ESC50_AUDIO_BASE}{fname}"
                print(f"  Downloading [{file_id}] {cat_name}/{fname}...")
                download_url(url, dest)
            else:
                print(f"  Cached [{file_id}] {cat_name}/{fname}")

            info = sf.info(dest)
            manifest_records.append({
                "file_id": file_id,
                "relative_path": rel_path,
                "name": fname,
                "category": cat_name,
                "dataset": "ESC-50",
                "duration_seconds": round(float(info.duration), 2),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "description": f"Real environmental audio: {cat_name.replace('_', ' ')}",
            })
            esc_idx += 1

        queries.append({
            "query_id": f"Q_ESC_{cat_idx:02d}",
            "query_text": cat_query,
            "query_type": "environmental",
            "source_dataset": "ESC-50",
            "relevant_file_ids": cat_file_ids,
        })

    # --- 2. CLOTHO SUBSET ---
    print("\n--- 2. Fetching Clotho DCASE Subset (20 real audio files + human captions) ---")
    req = urllib.request.Request(CLOTHO_CSV_URL, headers={"User-Agent": "SoundScout/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        clotho_csv_text = resp.read().decode("utf-8", errors="ignore")

    clotho_reader = csv.DictReader(io.StringIO(clotho_csv_text))
    clotho_rows = list(clotho_reader)

    # Pick 20 files
    target_clotho = clotho_rows[:20]
    for c_idx, row in enumerate(target_clotho, start=1):
        file_id = f"CLO_{c_idx:02d}"
        fname = row["file_name"]
        captions = [
            row.get(f"caption_{i}", "").strip()
            for i in range(1, 6)
            if row.get(f"caption_{i}", "").strip()
        ]
        primary_caption = captions[0] if captions else "Ambient audio recording"

        # Sanitize local filename
        clean_name = "".join(c for c in fname if c.isalnum() or c in (" ", ".", "_", "-")).strip()
        rel_path = f"Clotho/{clean_name}"
        dest = library_dir / "Clotho" / clean_name

        if not dest.exists() or dest.stat().st_size == 0:
            quoted_name = urllib.parse.quote(fname)
            url = f"{CLOTHO_HF_BASE}{quoted_name}"
            print(f"  Downloading [{file_id}] Clotho: {clean_name[:40]}...")
            try:
                download_url(url, dest)
            except Exception as exc:
                print(f"    Failed downloading from Hugging Face ({exc})")
                raise exc
        else:
            print(f"  Cached [{file_id}] Clotho: {clean_name[:40]}")

        info = sf.info(dest)
        manifest_records.append({
            "file_id": file_id,
            "relative_path": rel_path,
            "name": clean_name,
            "category": "Clotho-DCASE",
            "dataset": "Clotho",
            "duration_seconds": round(float(info.duration), 2),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "description": primary_caption,
            "all_human_captions": captions,
        })

        # Authentic human query from caption 1
        queries.append({
            "query_id": f"Q_CLO_{c_idx:02d}",
            "query_text": primary_caption,
            "query_type": "human_caption",
            "source_dataset": "Clotho",
            "relevant_file_ids": [file_id],
        })

    # Save Real Manifest & Queries
    manifest_file = eval_dir / "manifest.json"
    manifest_file.write_text(json.dumps({
        "version": 1,
        "total_files": len(manifest_records),
        "source": "ESC-50 (GitHub) & Clotho DCASE (Tampere University / Zenodo)",
        "records": manifest_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    queries_file = eval_dir / "queries.json"
    queries_file.write_text(json.dumps({
        "version": 1,
        "total_queries": len(queries),
        "queries": queries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Fetched {len(manifest_records)} real audio files into: {library_dir}")
    print(f"[OK] Generated manifest: {manifest_file}")
    print(f"[OK] Generated queries: {queries_file}")
    return library_dir, manifest_file, queries_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=".", help="Base project directory")
    args = parser.parse_args()
    fetch_datasets(Path(args.base_dir).resolve())
