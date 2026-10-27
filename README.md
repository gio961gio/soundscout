# SoundScout

**Natural-language search for local sound-effect and sample libraries.**

SoundScout indexes audio files on your computer and lets you search them with descriptions such as *"distant metallic impact with long reverb"*. It uses CLAP (Contrastive Language-Audio Pretraining) to embed audio and text in a shared representation space, then ranks files by cosine similarity.

> Portfolio MVP for audio / music technology internships. Search scores are ranking signals, not calibrated probabilities; audition the files before choosing them.

## What works in SoundScout

- **Semantic Text-to-Audio Retrieval:** powered by CLAP (`laion/clap-htsat-unfused`) with 100% local-first privacy.
- **Segment-Level Search & Timestamp Detection:** multi-segment sliding windows (10s window, 5s hop) with max-pooling per file, pinpointing the exact event start and end time.
- **Interactive Waveform Highlight:** Plotly waveform displaying the salient matched range in an overlay color with auto-seeking audio playback.
- **Session Shortlist:** collect, review, and manage favorite candidate sounds during sound design sessions.
- **DAW & Cue Sheet CSV Export:** one-click export formatted with standard broadcast timecodes (`HH:MM:SS.mmm`) and query tracking, encoded in `utf-8-sig` for instant Excel / DAW compatibility.
- **Hybrid Search:** combine natural language semantic prompts with filename / folder keyword filters.
- **Incremental Cache v2:** unchanged files reuse saved embeddings on subsequent runs; 100% vector reuse without re-reading audio.
- **Multi-Format Decoder:** WAV, FLAC, OGG, AIFF, and MP3 (with FFmpeg fallback).
- **Automated Retrieval Benchmark:** formal test harness measuring Recall@K, Precision@K, and MRR.

## Quick start

Python 3.10 or 3.11 is a good starting point. A modern CPU is enough for a small test library; indexing is faster with a compatible CUDA GPU. The initial model download and first indexing pass can take a while.

### 1. Create an environment

macOS / Linux:

```bash
cd soundscout_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd soundscout_project
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If pip reports a PyTorch installation problem, install the suitable PyTorch build for your OS from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), then install the remaining requirements.

### 2. Launch the app

```bash
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

### 3. Index a sound library

In the sidebar, enter the **absolute path** to a folder containing your samples, then click **Index / update library**. Subfolders are scanned automatically. Example folder layout:

```text
MySamples/
├── Foley/
│   ├── SFX_0247.wav
│   └── footsteps_03.wav
├── Ambience/
│   └── forest_night.flac
└── Impacts/
    └── metal_hit.wav
```

The index cache is stored separately under `~/.soundscout/indexes/`, not inside your sample folder. When files change, run **Index / update library** again; unchanged embeddings are reused.

### 4. Search

Try concise, descriptive prompts, for example:

- `a heavy metallic impact with a long reverb tail`
- `distant crowd ambience in a large indoor space`
- `short dry hand clap`
- `low rumbling engine starting`
- `soft footsteps on gravel`

CLAP is often a better starting point with English descriptions. Results can vary by sound domain and query wording, so audition multiple candidates.

## Architecture

```text
Local sample folder
        │
        ▼
File discovery + metadata ──► incremental cache
        │                            │
        ▼                            ▼
Decode / mono / resample       audio embeddings (CLAP)
        │                            │
        └──────────────┬─────────────┘
                       ▼
                cosine similarity
                       ▲
                       │
             text embedding (CLAP)
                       ▲
                       │
                 search prompt
```

Code layout:

- `app.py` — Streamlit UI, filters, audio and waveform previews.
- `soundscout/audio.py` — decoding, resampling, segment selection and waveform generation.
- `soundscout/model.py` — CLAP audio/text embedding wrapper.
- `soundscout/index.py` — incremental indexing and persistent local cache.
- `soundscout/search.py` — cosine ranking and metadata filtering.
- `tests/` — basic tests for segmentation, filtering and ranking.

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The test suite covers audio segmentation, metadata filters, cosine ranking, incremental cache reuse, error tolerance, and evaluation metrics.

## Retrieval Quality Benchmark (Milestone P1)

SoundScout includes an automated benchmark suite to evaluate text-to-audio retrieval quality using information retrieval metrics (Recall@K, Precision@K, MRR).

### 1. Generate the 36-sound evaluation library

```bash
python scripts/generate_evaluation_dataset.py
```

This synthesizes 36 parameterized audio files across 6 professional categories (`Impacts`, `Ambiences`, `Foley`, `Tones`, `Mechanical`, `Transitions`) and generates `evaluation/manifest.json` and 20 ground-truth queries in `evaluation/queries.json`.

### 2. Run the benchmark

```bash
python scripts/benchmark_retrieval.py
```

### Measured Baseline Results (Windows 11, Python 3.11, CPU execution)

| Metric | Result | Meaning |
|---|---|---|
| **Recall@5** | **85.00%** | Relevant audio is retrieved within top 5 candidates for 85% of queries |
| **Recall@3** | **77.50%** | Relevant audio retrieved within top 3 candidates |
| **Recall@1** | **40.00%** | Exact target match at rank #1 |
| **MRR (Mean Reciprocal Rank)** | **0.6184** | Average reciprocal rank of first relevant item |
| **Query Latency (Median)** | **78.4 ms** | End-to-end text encoding + NumPy matrix cosine ranking |
| **Query Latency (p95)** | **107.2 ms** | 95th percentile latency |
| **Indexing Throughput (CPU)** | **2.94 files/s** | Initial extraction pass (12.2s for 36 files) |
| **Cache Re-index Latency** | **125.9 ms** | Subsequent indexing pass with 100% vector reuse |

See [`evaluation/benchmark_report.md`](evaluation/benchmark_report.md) for detailed error analysis, breakdown by category, and failure cases.

## Generate a test library (optional smoke test)

You can create five short synthetic files to check the indexing and preview workflow without downloading audio assets:

```bash
python scripts/generate_test_library.py ./demo_audio
```

Then index `./demo_audio` in the app. These synthetic signals are for a smoke test only; they are not representative of real sound effects and should not be used to claim semantic-search quality.

## Known limitations / next steps

This is a deliberately small MVP, not a production-ready asset manager.

1. **Evaluation set:** create a small labelled set of 50–100 sounds and 10–20 queries; measure Recall@5 and inspect failure cases.
2. **Better long-file search:** store embeddings per segment and return the matching time range, not only one pooled vector per file.
3. **Music-specific metadata:** add BPM / beat-grid estimation for loops, key estimation and onset density, each with confidence and explicit limits.
4. **Better cataloguing:** editable tags, favourites, duplicate detection and exportable CSV.
5. **Scale:** use a vector index such as FAISS for very large libraries; current NumPy cosine search is simple and appropriate for an MVP.
6. **Packaging:** add a desktop wrapper or a local API if users prefer not to open Streamlit.
7. **Product validation:** test with sound designers and ask which repeated task is painful enough that they would pay to save time.

## Model and licensing note

The selected Hugging Face checkpoint, [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused), currently lists an Apache-2.0 model license. This repository's source code is MIT-licensed. Those licences do **not** automatically resolve the rights for every dataset, sample library, or asset you index. For any commercial distribution, review the current model card, checkpoint terms, training-data provenance where relevant, and the licences of the audio assets you use. Do not redistribute someone else's sample pack without permission.

## Suggested portfolio demo

Record a 60–90 second walkthrough: (1) show a messy folder with descriptive and non-descriptive filenames, (2) index it, (3) search using three distinct natural-language prompts, (4) audition the top results, and (5) show how filters narrow results. Include your hardware, index size, indexing time, search latency and a small Recall@5 evaluation in the README after measuring them on your own machine.

---

## Handoff for Antigravity

This repository includes project handoff documents for continuing development with an AI coding agent:

- `ANTIGRAVITY_START_HERE.md` — where to begin and current verified status.
- `PROJECT_HANDOFF.md` — product goal, target user, scope and success criteria.
- `ARCHITECTURE.md` — current technical architecture.
- `ROADMAP.md` — prioritized next milestones.
- `VALIDATION_PLAN.md` — end-to-end and retrieval evaluation plan.
- `AGENTS.md` — coding-agent rules.
- `ANTIGRAVITY_TASK.md` — ready-to-paste initial task.
- `CHANGELOG_HANDOFF.md` — initial project handoff record.
