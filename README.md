# SoundScout

**Natural-language search for local sound-effect and sample libraries.**

SoundScout indexes audio files on your computer and lets you search them with descriptions such as *"distant metallic impact with long reverb"*. It uses CLAP (Contrastive Language-Audio Pretraining) to embed audio and text in a shared representation space, then ranks files by cosine similarity.

> Portfolio MVP for audio / music technology internships. Search scores are ranking signals, not calibrated probabilities; audition the files before choosing them.

## What works in this MVP

- Recursive indexing of a local folder and its subfolders.
- Text-to-audio semantic retrieval using a pretrained CLAP checkpoint.
- Incremental cache: unchanged files reuse their saved embeddings on subsequent indexing runs.
- Audio preview and waveform preview per result.
- Metadata filters for duration, sample rate, channel count, format and top-level folder.
- WAV, FLAC, OGG and AIFF support through `soundfile`; MP3 decoding can use FFmpeg when installed.
- Embeddings stay on the machine running the app. The first run downloads the pretrained model from Hugging Face.

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

## Generate a test library (optional)

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
