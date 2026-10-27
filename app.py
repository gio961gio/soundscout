from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from soundscout.audio import SUPPORTED_EXTENSIONS, waveform_minmax
from soundscout.index import build_or_update_index, load_saved_index
from soundscout.model import ClapEmbedder
from soundscout.search import filter_results, rank_by_text

st.set_page_config(page_title="SoundScout", page_icon="🎧", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
      .hero {padding: 1.35rem 1.5rem; border: 1px solid rgba(128,128,128,.28); border-radius: 18px; margin-bottom: 1rem;}
      .muted {opacity: .72; font-size: .9rem;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,.06); padding: .8rem 1rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedder() -> ClapEmbedder:
    return ClapEmbedder()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}" if minutes else f"{secs}s"


def mime_for(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".aif": "audio/aiff",
        ".aiff": "audio/aiff",
    }.get(ext, "audio/wav")


def show_waveform(path: str, key: str) -> None:
    waveform = waveform_minmax(path)
    if waveform is None:
        st.caption("Waveform preview unavailable for this format.")
        return
    x, bounds = waveform
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bounds[:, 1], mode="lines", line={"width": 1}, name="Peak", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bounds[:, 0], mode="lines", line={"width": 1}, fill="tonexty", name="Trough", hoverinfo="skip"))
    fig.update_layout(
        height=105, margin={"l": 0, "r": 0, "t": 5, "b": 20},
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Seconds", yaxis_title="Amplitude", font={"size": 10},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=True, zerolinecolor="rgba(128,128,128,.3)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"wave-{key}")


with st.sidebar:
    st.markdown("# 🎧 SoundScout")
    st.caption("Search your local sound library with natural language.")
    library_path = st.text_input(
        "Library folder",
        value=str(Path.home() / "Music" / "Samples"),
        help="Path on the computer running this app. Subfolders are indexed recursively.",
    )
    st.caption("Supported: WAV, FLAC, OGG, AIFF. MP3 may require FFmpeg.")
    index_button = st.button("Index / update library", type="primary", use_container_width=True)
    st.divider()
    st.markdown("#### Filters")

root = Path(library_path).expanduser()
index_data = None
if root.exists() and root.is_dir():
    index_data = load_saved_index(root)

if index_button:
    try:
        with st.spinner("Loading CLAP and indexing audio… the model downloads on first use."):
            embedder_for_index = get_embedder()
            progress = st.progress(0, text="Preparing index…")

            def on_progress(done: int, total: int, current: str) -> None:
                progress.progress(min(1.0, done / max(1, total)), text=f"{done}/{total}: {current[:70]}")

            stats = build_or_update_index(root, embedder_for_index, progress_callback=on_progress)
            progress.empty()
        st.success(
            f"Index ready: {stats['indexed']} files · {stats['embedded_now']} embedded · "
            f"{stats['reused']} reused from cache."
        )
        if stats["errors"]:
            with st.expander(f"{len(stats['errors'])} file(s) skipped — click to inspect"):
                for error in stats["errors"]:
                    st.write(f"**{Path(error['file']).name}** — {error['error']}")
        index_data = load_saved_index(root)
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0">Find the sound, not the filename.</h1>
      <p style="margin:.45rem 0 0 0; opacity:.78">Semantic search for sound effects, recordings and sample libraries — local-first, powered by CLAP.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if index_data is None:
    st.info("Choose a local sample folder in the sidebar, then click **Index / update library** to build your first index.")
    st.markdown("**Try queries like:** `distant metallic impact with long reverb` · `a short dry hand clap` · `low rumbling engine`")
    st.stop()

records, vectors = index_data
if not records:
    st.warning("The index is empty. Check that your folder contains readable audio files, then update the index.")
    st.stop()

max_duration_all = max(float(r.get("duration_seconds", 0)) for r in records)
sample_rates_all = sorted({int(r.get("sample_rate", 0)) for r in records})
extensions_all = sorted({str(r.get("extension", "")) for r in records})
channels_all = sorted({int(r.get("channels", 0)) for r in records})
categories_all = sorted({str(r.get("category", "Uncategorized")) for r in records})

with st.sidebar:
    duration_range = st.slider(
        "Duration (seconds)", min_value=0.0, max_value=max(1.0, float(np.ceil(max_duration_all))),
        value=(0.0, max(1.0, float(np.ceil(max_duration_all)))), step=0.5,
    )
    selected_sample_rates = st.multiselect("Sample rate", options=sample_rates_all, default=[])
    selected_extensions = st.multiselect("Format", options=extensions_all, default=[])
    selected_channels = st.multiselect("Channels", options=channels_all, default=[])
    selected_categories = st.multiselect("Top-level folder", options=categories_all, default=[])

col_a, col_b, col_c = st.columns([3, 1, 1])
with col_a:
    st.metric("Indexed audio files", len(records))
with col_b:
    st.metric("Formats", len(extensions_all))
with col_c:
    st.metric("Folders / categories", len(categories_all))

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Describe the sound you want",
        value=st.session_state.get("search_query", ""),
        placeholder="e.g. distant metallic impact with long reverb",
        help="CLAP works best with concise descriptive prompts. English queries are a useful starting point.",
    )
    search_clicked = st.form_submit_button("Search sounds", type="primary", use_container_width=True)
if search_clicked:
    st.session_state["search_query"] = query.strip()

active_query = st.session_state.get("search_query", "")
if not active_query:
    st.caption("Tip: describe texture, distance, space, material and action — not just a broad category.")
    st.stop()

try:
    with st.spinner("Comparing your query with the indexed sounds…"):
        embedder = get_embedder()
        ranked = rank_by_text(active_query, records, vectors, embedder, top_k=len(records))
    filtered = filter_results(
        ranked,
        min_duration=float(duration_range[0]),
        max_duration=float(duration_range[1]),
        sample_rates=selected_sample_rates,
        extensions=selected_extensions,
        channels=selected_channels,
        categories=selected_categories,
    )
except Exception as exc:
    st.error(f"Search failed: {exc}")
    st.stop()

st.markdown(f"### Results for *{active_query}*")
st.caption(f"{len(filtered)} result(s) match the current filters. Similarity is a ranking score, not a probability.")

if not filtered:
    st.warning("No files match these filters. Widen the duration or metadata filters and try again.")
    st.stop()

for rank, item in enumerate(filtered[:30], start=1):
    path = str(item["path"])
    title_col, score_col = st.columns([5, 1])
    with title_col:
        st.markdown(f"#### {rank}. {item.get('name', Path(path).name)}")
        st.caption(item.get("relative_path", path))
    with score_col:
        st.metric("Cosine score", f"{item['similarity']:.3f}")

    details = st.columns(4)
    details[0].caption(f"Duration: **{format_duration(item.get('duration_seconds', 0))}**")
    details[1].caption(f"Sample rate: **{item.get('sample_rate', 0):,} Hz**")
    details[2].caption(f"Channels: **{item.get('channels', 0)}**")
    details[3].caption(f"Category: **{item.get('category', 'Uncategorized')}**")

    left, right = st.columns([1, 1.6])
    with left:
        try:
            st.audio(path, format=mime_for(path))
        except Exception:
            st.caption("Audio preview is not available in this browser for this format.")
    with right:
        show_waveform(path, key=f"{rank}-{abs(hash(path))}")
    st.divider()

if len(filtered) > 30:
    st.caption("Showing the first 30 results. Narrow the query or filters to refine the list.")

with st.expander("About this MVP"):
    st.markdown(
        """
        - Audio is indexed on the machine running SoundScout; files are not uploaded to a remote app by this code.
        - The first indexing run downloads the pretrained CLAP model. Later runs reuse local model and embedding caches.
        - Text/audio similarity is semantic ranking, not a calibrated confidence score. Always audition the results.
        - Search quality depends on the model, sound domain, prompt wording and the contents of the library.
        """
    )
