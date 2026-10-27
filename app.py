from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from soundscout.audio import SUPPORTED_EXTENSIONS, waveform_minmax
from soundscout.export import results_to_csv
from soundscout.index import build_or_update_index, load_saved_index
from soundscout.model import ClapEmbedder
from soundscout.search import filter_results, rank_by_audio_example, rank_by_text

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


def show_waveform(
    path: str,
    key: str,
    match_start: float | None = None,
    match_end: float | None = None,
) -> None:
    waveform = waveform_minmax(path)
    if waveform is None:
        st.caption("Waveform preview unavailable for this format.")
        return
    x, bounds = waveform
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bounds[:, 1], mode="lines", line={"width": 1, "color": "#4a90e2"}, name="Peak", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bounds[:, 0], mode="lines", line={"width": 1, "color": "#4a90e2"}, fill="tonexty", fillcolor="rgba(74, 144, 226, 0.22)", name="Trough", hoverinfo="skip"))

    if match_start is not None and match_end is not None and match_end > match_start:
        fig.add_vrect(
            x0=match_start,
            x1=match_end,
            fillcolor="rgba(255, 152, 0, 0.30)",
            line_width=1.5,
            line_color="#ff9800",
            annotation_text="Match",
            annotation_position="top left",
            annotation_font={"size": 9, "color": "#ff9800"},
        )

    fig.update_layout(
        height=110, margin={"l": 0, "r": 0, "t": 8, "b": 20},
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Seconds", yaxis_title="Amplitude", font={"size": 10},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=True, zerolinecolor="rgba(128,128,128,.3)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"wave-{key}")


with st.sidebar:
    st.markdown("# 🎧 SoundScout")
    st.caption("Search your local sound library with natural language.")
    default_dir = Path("demo_audio").resolve() if Path("demo_audio").exists() else Path.home() / "Music" / "Samples"
    library_path = st.text_input(
        "Library folder",
        value=str(default_dir),
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
    keyword_filter = st.text_input("Filter by filename/keyword", value="", placeholder="e.g. reverb, metallic, 01")
    selected_sample_rates = st.multiselect("Sample rate", options=sample_rates_all, default=[])
    selected_extensions = st.multiselect("Format", options=extensions_all, default=[])
    selected_channels = st.multiselect("Channels", options=channels_all, default=[])
    selected_categories = st.multiselect("Top-level folder", options=categories_all, default=[])

shortlist: dict[str, dict] = st.session_state.setdefault("shortlist", {})

col_a, col_b, col_c, col_d = st.columns([2.5, 1, 1, 1])
with col_a:
    st.metric("Indexed audio files", len(records))
with col_b:
    st.metric("Formats", len(extensions_all))
with col_c:
    st.metric("Folders / categories", len(categories_all))
with col_d:
    st.metric("Shortlisted sounds", len(shortlist))

tab_search, tab_shortlist = st.tabs(["🔍 Search & Browse", f"⭐ Session Shortlist ({len(shortlist)})"])

with tab_search:
    col_input, col_btn = st.columns([4.2, 1], vertical_alignment="bottom")
    with col_input:
        query_input = st.text_input(
            "Describe the sound you want",
            key="search_query_input",
            placeholder="e.g. distant metallic impact with long reverb",
            help="CLAP works best with concise descriptive prompts. English queries are a useful starting point.",
        )
    with col_btn:
        search_clicked = st.button("Search sounds", type="primary", use_container_width=True)

    with st.expander("🎯 Or pick a sound from your library to find matches immediately", expanded=False):
        pick_col1, pick_col2 = st.columns([4, 1.2], vertical_alignment="bottom")
        with pick_col1:
            sound_names = [f"{r.get('name', 'audio')} [{r.get('category', 'Uncategorized')}] — {r.get('relative_path', '')}" for r in records]
            chosen_idx = st.selectbox(
                "Select reference audio:",
                range(len(records)),
                format_func=lambda i: sound_names[i],
                key="dropdown_pick_sim",
                label_visibility="collapsed",
            )
        with pick_col2:
            if st.button("Find Similar", key="btn_dropdown_sim", type="secondary", use_container_width=True):
                st.session_state["find_similar_source"] = records[chosen_idx]
                st.rerun()

    active_query = query_input.strip()
    sim_source = st.session_state.get("find_similar_source")

    # If the user clicks search or types a new query, exit reverse audio search mode
    if search_clicked and active_query:
        st.session_state.pop("find_similar_source", None)
        sim_source = None

    if sim_source:
        b_col, c_col = st.columns([5, 1])
        with b_col:
            st.info(f"🎯 **Reverse Audio Search:** Finding sounds timbrally similar to **{sim_source.get('name', 'selected audio')}**")
        with c_col:
            if st.button("✖ Clear", key="clear_find_similar", use_container_width=True):
                del st.session_state["find_similar_source"]
                st.rerun()

    is_browse_mode = not active_query and not sim_source

    try:
        if sim_source:
            with st.spinner("Comparing acoustic timbre across library…"):
                ranked = rank_by_audio_example(sim_source, records, vectors, top_k=len(records), exclude_self=True)
            active_label = f"similar to *{sim_source.get('name', 'audio')}*"
            export_query_name = f"Similar to: {sim_source.get('name', '')}"
        elif active_query:
            with st.spinner("Comparing your query with the indexed sounds…"):
                embedder = get_embedder()
                ranked = rank_by_text(active_query, records, vectors, embedder, top_k=len(records))
            active_label = f"*{active_query}*"
            export_query_name = active_query
        else:
            # Browse library mode: show all records matching sidebar filters
            ranked = list(records)
            active_label = "Library Collection"
            export_query_name = "Library Browse"

        filtered = filter_results(
            ranked,
            min_duration=float(duration_range[0]),
            max_duration=float(duration_range[1]),
            sample_rates=selected_sample_rates,
            extensions=selected_extensions,
            channels=selected_channels,
            categories=selected_categories,
            keyword=keyword_filter,
        )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        st.stop()

    res_header_col, export_col = st.columns([3.5, 1.5])
    with res_header_col:
        if is_browse_mode:
            st.markdown(f"### 📂 Library Collection ({len(filtered)} sounds)")
            st.caption("Browse and audition your files, filter by category/duration, or click **'🎯 Similar'** on any sound to find acoustic matches.")
        else:
            st.markdown(f"### Results for {active_label}")
            st.caption(f"{len(filtered)} result(s) match the current filters. Similarity is a ranking score, not a probability.")
    with export_col:
        if filtered:
            results_with_query = [dict(item, query=export_query_name) for item in filtered]
            st.download_button(
                "📥 Export Results (CSV)",
                data=results_to_csv(results_with_query).encode("utf-8-sig"),
                file_name="soundscout_search_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if not filtered:
        st.warning("No files match these filters. Widen the duration or metadata filters and try again.")
    else:
        for rank, item in enumerate(filtered[:30], start=1):
            path = str(item["path"])
            title_col, score_col = st.columns([5, 1])
            with title_col:
                t_col, sim_btn_col, fav_col = st.columns([3.6, 1.2, 1.2])
                with t_col:
                    st.markdown(f"#### {rank}. {item.get('name', Path(path).name)}")
                    st.caption(item.get("relative_path", path))
                with sim_btn_col:
                    if st.button("🎯 Similar", key=f"sim-search-{rank}-{abs(hash(path))}", help="Find timbrally similar sounds in library"):
                        st.session_state["find_similar_source"] = item
                        st.rerun()
                with fav_col:
                    is_fav = path in shortlist
                    if is_fav:
                        if st.button("⭐ In Shortlist", key=f"fav-search-{rank}-{abs(hash(path))}", help="Remove from session shortlist"):
                            del shortlist[path]
                            st.rerun()
                    else:
                        if st.button("☆ Add to Shortlist", key=f"fav-search-{rank}-{abs(hash(path))}", help="Add to session shortlist"):
                            item_to_save = dict(item)
                            item_to_save["query"] = export_query_name
                            shortlist[path] = item_to_save
                            st.rerun()
            with score_col:
                if is_browse_mode:
                    st.caption(f"Format: **{item.get('extension', 'audio').upper()}**")
                else:
                    st.metric("Cosine score", f"{item.get('similarity', 0.0):.3f}")

            match_s = float(item.get("match_start_seconds", 0.0))
            match_e = float(item.get("match_end_seconds", item.get("duration_seconds", 0.0)))
            total_segs = int(item.get("total_segments", 1))
            seg_idx = int(item.get("match_segment_index", 0))

            details = st.columns(4)
            details[0].caption(f"Duration: **{format_duration(item.get('duration_seconds', 0))}**")
            if not is_browse_mode and total_segs > 1:
                details[1].caption(f"Best match: **{format_duration(match_s)} – {format_duration(match_e)}**")
            else:
                details[1].caption(f"Category: **{item.get('category', 'Uncategorized')}**")
            details[2].caption(f"Sample rate: **{item.get('sample_rate', 0):,} Hz**")
            details[3].caption(f"Segments: **{total_segs}**")

            left, right = st.columns([1, 1.6])
            with left:
                try:
                    start_offset = int(match_s) if (not is_browse_mode and total_segs > 1) else 0
                    st.audio(path, format=mime_for(path), start_time=start_offset)
                    if not is_browse_mode and total_segs > 1:
                        st.caption(f"🎯 Salient event starts at **{format_duration(match_s)}**")
                except Exception:
                    st.caption("Audio preview is not available in this browser for this format.")
            with right:
                if is_browse_mode:
                    show_waveform(path, key=f"{rank}-{abs(hash(path))}")
                else:
                    show_waveform(path, key=f"{rank}-{abs(hash(path))}", match_start=match_s, match_end=match_e)
            st.divider()

        if len(filtered) > 30:
            st.caption("Showing the first 30 sounds. Narrow by category or filters to refine the list.")

with tab_shortlist:
    st.markdown("### ⭐ Session Shortlist")
    st.caption("Collect candidate sounds during your session and export timecoded cue sheets directly for your DAW.")

    if not shortlist:
        st.info("Your shortlist is currently empty. Search for sounds and click **'☆ Add to Shortlist'** to collect candidates here.")
    else:
        top_s_col1, top_s_col2, top_s_col3 = st.columns([3, 1.2, 1])
        with top_s_col1:
            st.write(f"**{len(shortlist)} sound(s)** selected in this session.")
        with top_s_col2:
            st.download_button(
                "📥 Export Shortlist (CSV)",
                data=results_to_csv(list(shortlist.values())).encode("utf-8-sig"),
                file_name="soundscout_shortlist.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with top_s_col3:
            if st.button("🗑️ Clear", use_container_width=True):
                shortlist.clear()
                st.rerun()

        st.divider()

        for s_idx, (s_path, s_item) in enumerate(list(shortlist.items()), start=1):
            s_title_col, s_score_col = st.columns([5, 1])
            with s_title_col:
                s_t_col, s_sim_col = st.columns([4, 1.2])
                with s_t_col:
                    st.markdown(f"#### {s_idx}. {s_item.get('name', Path(s_path).name)}")
                    query_info = f" · Prompt: *\"{s_item['query']}\"*" if s_item.get("query") else ""
                    st.caption(f"{s_item.get('relative_path', s_path)}{query_info}")
                with s_sim_col:
                    if st.button("🎯 Similar", key=f"sim-short-{s_idx}-{abs(hash(s_path))}", help="Find timbrally similar sounds in library"):
                        st.session_state["find_similar_source"] = s_item
                        st.rerun()
            with s_score_col:
                if st.button("❌ Remove", key=f"del-short-{s_idx}-{abs(hash(s_path))}"):
                    del shortlist[s_path]
                    st.rerun()

            s_match_s = float(s_item.get("match_start_seconds", 0.0))
            s_match_e = float(s_item.get("match_end_seconds", s_item.get("duration_seconds", 0.0)))

            s_details = st.columns(4)
            s_details[0].caption(f"Duration: **{format_duration(s_item.get('duration_seconds', 0))}**")
            s_details[1].caption(f"Best match: **{format_duration(s_match_s)} – {format_duration(s_match_e)}**")
            s_details[2].caption(f"Sample rate: **{s_item.get('sample_rate', 0):,} Hz**")
            s_details[3].caption(f"Category: **{s_item.get('category', 'Uncategorized')}**")

            s_left, s_right = st.columns([1, 1.6])
            with s_left:
                try:
                    s_offset = int(s_match_s) if s_item.get("total_segments", 1) > 1 else 0
                    st.audio(s_path, format=mime_for(s_path), start_time=s_offset)
                except Exception:
                    st.caption("Audio preview unavailable.")
            with s_right:
                show_waveform(s_path, key=f"short-{s_idx}-{abs(hash(s_path))}", match_start=s_match_s, match_end=s_match_e)
            st.divider()

with st.expander("About this MVP"):
    st.markdown(
        """
        - Audio is indexed on the machine running SoundScout; files are not uploaded to a remote app by this code.
        - The first indexing run downloads the pretrained CLAP model. Later runs reuse local model and embedding caches.
        - Text/audio similarity is semantic ranking, not a calibrated confidence score. Always audition the results.
        - Search quality depends on the model, sound domain, prompt wording and the contents of the library.
        """
    )
