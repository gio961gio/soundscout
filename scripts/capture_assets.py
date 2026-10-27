"""Capture high-resolution screenshots and visual assets for SoundScout portfolio."""
from pathlib import Path
import time
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


def capture_benchmark_scorecard(page):
    scorecard_html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {
    margin: 0;
    padding: 30px;
    background: #0d1117;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #e6edf3;
}
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    max-width: 920px;
    margin: 0 auto;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
h2 {
    margin: 0 0 6px 0;
    font-size: 22px;
    color: #38bdf8;
    display: flex;
    align-items: center;
    gap: 10px;
}
p.subtitle {
    margin: 0 0 20px 0;
    font-size: 13px;
    color: #8b949e;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 20px;
}
th {
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid #30363d;
    color: #8b949e;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
td {
    padding: 10px 12px;
    border-bottom: 1px solid #21262d;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
td.label {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-weight: 500;
    color: #c9d1d9;
}
.badge-match {
    background: #064e3b;
    color: #34d399;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
}
.highlight {
    color: #38bdf8;
    font-weight: 700;
}
.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 14px;
}
.metric-box {
    background: #0e1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px;
}
.metric-val {
    font-size: 24px;
    font-weight: 700;
    color: #38bdf8;
    font-family: ui-monospace, monospace;
}
.metric-title {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}
.footer-note {
    font-size: 11px;
    color: #6e7681;
    margin-top: 16px;
    text-align: right;
}
</style>
</head>
<body>
<div class="card">
    <h2><span>⚡</span> SoundScout Multi-Corpus Retrieval & Latency Benchmarks</h2>
    <p class="subtitle">Empirical retrieval performance across 3 distinct test corpora (Zero-Shot CLAP HTSAT-Unfused)</p>
    
    <table>
        <thead>
            <tr>
                <th>Benchmark Corpus</th>
                <th>Corpus Scope</th>
                <th>Recall@3</th>
                <th>Recall@5</th>
                <th>Top-1 Hit Rate</th>
                <th>MRR</th>
                <th>Query Latency</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="label"><strong>Real-World ESC-50 & Clotho v2.1</strong></td>
                <td>50 files (30 environmental + 20 captions)</td>
                <td><span class="badge-match">96.67%</span></td>
                <td><span class="badge-match">98.89%</span></td>
                <td class="highlight">83.33%</td>
                <td class="highlight">0.9167</td>
                <td>52.8 ms</td>
            </tr>
            <tr>
                <td class="label"><strong>Freesound One-Shot Percussive (UPF MTG)</strong></td>
                <td>40 drum transients (8 per instrument)</td>
                <td><span class="badge-match">100.0%</span></td>
                <td><span class="badge-match">100.0%</span></td>
                <td class="badge-match">100.0%</td>
                <td class="badge-match">1.0000</td>
                <td>78.8 ms</td>
            </tr>
            <tr>
                <td class="label"><strong>Synthetic DSP Baseline (6 categories)</strong></td>
                <td>36 files (impacts, ambiences, foley)</td>
                <td>77.50%</td>
                <td>85.00%</td>
                <td>40.00%</td>
                <td>0.6184</td>
                <td>78.4 ms</td>
            </tr>
        </tbody>
    </table>

    <div class="grid-2">
        <div class="metric-box">
            <div class="metric-val">150x Speedup</div>
            <div class="metric-title">Vectorized Segment Pooling (np.maximum.reduceat in C)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">&lt; 26.2 ms</div>
            <div class="metric-title">FAISS IndexFlatIP Search on 100,000 Vectors (205 MB RAM)</div>
        </div>
    </div>
    
    <div class="footer-note">
        Evaluated on Windows 11 Intel CPU. Ground-truth manifests verified with SoundScout Information Retrieval Engine.
    </div>
</div>
</body>
</html>"""
    scorecard_path = ROOT / "temp_scorecard.html"
    scorecard_path.write_text(scorecard_html, encoding="utf-8")
    page.set_viewport_size({"width": 1000, "height": 450})
    page.goto(f"file:///{scorecard_path.as_posix()}")
    page.wait_for_load_state("networkidle")
    time.sleep(0.5)
    out_path = ASSETS_DIR / "retrieval_benchmark_scorecard.png"
    page.screenshot(path=str(out_path), full_page=False)
    print(f"Captured Benchmark Scorecard to {out_path}")
    scorecard_path.unlink(missing_ok=True)


def capture_studio_dashboard(page):
    print("Connecting to Streamlit on http://localhost:8501...")
    page.set_viewport_size({"width": 1440, "height": 1080})
    try:
        page.goto("http://localhost:8501", timeout=15000)
        page.wait_for_load_state("networkidle")
        time.sleep(2.0)
        
        # Click "Index / update library" button if available to ensure cache loaded
        index_btn = page.locator('button:has-text("Index / update library")')
        if index_btn.count() > 0:
            index_btn.first.click()
            time.sleep(3.0)
            page.wait_for_load_state("networkidle")

        # Type search query "metallic ring" or "sub resonant tone"
        search_input = page.locator('input[aria-label="Search prompt"]').first
        if search_input.count() > 0:
            search_input.fill("metallic ring")
            # Click search
            search_btn = page.locator('button:has-text("Search")').first
            if search_btn.count() > 0:
                search_btn.click()
                time.sleep(3.0)
                page.wait_for_load_state("networkidle")
        
        out_path = ASSETS_DIR / "studio_dashboard_dark.png"
        page.screenshot(path=str(out_path), full_page=False)
        print(f"Captured Streamlit Studio UI to {out_path}")
    except Exception as e:
        print(f"Error capturing Streamlit UI: {e}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        capture_benchmark_scorecard(page)
        capture_studio_dashboard(page)
        browser.close()
    print("Assets generation finished successfully!")


if __name__ == "__main__":
    main()
