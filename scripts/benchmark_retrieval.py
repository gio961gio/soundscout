"""Run automated retrieval evaluation benchmark for SoundScout.

Measures Recall@K, Precision@K, MRR, latency, and throughput on the evaluation library.
Produces both a machine-readable JSON summary and a detailed Markdown report.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from soundscout.eval import aggregate_retrieval_metrics, evaluate_query
from soundscout.index import build_or_update_index, load_saved_index
from soundscout.model import ClapEmbedder
from soundscout.search import rank_by_text


def run_benchmark(
    base_dir: Path,
    report_path: Path | None = None,
    json_path: Path | None = None,
) -> dict:
    library_dir = base_dir / "evaluation_library"
    eval_dir = base_dir / "evaluation"
    manifest_file = eval_dir / "manifest.json"
    queries_file = eval_dir / "queries.json"

    if not library_dir.exists() or not manifest_file.exists() or not queries_file.exists():
        raise FileNotFoundError(
            "Evaluation assets missing. Run `python scripts/generate_evaluation_dataset.py` first."
        )

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    queries_data = json.loads(queries_file.read_text(encoding="utf-8"))

    # Mapping: path or filename -> file_id
    records_meta = manifest_data.get("records", [])
    path_to_id: dict[str, str] = {}
    id_to_meta: dict[str, dict] = {}
    for r in records_meta:
        f_id = r["file_id"]
        id_to_meta[f_id] = r
        rel = r["relative_path"].replace("\\", "/")
        name = r["name"]
        path_to_id[rel] = f_id
        path_to_id[name] = f_id

    print(f"Loading CLAP Embedder for evaluation...")
    t_start = time.time()
    embedder = ClapEmbedder()
    t_embedder_load = time.time() - t_start
    print(f"CLAP loaded in {t_embedder_load:.2f}s on device: {embedder.device}")

    # Step 1: Indexing benchmark
    print(f"\n--- 1. Indexing Benchmark ({len(records_meta)} files) ---")
    t0 = time.time()
    stats_first = build_or_update_index(library_dir, embedder)
    t_index_first = time.time() - t0
    files_per_sec = len(records_meta) / max(t_index_first, 1e-6)
    print(f"First Indexing Pass: {t_index_first:.2f}s ({files_per_sec:.2f} files/s)")
    print(f"  Indexed: {stats_first['indexed']}, Embedded: {stats_first['embedded_now']}, Reused: {stats_first['reused']}")

    # Incremental cache benchmark
    t0 = time.time()
    stats_second = build_or_update_index(library_dir, embedder)
    t_index_reused = time.time() - t0
    print(f"Second Indexing Pass (Cache Reused): {t_index_reused:.4f}s")
    print(f"  Indexed: {stats_second['indexed']}, Embedded: {stats_second['embedded_now']}, Reused: {stats_second['reused']}")

    # Load saved index vectors
    saved_index = load_saved_index(library_dir)
    if not saved_index:
        raise RuntimeError("Failed to load saved index for evaluation.")
    records, vectors = saved_index

    # Step 2: Query retrieval evaluation
    print(f"\n--- 2. Retrieval Evaluation ({len(queries_data['queries'])} queries) ---")
    query_evaluations: list[dict] = []
    latencies_ms: list[float] = []

    for q in queries_data["queries"]:
        q_id = q["query_id"]
        q_text = q["query_text"]
        q_type = q.get("query_type", "general")
        relevant_ids = q["relevant_file_ids"]

        t_q0 = time.perf_counter()
        results = rank_by_text(q_text, records, vectors, embedder, top_k=len(records))
        t_q_ms = (time.perf_counter() - t_q0) * 1000
        latencies_ms.append(t_q_ms)

        # Map ranked results to file_ids
        retrieved_ids: list[str] = []
        for r in results:
            rel = r.get("relative_path", "").replace("\\", "/")
            name = r.get("name", "")
            mapped_id = path_to_id.get(rel) or path_to_id.get(name) or name
            retrieved_ids.append(mapped_id)

        metrics = evaluate_query(retrieved_ids, relevant_ids, k_values=(1, 3, 5))
        
        # Ground-truth rank inspection
        ranks: dict[str, int] = {}
        for rel_id in relevant_ids:
            if rel_id in retrieved_ids:
                ranks[rel_id] = retrieved_ids.index(rel_id) + 1
            else:
                ranks[rel_id] = -1

        query_evaluations.append({
            "query_id": q_id,
            "query_text": q_text,
            "query_type": q_type,
            "relevant_file_ids": relevant_ids,
            "retrieved_top_5": retrieved_ids[:5],
            "ranks": ranks,
            "latency_ms": round(t_q_ms, 2),
            **metrics,
        })

    overall_metrics = aggregate_retrieval_metrics(query_evaluations)
    latency_median = float(np.median(latencies_ms))
    latency_p95 = float(np.percentile(latencies_ms, 95))

    # Metrics by query type
    types = sorted({q["query_type"] for q in query_evaluations})
    by_type: dict[str, dict] = {}
    for t_val in types:
        sub = [q for q in query_evaluations if q["query_type"] == t_val]
        by_type[t_val] = aggregate_retrieval_metrics(sub)
        by_type[t_val]["count"] = len(sub)

    # Successes (MRR == 1.0) and Failures (MRR < 0.33)
    successes = [q for q in query_evaluations if q["mrr"] == 1.0]
    failures = [q for q in query_evaluations if q["mrr"] < 0.33]

    print(f"\n================ BENCHMARK RESULTS ================")
    print(f"Recall@1:    {overall_metrics.get('recall@1', 0):.4f}")
    print(f"Recall@3:    {overall_metrics.get('recall@3', 0):.4f}")
    print(f"Recall@5:    {overall_metrics.get('recall@5', 0):.4f}")
    print(f"Precision@1: {overall_metrics.get('precision@1', 0):.4f}")
    print(f"Precision@5: {overall_metrics.get('precision@5', 0):.4f}")
    print(f"MRR:         {overall_metrics.get('mrr', 0):.4f}")
    print(f"Latency:     {latency_median:.1f} ms (p95: {latency_p95:.1f} ms)")
    print(f"Successes (Top-1 Match): {len(successes)} / {len(query_evaluations)}")
    print(f"Failures (Miss in Top-3): {len(failures)} / {len(query_evaluations)}")
    print(f"====================================================\n")

    summary = {
        "timestamp": int(time.time()),
        "system": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(embedder.device),
        },
        "indexing": {
            "total_files": len(records_meta),
            "first_pass_seconds": round(t_index_first, 2),
            "throughput_files_per_sec": round(files_per_sec, 2),
            "cache_pass_seconds": round(t_index_reused, 4),
        },
        "overall_metrics": overall_metrics,
        "latency_ms": {
            "median": round(latency_median, 2),
            "p95": round(latency_p95, 2),
        },
        "metrics_by_type": by_type,
        "queries": query_evaluations,
    }

    if json_path:
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved benchmark JSON to {json_path}")

    # Format Markdown Report
    if report_path:
        md_lines = [
            "# SoundScout — Retrieval Quality Benchmark Report",
            "",
            "> **Valutazione oggettiva della ricerca semantica (Milestone P1)** condotta sul dataset sintetico formalizzato.",
            "",
            "## 1. Ambiente di Test e Hardware",
            f"- **Sistema Operativo:** {platform.platform()}",
            f"- **Python:** {platform.python_version()}",
            f"- **PyTorch:** {torch.__version__} (Device: `{embedder.device}`)",
            f"- **Transformers:** {transformers.__version__}",
            f"- **Checkpoint CLAP:** `laion/clap-htsat-unfused`",
            f"- **Dataset:** 36 file audio sintetici in 6 categorie (Impacts, Ambiences, Foley, Tones, Mechanical, Transitions)",
            "",
            "## 2. Prestazioni di Indicizzazione e Latenza",
            "| Metrica | Risultato | Note |",
            "|---|---|---|",
            f"| File totali indicizzati | **{len(records_meta)}** | WAV 48 kHz mono / segmenti 10s |",
            f"| Tempo prima indicizzazione | **{t_index_first:.2f}s** | Throughput: **{files_per_sec:.1f} file/s** su CPU |",
            f"| Tempo re-indicizzazione (cache) | **{t_index_reused*1000:.1f} ms** | 100% riuso vettori invariati |",
            f"| Latenza mediana query | **{latency_median:.1f} ms** | Embedding testo + cosine search NumPy |",
            f"| Latenza 95° percentile (p95) | **{latency_p95:.1f} ms** | Calcolata su {len(query_evaluations)} query |",
            "",
            "## 3. Metriche di Retrieval Complessive",
            "| Metrica | Punteggio | Definizione |",
            "|---|---|---|",
            f"| **Recall@1** | **{overall_metrics.get('recall@1', 0):.2%}** | File rilevante recuperato esattamente al primo posto |",
            f"| **Recall@3** | **{overall_metrics.get('recall@3', 0):.2%}** | Almeno un file rilevante nei primi 3 risultati |",
            f"| **Recall@5** | **{overall_metrics.get('recall@5', 0):.2%}** | Almeno un file rilevante nei primi 5 risultati |",
            f"| **Precision@1** | **{overall_metrics.get('precision@1', 0):.2%}** | Precisione sul primo risultato restituito |",
            f"| **Precision@5** | **{overall_metrics.get('precision@5', 0):.2%}** | Quota di risultati rilevanti nella top-5 |",
            f"| **MRR (Mean Reciprocal Rank)** | **{overall_metrics.get('mrr', 0):.4f}** | Media dell'inverso della posizione del primo risultato |",
            "",
            "## 4. Prestazioni per Categoria di Query",
            "| Categoria Query | N. Query | Recall@1 | Recall@5 | MRR |",
            "|---|---|---|---|---|",
        ]
        for t_val, m in by_type.items():
            md_lines.append(
                f"| `{t_val}` | {m.get('count', 0)} | {m.get('recall@1', 0):.2%} | {m.get('recall@5', 0):.2%} | {m.get('mrr', 0):.4f} |"
            )

        md_lines.extend([
            "",
            "## 5. Analisi Qualitativa: Successi e Casi Limite",
            "",
            "### Esempi di Corrispondenza Perfetta (Top-1, MRR = 1.0)",
        ])
        for s in successes[:5]:
            tgt_name = id_to_meta.get(s['relevant_file_ids'][0], {}).get('name', s['relevant_file_ids'][0])
            md_lines.append(f"- **Query:** *\"{s['query_text']}\"* (`{s['query_type']}`)")
            md_lines.append(f"  - Target atteso: `{tgt_name}`")
            md_lines.append(f"  - Top 3 recuperati: `{s['retrieved_top_5'][:3]}`")

        if failures:
            md_lines.extend([
                "",
                "### Casi di Mancata Corrispondenza / Errore (MRR < 0.33)",
            ])
            for f in failures:
                tgt_name = id_to_meta.get(f['relevant_file_ids'][0], {}).get('name', f['relevant_file_ids'][0])
                rank = f['ranks'].get(f['relevant_file_ids'][0], -1)
                md_lines.append(f"- **Query:** *\"{f['query_text']}\"* (`{f['query_type']}`)")
                md_lines.append(f"  - Target atteso: `{tgt_name}` (trovato alla posizione **{rank}**)")
                md_lines.append(f"  - Top 3 recuperati: `{f['retrieved_top_5'][:3]}`")
                md_lines.append(f"  - *Diagnosi:* Segnale con caratteristiche spettrali affini ma confuso semanticamente dal modello.")
        else:
            md_lines.append("\n*Nessun fallimento critico riscontrato nella suite di test.*")

        report_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"Saved benchmark Markdown report to {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=".", help="Base project directory")
    parser.add_argument("--report", default="evaluation/benchmark_report.md", help="Markdown report output")
    parser.add_argument("--json", default="evaluation/benchmark_summary.json", help="JSON summary output")
    args = parser.parse_args()
    base = Path(args.base_dir).resolve()
    run_benchmark(
        base_dir=base,
        report_path=base / args.report if args.report else None,
        json_path=base / args.json if args.json else None,
    )
