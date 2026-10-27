from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Sequence

EXPORT_FIELDS = [
    "rank",
    "query",
    "name",
    "similarity",
    "match_start_seconds",
    "match_end_seconds",
    "match_start_timecode",
    "match_end_timecode",
    "duration_seconds",
    "sample_rate",
    "channels",
    "category",
    "format",
    "relative_path",
    "path",
]


def format_timecode(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm broadcast timecode standard."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def results_to_csv(results: Sequence[dict[str, Any]]) -> str:
    """Convert search or shortlist results to clean CSV string for DAW / cue sheet logging."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()

    for idx, item in enumerate(results, start=1):
        match_s = float(item.get("match_start_seconds", 0.0))
        match_e = float(item.get("match_end_seconds", item.get("duration_seconds", 0.0)))
        row = {
            "rank": item.get("rank", idx),
            "query": item.get("query", ""),
            "name": item.get("name", Path(str(item.get("path", ""))).name),
            "similarity": f"{float(item.get('similarity', 0.0)):.4f}",
            "match_start_seconds": f"{match_s:.3f}",
            "match_end_seconds": f"{match_e:.3f}",
            "match_start_timecode": format_timecode(match_s),
            "match_end_timecode": format_timecode(match_e),
            "duration_seconds": f"{float(item.get('duration_seconds', 0.0)):.3f}",
            "sample_rate": int(item.get("sample_rate", 0)),
            "channels": int(item.get("channels", 0)),
            "category": item.get("category", "Uncategorized"),
            "format": item.get("extension", item.get("file_format", "")),
            "relative_path": item.get("relative_path", ""),
            "path": item.get("path", ""),
        }
        writer.writerow(row)

    return output.getvalue()
