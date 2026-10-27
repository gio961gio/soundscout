import csv
import io
import pytest

from soundscout.export import format_timecode, results_to_csv
from soundscout.search import filter_results


def test_format_timecode_values():
    assert format_timecode(0.0) == "00:00:00.000"
    assert format_timecode(1.5) == "00:00:01.500"
    assert format_timecode(65.123) == "00:01:05.123"
    assert format_timecode(3661.050) == "01:01:01.050"
    assert format_timecode(-5.0) == "00:00:00.000"


def test_results_to_csv_output():
    results = [
        {
            "rank": 1,
            "name": "metal_impact.wav",
            "similarity": 0.4521,
            "match_start_seconds": 10.5,
            "match_end_seconds": 20.5,
            "duration_seconds": 25.0,
            "sample_rate": 48000,
            "channels": 2,
            "category": "Impacts",
            "extension": "wav",
            "relative_path": "Impacts/metal_impact.wav",
            "path": "/audio/Impacts/metal_impact.wav",
        }
    ]

    csv_text = results_to_csv(results)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "metal_impact.wav"
    assert r["similarity"] == "0.4521"
    assert r["match_start_timecode"] == "00:00:10.500"
    assert r["match_end_timecode"] == "00:00:20.500"
    assert r["sample_rate"] == "48000"
    assert r["category"] == "Impacts"


def test_results_to_csv_empty():
    csv_text = results_to_csv([])
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 0
    assert "name" in reader.fieldnames
    assert "match_start_timecode" in reader.fieldnames


def test_filter_results_with_keyword():
    items = [
        {"name": "laser_gun.wav", "relative_path": "Weapons/laser_gun.wav", "duration_seconds": 2.0},
        {"name": "heavy_anvil.wav", "relative_path": "Impacts/heavy_anvil.wav", "duration_seconds": 3.0},
        {"name": "gun_reload.wav", "relative_path": "Foley/gun_reload.wav", "duration_seconds": 1.5},
    ]

    # Keyword match in filename
    res_gun = filter_results(items, keyword="gun")
    assert len(res_gun) == 2
    assert [x["name"] for x in res_gun] == ["laser_gun.wav", "gun_reload.wav"]

    # Keyword match in relative folder path
    res_impacts = filter_results(items, keyword="impacts")
    assert len(res_impacts) == 1
    assert res_impacts[0]["name"] == "heavy_anvil.wav"

    # No match
    assert len(filter_results(items, keyword="unmatched_xyz")) == 0
