"""Tests for dependency-free local result visualizations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "examples" / "end_to_end" / "local_backtest" / "run.py"
VIS_SCRIPT = REPO_ROOT / "examples" / "end_to_end" / "local_backtest" / "visualize.py"


def test_visualize_benchmark_and_dataset_smoke_outputs(tmp_path: Path) -> None:
    """SVG rendering should work with only JSON artifacts and the Python stdlib."""
    output_dir = tmp_path / "local_backtest"
    subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "run-all", "--output-dir", str(output_dir)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    benchmark_svg = output_dir / "benchmark_summary.svg"
    subprocess.run(
        [
            sys.executable,
            str(VIS_SCRIPT),
            "benchmark",
            "--report",
            str(output_dir / "benchmark_report.json"),
            "--output",
            str(benchmark_svg),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    smoke_json = output_dir / "dataset_smoke.json"
    smoke_json.write_text(
        json.dumps(
            [
                {"dataset_id": "Torch-Trade/example_ok", "rows": 2, "columns": ["timestamp", "close"]},
                {"dataset_id": "Torch-Trade/example_fail", "error": "403 Forbidden"},
            ]
        ),
        encoding="utf-8",
    )
    dataset_svg = output_dir / "dataset_smoke.svg"
    subprocess.run(
        [
            sys.executable,
            str(VIS_SCRIPT),
            "datasets",
            "--smoke-results",
            str(smoke_json),
            "--output",
            str(dataset_svg),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "Local Backtest Benchmark" in benchmark_svg.read_text(encoding="utf-8")
    assert "Torch-Trade Dataset Smoke Test" in dataset_svg.read_text(encoding="utf-8")
    assert "example_ok" in dataset_svg.read_text(encoding="utf-8")
