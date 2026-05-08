"""Smoke tests for the dependency-free local backtest example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "end_to_end" / "local_backtest" / "run.py"


def test_local_backtest_run_all_writes_artifacts(tmp_path: Path) -> None:
    """The local project should run end-to-end with only the Python stdlib."""
    output_dir = tmp_path / "local_backtest"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "run-all", "--output-dir", str(output_dir)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "artifacts written" in completed.stdout
    data_path = output_dir / "synthetic_ohlcv.csv"
    policy_path = output_dir / "policy.json"
    evaluation_path = output_dir / "evaluation.json"

    assert data_path.exists()
    assert policy_path.exists()
    assert evaluation_path.exists()

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert policy["policy"]["short_window"] < policy["policy"]["long_window"]
    assert evaluation["evaluation_metrics"]["bars"] > 0
    assert evaluation["evaluation_metrics"]["final_equity"] > 0
