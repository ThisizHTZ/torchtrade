"""Tests for the 19 free Torch-Trade dataset catalog."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from examples.end_to_end.local_backtest.torchtrade_datasets import (
    EXPECTED_DATASET_COUNT,
    FREE_DATASETS,
    validate_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "end_to_end" / "local_backtest" / "torchtrade_datasets.py"


def test_validate_catalog_checks_all_19_free_datasets() -> None:
    """Validate the full free dataset catalog in-process."""
    ids = validate_catalog()

    assert len(ids) == EXPECTED_DATASET_COUNT == 19
    assert len(set(ids)) == EXPECTED_DATASET_COUNT


@pytest.mark.parametrize("dataset", FREE_DATASETS, ids=[dataset.dataset_id for dataset in FREE_DATASETS])
def test_each_free_dataset_has_research_metadata(dataset) -> None:
    """Every dataset should have enough metadata to be cited in an experiment manifest."""
    assert dataset.dataset_id.startswith("Torch-Trade/")
    assert dataset.asset.endswith("USDT")
    assert dataset.frequency in {"1m", "5m", "8h"}
    assert dataset.start <= dataset.end
    assert "timestamp" in dataset.expected_columns
    assert dataset.rows_label
    assert dataset.description.endswith(".")


def test_catalog_cli_validates_and_exports_all_datasets(tmp_path: Path) -> None:
    """The CLI should test all 19 entries and export a CSV manifest."""
    validate_run = subprocess.run(
        [sys.executable, str(SCRIPT), "validate-catalog"],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert "19/19 OK" in validate_run.stdout

    output = tmp_path / "torchtrade_free_datasets.csv"
    subprocess.run(
        [sys.executable, str(SCRIPT), "catalog", "--output", str(output)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == EXPECTED_DATASET_COUNT
    assert rows[0]["dataset_id"] == FREE_DATASETS[0].dataset_id
