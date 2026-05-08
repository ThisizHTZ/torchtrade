"""Tests for the dependency-free Binance public-data downloader."""

from __future__ import annotations

import csv
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "end_to_end" / "local_backtest" / "download_binance.py"


def test_binance_downloader_converts_monthly_zip_to_canonical_csv(tmp_path: Path) -> None:
    """The downloader should work without network when pointed at a local archive mirror."""
    mirror = tmp_path / "mirror" / "BTCUSDT" / "1m"
    mirror.mkdir(parents=True)
    archive_path = mirror / "BTCUSDT-1m-2024-01.zip"
    csv_name = "BTCUSDT-1m-2024-01.csv"
    rows = [
        ["1704067200000", "100.0", "101.0", "99.0", "100.5", "12.0", "0", "0", "0", "0", "0", "0"],
        ["1704067260000", "100.5", "102.0", "100.0", "101.5", "14.0", "0", "0", "0", "0", "0", "0"],
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        payload = "\n".join(",".join(row) for row in rows) + "\n"
        archive.writestr(csv_name, payload)

    output = tmp_path / "btcusdt.csv"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1m",
            "--start-month",
            "2024-01",
            "--end-month",
            "2024-01",
            "--output",
            str(output),
            "--base-url",
            (tmp_path / "mirror").as_uri(),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        converted = list(csv.DictReader(handle))

    assert [row["timestamp"] for row in converted] == ["2024-01-01T00:00:00+00:00", "2024-01-01T00:01:00+00:00"]
    assert converted[0]["open"] == "100.0"
    assert converted[1]["close"] == "101.5"
