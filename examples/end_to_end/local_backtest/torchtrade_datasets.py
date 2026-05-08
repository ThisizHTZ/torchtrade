"""TorchTrade free dataset catalog and optional Hugging Face smoke loader.

This module intentionally keeps the canonical list of free Torch-Trade datasets
in source control so examples and tests can validate the whole dataset matrix
without relying on network access. When the optional ``datasets`` package and
network access are available, use ``smoke-load`` to load a small slice from one
or more datasets.

Examples:

    python examples/end_to_end/local_backtest/torchtrade_datasets.py catalog
    python examples/end_to_end/local_backtest/torchtrade_datasets.py validate-catalog
    python examples/end_to_end/local_backtest/torchtrade_datasets.py smoke-load --max-rows 32 --limit 2
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

EXPECTED_DATASET_COUNT = 19


@dataclass(frozen=True)
class TorchTradeDataset:
    """Metadata for one free Torch-Trade dataset hosted on Hugging Face."""

    dataset_id: str
    asset: str
    market: str
    frequency: str
    start: str
    end: str
    rows_label: str
    expected_columns: tuple[str, ...]
    description: str


BASE_OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
BASIS_COLUMNS = BASE_OHLCV_COLUMNS + ("index_price", "mark_price", "basis", "basis_pct")
FUNDING_COLUMNS = ("timestamp", "funding_rate")
METRICS_COLUMNS = ("timestamp",)
BOOKTICKER_FEATURE_COLUMNS = ("timestamp",)


FREE_DATASETS: tuple[TorchTradeDataset, ...] = (
    TorchTradeDataset(
        "Torch-Trade/btcusdt_spot_1m_05_2021_to_03_2026",
        "BTCUSDT",
        "spot",
        "1m",
        "2021-05",
        "2026-03",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "BTC/USDT spot one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/btcusdt_spot_1m_03_2023_to_03_2026",
        "BTCUSDT",
        "spot",
        "1m",
        "2023-03",
        "2026-03",
        "1.58M",
        BASE_OHLCV_COLUMNS,
        "Shorter BTC/USDT spot one-minute OHLCV window for faster experiments.",
    ),
    TorchTradeDataset(
        "Torch-Trade/ethusdt_spot_1m_05_2021_to_03_2026",
        "ETHUSDT",
        "spot",
        "1m",
        "2021-05",
        "2026-03",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "ETH/USDT spot one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/bnbusdt_spot_1m_10_2021_to_03_2026",
        "BNBUSDT",
        "spot",
        "1m",
        "2021-10",
        "2026-03",
        "2.32M",
        BASE_OHLCV_COLUMNS,
        "BNB/USDT spot one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/xrpusdt_spot_1m_05_2021_to_03_2026",
        "XRPUSDT",
        "spot",
        "1m",
        "2021-05",
        "2026-03",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "XRP/USDT spot one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/trxusdt_spot_1m_05_2021_to_03_2026",
        "TRXUSDT",
        "spot",
        "1m",
        "2021-05",
        "2026-03",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "TRX/USDT spot one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/adausdt_spot_1m_05_2021_to_03_2026",
        "ADAUSDT",
        "spot",
        "1m",
        "2021-05",
        "2026-03",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "ADA/USDT spot one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/solusdt_spot_1m_05_2021_to_03_2026",
        "SOLUSDT",
        "spot",
        "1m",
        "2021-05",
        "2026-03",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "SOL/USDT spot one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/btcusdt_perp_1m_05_2021_to_02_2026",
        "BTCUSDT",
        "perp",
        "1m",
        "2021-05",
        "2026-02",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "BTC/USDT perpetual futures one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/ethusdt_perp_1m_05_2021_to_02_2026",
        "ETHUSDT",
        "perp",
        "1m",
        "2021-05",
        "2026-02",
        "2.54M",
        BASE_OHLCV_COLUMNS,
        "ETH/USDT perpetual futures one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/bnbusdt_perp_1m_10_2021_to_02_2026",
        "BNBUSDT",
        "perp",
        "1m",
        "2021-10",
        "2026-02",
        "2.32M",
        BASE_OHLCV_COLUMNS,
        "BNB/USDT perpetual futures one-minute OHLCV.",
    ),
    TorchTradeDataset(
        "Torch-Trade/btcusdt_perp_basis_1m_05_2021_to_02_2026",
        "BTCUSDT",
        "perp_basis",
        "1m",
        "2021-05",
        "2026-02",
        "2.53M",
        BASIS_COLUMNS,
        "BTC/USDT perpetual basis features at one-minute frequency.",
    ),
    TorchTradeDataset(
        "Torch-Trade/ethusdt_perp_basis_1m_05_2021_to_02_2026",
        "ETHUSDT",
        "perp_basis",
        "1m",
        "2021-05",
        "2026-02",
        "2.53M",
        BASIS_COLUMNS,
        "ETH/USDT perpetual basis features at one-minute frequency.",
    ),
    TorchTradeDataset(
        "Torch-Trade/bnbusdt_perp_basis_1m_10_2021_to_02_2026",
        "BNBUSDT",
        "perp_basis",
        "1m",
        "2021-10",
        "2026-02",
        "2.32M",
        BASIS_COLUMNS,
        "BNB/USDT perpetual basis features at one-minute frequency.",
    ),
    TorchTradeDataset(
        "Torch-Trade/btcusdt_perp_funding_8h_05_2021_to_02_2026",
        "BTCUSDT",
        "perp_funding",
        "8h",
        "2021-05",
        "2026-02",
        "5.3k",
        FUNDING_COLUMNS,
        "BTC/USDT perpetual funding rates at eight-hour frequency.",
    ),
    TorchTradeDataset(
        "Torch-Trade/ethusdt_perp_funding_8h_05_2021_to_02_2026",
        "ETHUSDT",
        "perp_funding",
        "8h",
        "2021-05",
        "2026-02",
        "5.3k",
        FUNDING_COLUMNS,
        "ETH/USDT perpetual funding rates at eight-hour frequency.",
    ),
    TorchTradeDataset(
        "Torch-Trade/bnbusdt_perp_funding_8h_10_2021_to_02_2026",
        "BNBUSDT",
        "perp_funding",
        "8h",
        "2021-10",
        "2026-02",
        "4.84k",
        FUNDING_COLUMNS,
        "BNB/USDT perpetual funding rates at eight-hour frequency.",
    ),
    TorchTradeDataset(
        "Torch-Trade/btcusdt_perp_metrics_5m_09_2020_to_04_2026",
        "BTCUSDT",
        "perp_metrics",
        "5m",
        "2020-09",
        "2026-04",
        "593k",
        METRICS_COLUMNS,
        "BTC/USDT derivatives market metrics at five-minute frequency.",
    ),
    TorchTradeDataset(
        "Torch-Trade/btcusdt_perp_bookticker_features_1m_05_2023_to_03_2024",
        "BTCUSDT",
        "perp_bookticker_features",
        "1m",
        "2023-05",
        "2024-03",
        "460k",
        BOOKTICKER_FEATURE_COLUMNS,
        "BTC/USDT book-ticker-derived one-minute features.",
    ),
)


def dataset_catalog() -> list[dict[str, object]]:
    """Return the dataset catalog as JSON-serializable dictionaries."""
    return [asdict(dataset) for dataset in FREE_DATASETS]


def validate_catalog(datasets: Iterable[TorchTradeDataset] = FREE_DATASETS) -> list[str]:
    """Validate all catalog entries and return their dataset ids."""
    entries = list(datasets)
    if len(entries) != EXPECTED_DATASET_COUNT:
        raise ValueError(f"expected {EXPECTED_DATASET_COUNT} datasets, found {len(entries)}")

    seen: set[str] = set()
    for entry in entries:
        if not entry.dataset_id.startswith("Torch-Trade/"):
            raise ValueError(f"dataset id must use Torch-Trade namespace: {entry.dataset_id}")
        if entry.dataset_id in seen:
            raise ValueError(f"duplicate dataset id: {entry.dataset_id}")
        if not entry.expected_columns or "timestamp" not in entry.expected_columns:
            raise ValueError(f"dataset must include timestamp column metadata: {entry.dataset_id}")
        if entry.start > entry.end:
            raise ValueError(f"invalid date range for {entry.dataset_id}: {entry.start} > {entry.end}")
        seen.add(entry.dataset_id)
    return [entry.dataset_id for entry in entries]


def write_catalog_csv(path: Path) -> None:
    """Write the catalog to CSV for experiment manifests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TorchTradeDataset.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in FREE_DATASETS:
            row = asdict(dataset)
            row["expected_columns"] = ",".join(dataset.expected_columns)
            writer.writerow(row)


def smoke_load_dataset(dataset_id: str, max_rows: int) -> dict[str, object]:
    """Load a small split slice from Hugging Face when optional deps/network exist."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the optional `datasets` dependency to smoke-load remote datasets") from exc

    split = f"train[:{max_rows}]"
    dataset = load_dataset(dataset_id, split=split)
    return {"dataset_id": dataset_id, "rows": len(dataset), "columns": list(dataset.column_names)}


def command_catalog(args: argparse.Namespace) -> None:
    validate_catalog()
    payload = dataset_catalog()
    if args.output:
        write_catalog_csv(args.output)
        print(f"wrote {len(payload)} datasets -> {args.output}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def command_validate_catalog(_args: argparse.Namespace) -> None:
    ids = validate_catalog()
    for index, dataset_id in enumerate(ids, start=1):
        print(f"{index:02d}/{len(ids)} OK {dataset_id}")


def command_smoke_load(args: argparse.Namespace) -> None:
    validate_catalog()
    selected = FREE_DATASETS[: args.limit] if args.limit else FREE_DATASETS
    results = []
    for dataset in selected:
        try:
            results.append(smoke_load_dataset(dataset.dataset_id, args.max_rows))
        except Exception as exc:  # noqa: BLE001 - CLI should report all dataset failures, not stop at the first.
            results.append({"dataset_id": dataset.dataset_id, "error": str(exc)})
    print(json.dumps(results, indent=2, sort_keys=True))
    failed = [result for result in results if "error" in result]
    if failed:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and smoke-test Torch-Trade free Hugging Face datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="Print the 19-dataset catalog or write it as CSV.")
    catalog.add_argument("--output", type=Path, help="Optional CSV output path for the catalog.")
    catalog.set_defaults(func=command_catalog)

    validate = subparsers.add_parser("validate-catalog", help="Validate every catalog entry without network access.")
    validate.set_defaults(func=command_validate_catalog)

    smoke = subparsers.add_parser("smoke-load", help="Load a small slice from Hugging Face for each dataset.")
    smoke.add_argument("--max-rows", type=int, default=16, help="Rows to request from each dataset.")
    smoke.add_argument("--limit", type=int, default=0, help="Optional limit for quick local checks; 0 means all 19.")
    smoke.set_defaults(func=command_smoke_load)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
