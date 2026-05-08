"""Download Binance public kline data into the local backtest CSV schema.

The downloader uses Binance's public data archive at data.binance.vision. It is
kept dependency-free so users can fetch a real BTCUSDT dataset with only Python.

Example:

    python examples/end_to_end/local_backtest/download_binance.py \
        --symbol BTCUSDT \
        --interval 1m \
        --start-month 2024-01 \
        --end-month 2024-03 \
        --output data/btcusdt_1m_2024_q1.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

BINANCE_PUBLIC_DATA_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CanonicalBar:
    """OHLCV row compatible with the local backtest example."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_month(month: str) -> datetime:
    """Parse YYYY-MM into a UTC datetime at the first day of the month."""
    try:
        return datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid month {month!r}; expected YYYY-MM") from exc


def iter_months(start_month: str, end_month: str) -> list[str]:
    """Return inclusive YYYY-MM month labels."""
    current = parse_month(start_month)
    end = parse_month(end_month)
    if current > end:
        raise ValueError("start-month must be <= end-month")

    labels: list[str] = []
    while current <= end:
        labels.append(current.strftime("%Y-%m"))
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)
    return labels


def build_monthly_kline_url(base_url: str, symbol: str, interval: str, month: str) -> str:
    """Build the Binance monthly kline zip URL."""
    base = base_url.rstrip("/")
    return f"{base}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def timestamp_to_iso(open_time: str) -> str:
    """Convert Binance millisecond or microsecond timestamps to UTC ISO strings."""
    raw = int(open_time)
    # Binance spot archives use millisecond timestamps historically and
    # microsecond timestamps for newer files. Detect by magnitude.
    divisor = 1_000_000 if raw >= 10_000_000_000_000 else 1_000
    return datetime.fromtimestamp(raw / divisor, tz=UTC).isoformat()


def convert_binance_row(row: list[str]) -> CanonicalBar:
    """Convert a Binance kline row to the local canonical OHLCV schema."""
    if len(row) < 6:
        raise ValueError(f"expected at least 6 Binance kline columns, got {len(row)}")
    return CanonicalBar(
        timestamp=timestamp_to_iso(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def read_zip_bytes(url: str) -> bytes:
    """Read a remote or file:// zip URL."""
    try:
        with urllib.request.urlopen(url) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not download {url}; check network access or use --base-url with a local mirror"
        ) from exc


def download_month(url: str) -> list[CanonicalBar]:
    """Download and convert one Binance monthly kline zip archive."""
    payload = read_zip_bytes(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected one CSV in {url}, found {csv_names}")
        with archive.open(csv_names[0]) as csv_file:
            text_file = io.TextIOWrapper(csv_file, encoding="utf-8")
            return [convert_binance_row(row) for row in csv.reader(text_file) if row]


def write_canonical_csv(path: Path, bars: list[CanonicalBar]) -> None:
    """Write canonical OHLCV bars to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for bar in bars:
            writer.writerow(asdict(bar))


def download_range(
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
    output: Path,
    base_url: str = BINANCE_PUBLIC_DATA_BASE_URL,
) -> list[CanonicalBar]:
    """Download an inclusive month range and write one canonical CSV."""
    all_bars: list[CanonicalBar] = []
    for month in iter_months(start_month, end_month):
        url = build_monthly_kline_url(base_url, symbol, interval, month)
        print(f"downloading {url}")
        all_bars.extend(download_month(url))
    all_bars.sort(key=lambda bar: bar.timestamp)
    write_canonical_csv(output, all_bars)
    return all_bars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Binance monthly klines into TorchTrade local CSV format.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance spot symbol, for example BTCUSDT or ETHUSDT.")
    parser.add_argument("--interval", default="1m", help="Kline interval, for example 1m, 5m, 15m, 1h, or 1d.")
    parser.add_argument("--start-month", default="2024-01", help="Inclusive start month in YYYY-MM format.")
    parser.add_argument("--end-month", default="2024-01", help="Inclusive end month in YYYY-MM format.")
    parser.add_argument("--output", type=Path, default=Path("data/btcusdt_1m_2024_01.csv"))
    parser.add_argument(
        "--base-url",
        default=BINANCE_PUBLIC_DATA_BASE_URL,
        help="Override for tests or mirrors; defaults to Binance public spot monthly klines.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bars = download_range(args.symbol, args.interval, args.start_month, args.end_month, args.output, args.base_url)
    print(f"wrote {len(bars)} rows -> {args.output}")


if __name__ == "__main__":
    main()
