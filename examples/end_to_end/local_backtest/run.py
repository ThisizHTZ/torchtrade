"""Local end-to-end TorchTrade-style backtest project.

This example is intentionally dependency-free so it can be run from a fresh
checkout with only Python 3.11+. It demonstrates the full workflow used by a
trading research project:

1. generate a deterministic OHLCV dataset,
2. train/select a simple moving-average crossover policy,
3. evaluate the selected policy on a holdout split,
4. write reproducible artifacts to disk.

Run the whole pipeline:

    python examples/end_to_end/local_backtest/run.py run-all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

STARTING_CASH = 10_000.0
DEFAULT_ROWS = 480
DEFAULT_TRAIN_RATIO = 0.7


@dataclass(frozen=True)
class Bar:
    """One OHLCV candle used by the local backtest."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class PolicyConfig:
    """Moving-average crossover policy parameters."""

    short_window: int
    long_window: int


@dataclass(frozen=True)
class BacktestSummary:
    """Serializable metrics for a completed backtest."""

    total_return: float
    final_equity: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    trades: int
    bars: int


def generate_synthetic_bars(rows: int = DEFAULT_ROWS, seed: int = 7) -> list[Bar]:
    """Create deterministic OHLCV bars with trend, seasonality, and noise."""
    if rows < 60:
        raise ValueError("rows must be at least 60 so moving averages have enough data")

    rng = random.Random(seed)
    current_time = datetime(2025, 1, 1, tzinfo=UTC)
    previous_close = 100.0
    bars: list[Bar] = []

    for index in range(rows):
        trend = 0.018 * index / rows
        seasonal = 1.4 * math.sin(index / 18.0) + 0.6 * math.sin(index / 7.0)
        noise = rng.gauss(0.0, 0.35)
        close = max(1.0, previous_close * (1.0 + trend / rows) + seasonal * 0.04 + noise)
        open_price = previous_close
        high = max(open_price, close) + abs(rng.gauss(0.12, 0.05))
        low = min(open_price, close) - abs(rng.gauss(0.12, 0.05))
        volume = 1_000.0 + 40.0 * math.sin(index / 9.0) + rng.uniform(0.0, 80.0)
        bars.append(
            Bar(
                timestamp=current_time.isoformat(),
                open=round(open_price, 6),
                high=round(high, 6),
                low=round(max(0.01, low), 6),
                close=round(close, 6),
                volume=round(volume, 6),
            )
        )
        previous_close = close
        current_time += timedelta(minutes=1)

    return bars


def write_bars(path: Path, bars: Iterable[Bar]) -> None:
    """Write bars to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Bar.__dataclass_fields__.keys()))
        writer.writeheader()
        for bar in bars:
            writer.writerow(asdict(bar))


def read_bars(path: Path) -> list[Bar]:
    """Read bars from CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            Bar(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in reader
        ]


def split_bars(bars: list[Bar], train_ratio: float = DEFAULT_TRAIN_RATIO) -> tuple[list[Bar], list[Bar]]:
    """Split bars into chronological train and evaluation windows."""
    if not 0.1 < train_ratio < 0.9:
        raise ValueError("train_ratio must be between 0.1 and 0.9")
    split_index = int(len(bars) * train_ratio)
    return bars[:split_index], bars[split_index:]


def moving_average(values: list[float], window: int, index: int) -> float | None:
    """Return the trailing moving average ending at index, if available."""
    start = index - window + 1
    if start < 0:
        return None
    return mean(values[start : index + 1])


def signal_for_index(closes: list[float], policy: PolicyConfig, index: int) -> int:
    """Return 1 for long exposure and 0 for cash."""
    short_ma = moving_average(closes, policy.short_window, index)
    long_ma = moving_average(closes, policy.long_window, index)
    if short_ma is None or long_ma is None:
        return 0
    return int(short_ma > long_ma)


def max_drawdown(equity_curve: list[float]) -> float:
    """Return maximum drawdown as a negative percentage."""
    peak = equity_curve[0]
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak
        worst = min(worst, drawdown)
    return worst


def sharpe_ratio(returns: list[float], periods_per_year: float = 365 * 24 * 60) -> float:
    """Compute a simple annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    volatility = pstdev(returns)
    if volatility == 0:
        return 0.0
    return mean(returns) / volatility * math.sqrt(periods_per_year)


def run_backtest(bars: list[Bar], policy: PolicyConfig) -> BacktestSummary:
    """Backtest a moving-average crossover policy on the supplied bars."""
    if len(bars) <= policy.long_window:
        raise ValueError("not enough bars for the selected policy")

    closes = [bar.close for bar in bars]
    equity = STARTING_CASH
    equity_curve = [equity]
    returns: list[float] = []
    winning_periods = 0
    active_periods = 0
    previous_signal = 0
    trades = 0

    for index in range(1, len(closes)):
        signal = signal_for_index(closes, policy, index - 1)
        if signal != previous_signal:
            trades += 1
            previous_signal = signal

        period_return = signal * ((closes[index] - closes[index - 1]) / closes[index - 1])
        equity *= 1.0 + period_return
        equity_curve.append(equity)
        returns.append(period_return)
        if signal:
            active_periods += 1
            if period_return > 0:
                winning_periods += 1

    return BacktestSummary(
        total_return=(equity_curve[-1] - equity_curve[0]) / equity_curve[0],
        final_equity=equity_curve[-1],
        max_drawdown=max_drawdown(equity_curve),
        sharpe=sharpe_ratio(returns),
        win_rate=winning_periods / active_periods if active_periods else 0.0,
        trades=trades,
        bars=len(bars),
    )


def train_policy(bars: list[Bar]) -> tuple[PolicyConfig, BacktestSummary]:
    """Grid-search a compact moving-average policy space."""
    candidates = [
        PolicyConfig(short_window=short, long_window=long)
        for short in (3, 5, 8, 13)
        for long in (21, 34, 55)
        if short < long
    ]
    scored = [(candidate, run_backtest(bars, candidate)) for candidate in candidates]
    return max(scored, key=lambda item: (item[1].total_return, item[1].sharpe))


def write_json(path: Path, payload: dict) -> None:
    """Write pretty JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_policy(path: Path) -> PolicyConfig:
    """Load a policy JSON file produced by the train command."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = payload["policy"]
    return PolicyConfig(short_window=int(policy["short_window"]), long_window=int(policy["long_window"]))


def default_paths(output_dir: Path) -> dict[str, Path]:
    """Return canonical artifact paths for an output directory."""
    return {
        "data": output_dir / "synthetic_ohlcv.csv",
        "policy": output_dir / "policy.json",
        "evaluation": output_dir / "evaluation.json",
    }


def command_generate_data(args: argparse.Namespace) -> None:
    bars = generate_synthetic_bars(rows=args.rows, seed=args.seed)
    write_bars(args.data_path, bars)
    print(f"generated {len(bars)} rows -> {args.data_path}")


def command_train(args: argparse.Namespace) -> None:
    bars = read_bars(args.data_path)
    train_bars, _ = split_bars(bars, train_ratio=args.train_ratio)
    policy, summary = train_policy(train_bars)
    write_json(args.policy_path, {"policy": asdict(policy), "train_metrics": asdict(summary)})
    print(f"trained policy {asdict(policy)} -> {args.policy_path}")


def command_evaluate(args: argparse.Namespace) -> None:
    bars = read_bars(args.data_path)
    _, eval_bars = split_bars(bars, train_ratio=args.train_ratio)
    policy = load_policy(args.policy_path)
    summary = run_backtest(eval_bars, policy)
    write_json(args.evaluation_path, {"policy": asdict(policy), "evaluation_metrics": asdict(summary)})
    print(f"evaluated policy -> {args.evaluation_path}")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


def command_run_all(args: argparse.Namespace) -> None:
    paths = default_paths(args.output_dir)
    command_generate_data(argparse.Namespace(data_path=paths["data"], rows=args.rows, seed=args.seed))
    command_train(
        argparse.Namespace(data_path=paths["data"], policy_path=paths["policy"], train_ratio=args.train_ratio)
    )
    command_evaluate(
        argparse.Namespace(
            data_path=paths["data"],
            policy_path=paths["policy"],
            evaluation_path=paths["evaluation"],
            train_ratio=args.train_ratio,
        )
    )
    print(f"artifacts written in {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a dependency-free local trading backtest pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="Create a deterministic synthetic OHLCV CSV.")
    generate.add_argument("--data-path", type=Path, default=Path("outputs/local_backtest/synthetic_ohlcv.csv"))
    generate.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    generate.add_argument("--seed", type=int, default=7)
    generate.set_defaults(func=command_generate_data)

    train = subparsers.add_parser("train", help="Select the best moving-average policy on the train split.")
    train.add_argument("--data-path", type=Path, default=Path("outputs/local_backtest/synthetic_ohlcv.csv"))
    train.add_argument("--policy-path", type=Path, default=Path("outputs/local_backtest/policy.json"))
    train.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved policy on the holdout split.")
    evaluate.add_argument("--data-path", type=Path, default=Path("outputs/local_backtest/synthetic_ohlcv.csv"))
    evaluate.add_argument("--policy-path", type=Path, default=Path("outputs/local_backtest/policy.json"))
    evaluate.add_argument("--evaluation-path", type=Path, default=Path("outputs/local_backtest/evaluation.json"))
    evaluate.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    evaluate.set_defaults(func=command_evaluate)

    run_all = subparsers.add_parser("run-all", help="Run data generation, training, and evaluation.")
    run_all.add_argument("--output-dir", type=Path, default=Path("outputs/local_backtest"))
    run_all.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    run_all.add_argument("--seed", type=int, default=7)
    run_all.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    run_all.set_defaults(func=command_run_all)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
