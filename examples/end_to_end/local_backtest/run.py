"""Research-oriented local TorchTrade-style backtest pipeline.

This example is dependency-free and runnable from a fresh checkout with only
Python 3.11+. It demonstrates a reproducible paper-style experiment loop:

1. generate deterministic OHLCV data,
2. train/select a moving-average crossover policy,
3. evaluate against simple baselines with costs and slippage,
4. run walk-forward validation,
5. write machine-readable artifacts for review and replication.

Run the whole pipeline:

    python examples/end_to_end/local_backtest/run.py run-all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable

STARTING_CASH = 10_000.0
DEFAULT_ROWS = 720
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_COST_BPS = 2.0
DEFAULT_SLIPPAGE_BPS = 1.0
PERIODS_PER_YEAR = 365 * 24 * 60


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
    exposure: float
    turnover: float
    total_cost: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducibility metadata for the local experiment."""

    rows: int
    seed: int
    train_ratio: float
    cost_bps: float
    slippage_bps: float
    starting_cash: float = STARTING_CASH


def generate_synthetic_bars(rows: int = DEFAULT_ROWS, seed: int = 7) -> list[Bar]:
    """Create deterministic OHLCV bars with regimes, seasonality, and noise."""
    if rows < 180:
        raise ValueError("rows must be at least 180 for train/eval and walk-forward validation")

    rng = random.Random(seed)
    current_time = datetime(2025, 1, 1, tzinfo=UTC)
    previous_close = 100.0
    bars: list[Bar] = []

    for index in range(rows):
        regime = 0.00018 if index < rows * 0.35 else (-0.00012 if index < rows * 0.65 else 0.00008)
        seasonal = 0.0016 * math.sin(index / 18.0) + 0.0008 * math.sin(index / 7.0)
        shock = rng.gauss(0.0, 0.0032)
        close = max(1.0, previous_close * (1.0 + regime + seasonal + shock))
        open_price = previous_close
        spread = abs(rng.gauss(0.0015, 0.0005)) * previous_close
        high = max(open_price, close) + spread
        low = min(open_price, close) - spread
        volume = 1_000.0 + 80.0 * math.sin(index / 9.0) + rng.uniform(0.0, 160.0)
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


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def ma_signal(closes: list[float], policy: PolicyConfig, index: int) -> int:
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


def sharpe_ratio(returns: list[float], periods_per_year: float = PERIODS_PER_YEAR) -> float:
    """Compute a simple annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    volatility = pstdev(returns)
    if volatility == 0:
        return 0.0
    return mean(returns) / volatility * math.sqrt(periods_per_year)


def summarize_backtest(
    bars: list[Bar],
    signal_fn: Callable[[list[float], int], int],
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BacktestSummary:
    """Backtest a long/cash signal with turnover costs and slippage."""
    if len(bars) < 2:
        raise ValueError("at least two bars are required")

    closes = [bar.close for bar in bars]
    equity = STARTING_CASH
    equity_curve = [equity]
    returns: list[float] = []
    winning_periods = 0
    active_periods = 0
    previous_signal = 0
    trades = 0
    turnover = 0.0
    total_cost = 0.0
    exposure_sum = 0.0
    round_trip_cost = (cost_bps + slippage_bps) / 10_000.0

    for index in range(1, len(closes)):
        signal = signal_fn(closes, index - 1)
        signal = 1 if signal else 0
        period_turnover = abs(signal - previous_signal)
        if period_turnover:
            trades += 1
            turnover += period_turnover
        gross_return = signal * ((closes[index] - closes[index - 1]) / closes[index - 1])
        cost_return = period_turnover * round_trip_cost
        period_return = gross_return - cost_return
        equity *= 1.0 + period_return
        equity_curve.append(equity)
        returns.append(period_return)
        total_cost += equity_curve[-2] * cost_return
        exposure_sum += signal
        if signal:
            active_periods += 1
            if period_return > 0:
                winning_periods += 1
        previous_signal = signal

    return BacktestSummary(
        total_return=(equity_curve[-1] - equity_curve[0]) / equity_curve[0],
        final_equity=equity_curve[-1],
        max_drawdown=max_drawdown(equity_curve),
        sharpe=sharpe_ratio(returns),
        win_rate=winning_periods / active_periods if active_periods else 0.0,
        trades=trades,
        bars=len(bars),
        exposure=exposure_sum / max(1, len(closes) - 1),
        turnover=turnover,
        total_cost=total_cost,
    )


def run_backtest(
    bars: list[Bar],
    policy: PolicyConfig,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BacktestSummary:
    """Backtest a moving-average crossover policy on the supplied bars."""
    if len(bars) <= policy.long_window:
        raise ValueError("not enough bars for the selected policy")
    return summarize_backtest(
        bars,
        signal_fn=lambda closes, index: ma_signal(closes, policy, index),
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )


def run_baselines(
    bars: list[Bar],
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict[str, BacktestSummary]:
    """Evaluate transparent local baselines for comparison."""
    return {
        "cash": summarize_backtest(bars, lambda _closes, _index: 0, cost_bps, slippage_bps),
        "buy_and_hold": summarize_backtest(bars, lambda _closes, _index: 1, cost_bps, slippage_bps),
    }


def train_policy(
    bars: list[Bar],
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> tuple[PolicyConfig, BacktestSummary, list[dict[str, object]]]:
    """Grid-search a compact moving-average policy space."""
    candidates = [
        PolicyConfig(short_window=short, long_window=long)
        for short in (3, 5, 8, 13, 21)
        for long in (34, 55, 89)
        if short < long
    ]
    leaderboard: list[dict[str, object]] = []
    for candidate in candidates:
        metrics = run_backtest(bars, candidate, cost_bps, slippage_bps)
        leaderboard.append({"policy": asdict(candidate), "metrics": asdict(metrics)})
    leaderboard.sort(key=lambda item: (item["metrics"]["total_return"], item["metrics"]["sharpe"]), reverse=True)
    best = leaderboard[0]
    policy = PolicyConfig(**best["policy"])
    return policy, BacktestSummary(**best["metrics"]), leaderboard


def walk_forward_validation(
    bars: list[Bar],
    folds: int = 3,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> list[dict[str, object]]:
    """Run expanding-window train/test validation for temporal robustness."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    fold_size = len(bars) // (folds + 2)
    if fold_size < 60:
        raise ValueError("not enough bars for walk-forward validation")

    results: list[dict[str, object]] = []
    for fold in range(folds):
        train_end = fold_size * (fold + 2)
        test_end = train_end + fold_size
        train_bars = bars[:train_end]
        test_bars = bars[train_end:test_end]
        policy, train_metrics, _ = train_policy(train_bars, cost_bps, slippage_bps)
        eval_metrics = run_backtest(test_bars, policy, cost_bps, slippage_bps)
        results.append(
            {
                "fold": fold + 1,
                "train_start": train_bars[0].timestamp,
                "train_end": train_bars[-1].timestamp,
                "test_start": test_bars[0].timestamp,
                "test_end": test_bars[-1].timestamp,
                "policy": asdict(policy),
                "train_metrics": asdict(train_metrics),
                "evaluation_metrics": asdict(eval_metrics),
            }
        )
    return results


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
        "report": output_dir / "benchmark_report.json",
    }


def build_metadata(config: ExperimentConfig, data_path: Path) -> dict[str, object]:
    """Build reproducibility metadata for experiment artifacts."""
    return {
        "config": asdict(config),
        "data_sha256": sha256_file(data_path) if data_path.exists() else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "schema_version": 1,
    }


def command_generate_data(args: argparse.Namespace) -> None:
    bars = generate_synthetic_bars(rows=args.rows, seed=args.seed)
    write_bars(args.data_path, bars)
    print(f"generated {len(bars)} rows -> {args.data_path}")


def command_train(args: argparse.Namespace) -> None:
    bars = read_bars(args.data_path)
    train_bars, _ = split_bars(bars, train_ratio=args.train_ratio)
    policy, summary, leaderboard = train_policy(train_bars, args.cost_bps, args.slippage_bps)
    write_json(
        args.policy_path,
        {
            "policy": asdict(policy),
            "train_metrics": asdict(summary),
            "leaderboard": leaderboard[:10],
            "cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
        },
    )
    print(f"trained policy {asdict(policy)} -> {args.policy_path}")


def command_evaluate(args: argparse.Namespace) -> None:
    bars = read_bars(args.data_path)
    _, eval_bars = split_bars(bars, train_ratio=args.train_ratio)
    policy = load_policy(args.policy_path)
    summary = run_backtest(eval_bars, policy, args.cost_bps, args.slippage_bps)
    baselines = run_baselines(eval_bars, args.cost_bps, args.slippage_bps)
    write_json(
        args.evaluation_path,
        {
            "policy": asdict(policy),
            "evaluation_metrics": asdict(summary),
            "baselines": {name: asdict(metrics) for name, metrics in baselines.items()},
            "cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
        },
    )
    print(f"evaluated policy -> {args.evaluation_path}")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


def command_report(args: argparse.Namespace) -> None:
    bars = read_bars(args.data_path)
    train_bars, eval_bars = split_bars(bars, train_ratio=args.train_ratio)
    policy, train_summary, leaderboard = train_policy(train_bars, args.cost_bps, args.slippage_bps)
    evaluation_summary = run_backtest(eval_bars, policy, args.cost_bps, args.slippage_bps)
    baselines = run_baselines(eval_bars, args.cost_bps, args.slippage_bps)
    config = ExperimentConfig(args.rows, args.seed, args.train_ratio, args.cost_bps, args.slippage_bps)
    report = {
        "metadata": build_metadata(config, args.data_path),
        "selected_policy": asdict(policy),
        "train_metrics": asdict(train_summary),
        "evaluation_metrics": asdict(evaluation_summary),
        "baselines": {name: asdict(metrics) for name, metrics in baselines.items()},
        "leaderboard": leaderboard[:10],
        "walk_forward": walk_forward_validation(bars, args.folds, args.cost_bps, args.slippage_bps),
    }
    write_json(args.report_path, report)
    print(f"wrote benchmark report -> {args.report_path}")


def command_run_all(args: argparse.Namespace) -> None:
    paths = default_paths(args.output_dir)
    command_generate_data(argparse.Namespace(data_path=paths["data"], rows=args.rows, seed=args.seed))
    common = {
        "data_path": paths["data"],
        "train_ratio": args.train_ratio,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "rows": args.rows,
        "seed": args.seed,
    }
    command_train(argparse.Namespace(**common, policy_path=paths["policy"]))
    command_evaluate(argparse.Namespace(**common, policy_path=paths["policy"], evaluation_path=paths["evaluation"]))
    command_report(argparse.Namespace(**common, report_path=paths["report"], folds=args.folds))
    print(f"artifacts written in {args.output_dir}")


def add_common_research_args(parser: argparse.ArgumentParser) -> None:
    """Add cost and split arguments shared by research commands."""
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)


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
    add_common_research_args(train)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved policy on the holdout split.")
    evaluate.add_argument("--data-path", type=Path, default=Path("outputs/local_backtest/synthetic_ohlcv.csv"))
    evaluate.add_argument("--policy-path", type=Path, default=Path("outputs/local_backtest/policy.json"))
    evaluate.add_argument("--evaluation-path", type=Path, default=Path("outputs/local_backtest/evaluation.json"))
    add_common_research_args(evaluate)
    evaluate.set_defaults(func=command_evaluate)

    report = subparsers.add_parser("report", help="Write a paper-style benchmark report JSON.")
    report.add_argument("--data-path", type=Path, default=Path("outputs/local_backtest/synthetic_ohlcv.csv"))
    report.add_argument("--report-path", type=Path, default=Path("outputs/local_backtest/benchmark_report.json"))
    report.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    report.add_argument("--seed", type=int, default=7)
    report.add_argument("--folds", type=int, default=3)
    add_common_research_args(report)
    report.set_defaults(func=command_report)

    run_all = subparsers.add_parser("run-all", help="Run data generation, training, evaluation, and reporting.")
    run_all.add_argument("--output-dir", type=Path, default=Path("outputs/local_backtest"))
    run_all.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    run_all.add_argument("--seed", type=int, default=7)
    run_all.add_argument("--folds", type=int, default=3)
    add_common_research_args(run_all)
    run_all.set_defaults(func=command_run_all)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
