# Local End-to-End Backtest

This is a complete local research project that runs from a fresh checkout with
only Python 3.11+. It does not require exchange credentials, HuggingFace tokens,
TorchRL, or internet access.

The goal is not to claim trading alpha from synthetic data. The goal is to give
contributors a reproducible, inspectable pipeline that mirrors the structure a
paper-quality experiment should have before moving to real datasets and heavier
TorchRL agents.

## Workflow

The project is split into concrete tasks:

1. **Generate data**: create deterministic synthetic OHLCV candles with multiple regimes.
2. **Train**: grid-search a moving-average crossover policy on the train split.
3. **Evaluate**: backtest the saved policy on a chronological holdout split.
4. **Benchmark**: compare against transparent cash and buy-and-hold baselines.
5. **Validate**: run expanding-window walk-forward validation.
6. **Inspect artifacts**: review generated CSV and JSON outputs with metadata.

## Run everything

From the repository root:

```bash
python examples/end_to_end/local_backtest/run.py run-all
```

By default this writes artifacts to `outputs/local_backtest/`:

- `synthetic_ohlcv.csv`: generated market data
- `policy.json`: selected moving-average policy, train metrics, and leaderboard
- `evaluation.json`: holdout metrics plus baseline comparisons
- `benchmark_report.json`: metadata, selected policy, baselines, leaderboard, and walk-forward folds

## Run each task separately

```bash
python examples/end_to_end/local_backtest/run.py generate-data \
  --data-path outputs/local_backtest/synthetic_ohlcv.csv \
  --rows 720 \
  --seed 7

python examples/end_to_end/local_backtest/run.py train \
  --data-path outputs/local_backtest/synthetic_ohlcv.csv \
  --policy-path outputs/local_backtest/policy.json \
  --cost-bps 2 \
  --slippage-bps 1

python examples/end_to_end/local_backtest/run.py evaluate \
  --data-path outputs/local_backtest/synthetic_ohlcv.csv \
  --policy-path outputs/local_backtest/policy.json \
  --evaluation-path outputs/local_backtest/evaluation.json \
  --cost-bps 2 \
  --slippage-bps 1

python examples/end_to_end/local_backtest/run.py report \
  --data-path outputs/local_backtest/synthetic_ohlcv.csv \
  --report-path outputs/local_backtest/benchmark_report.json \
  --folds 3
```

## Customize output

```bash
python examples/end_to_end/local_backtest/run.py run-all \
  --output-dir /tmp/torchtrade-local-backtest \
  --rows 1440 \
  --seed 42 \
  --train-ratio 0.75 \
  --cost-bps 2 \
  --slippage-bps 1 \
  --folds 4
```

## Use a real public dataset

The default `run-all` command uses synthetic data so CI and onboarding never
depend on network access. For repository-level research, you should switch to a
real, versioned dataset as soon as possible. Two practical starting points are:

1. **TorchTrade datasets on Hugging Face**: the Torch-Trade organization hosts
   ready-to-use crypto OHLCV and feature datasets. Install the optional
   `datasets` dependency already listed in `pyproject.toml`, then load a dataset
   such as `Torch-Trade/btcusdt_perp_1m_05_2021_to_02_2026` with
   `datasets.load_dataset(...)`.
2. **Binance public data archive**: Binance publishes public monthly and daily
   kline ZIP files at `data.binance.vision` with no API key. Use the included
   dependency-free downloader to convert monthly klines into this example's
   canonical CSV schema:

```bash
python examples/end_to_end/local_backtest/download_binance.py \
  --symbol BTCUSDT \
  --interval 1m \
  --start-month 2024-01 \
  --end-month 2024-03 \
  --output data/btcusdt_1m_2024_q1.csv

python examples/end_to_end/local_backtest/run.py train \
  --data-path data/btcusdt_1m_2024_q1.csv \
  --policy-path outputs/local_backtest/policy.json

python examples/end_to_end/local_backtest/run.py evaluate \
  --data-path data/btcusdt_1m_2024_q1.csv \
  --policy-path outputs/local_backtest/policy.json \
  --evaluation-path outputs/local_backtest/evaluation.json
```

The downloader writes columns `timestamp,open,high,low,close,volume`, which are
accepted by the local benchmark commands. For publishable experiments, keep the
raw downloaded ZIP files or record their checksum files alongside the converted
CSV so reviewers can audit provenance.

## What this adds for a publication workflow

This example now includes several elements reviewers expect in empirical trading
research:

- chronological train/test splitting to avoid lookahead leakage;
- transaction-cost and slippage assumptions;
- cash and buy-and-hold baselines;
- grid-search leaderboard for transparent model selection;
- expanding-window walk-forward validation;
- artifact hashes, runtime metadata, seed, and config in `benchmark_report.json`.

## What you still need for a top-tier journal submission

Before submitting a serious TorchTrade paper, extend this local template with:

- real multi-asset datasets with clear licenses and survivorship-bias handling;
- multiple market regimes and out-of-sample periods;
- stronger baselines such as buy-and-hold, momentum, mean reversion, PPO/IQL/GRPO,
  and simple supervised or imitation-learning policies;
- statistical tests or bootstrap confidence intervals across assets, periods, and seeds;
- ablations for rewards, feature sets, transaction costs, and environment assumptions;
- reproducibility assets: locked dependencies, experiment configs, seeds, raw logs,
  generated figures, and hardware/runtime notes.

The example is intentionally lightweight so it can remain a stable local smoke
test before you scale the same workflow to real TorchRL experiments.
