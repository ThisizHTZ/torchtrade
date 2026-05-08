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

## Use the 19 free Torch-Trade datasets

The default `run-all` command uses synthetic data so CI and onboarding never
depend on network access. For repository-level research, use the 19 free
Torch-Trade datasets hosted on Hugging Face instead of ad hoc exchange downloads.
The catalog is kept in `torchtrade_datasets.py` so every dataset can be validated
without network access, and optionally smoke-loaded when the `datasets` package
and network access are available.

```bash
# Validate all 19 catalog entries locally, no network required.
python examples/end_to_end/local_backtest/torchtrade_datasets.py validate-catalog

# Export an experiment manifest for your paper/reproduction bundle.
python examples/end_to_end/local_backtest/torchtrade_datasets.py catalog \
  --output outputs/local_backtest/torchtrade_free_datasets.csv

# Optional: actually load a small slice from every Hugging Face dataset.
python examples/end_to_end/local_backtest/torchtrade_datasets.py smoke-load \
  --max-rows 32
```

The 19-dataset catalog covers spot OHLCV, perpetual OHLCV, perpetual basis,
funding rates, market metrics, and book-ticker features:

| Group | Datasets |
|---|---:|
| Spot 1m OHLCV | 8 |
| Perpetual 1m OHLCV | 3 |
| Perpetual basis 1m | 3 |
| Perpetual funding 8h | 3 |
| BTC perpetual metrics 5m | 1 |
| BTC book-ticker features 1m | 1 |

For publishable experiments, record the Hugging Face dataset ids, revisions,
local cache paths, checksums, and any filtering/resampling code in the experiment
manifest.

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
