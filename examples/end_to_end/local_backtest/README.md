# Local End-to-End Backtest

This is a complete local project that runs from a fresh checkout with only
Python 3.11+. It does not require exchange credentials, HuggingFace tokens,
TorchRL, or internet access.

## Workflow

The project is split into concrete tasks:

1. **Generate data**: create deterministic synthetic OHLCV candles.
2. **Train**: grid-search a moving-average crossover policy on the train split.
3. **Evaluate**: backtest the saved policy on a chronological holdout split.
4. **Inspect artifacts**: review generated CSV and JSON outputs.

## Run everything

From the repository root:

```bash
python examples/end_to_end/local_backtest/run.py run-all
```

By default this writes artifacts to `outputs/local_backtest/`:

- `synthetic_ohlcv.csv`: generated market data
- `policy.json`: selected moving-average policy and train metrics
- `evaluation.json`: holdout evaluation metrics

## Run each task separately

```bash
python examples/end_to_end/local_backtest/run.py generate-data \
  --data-path outputs/local_backtest/synthetic_ohlcv.csv \
  --rows 480 \
  --seed 7

python examples/end_to_end/local_backtest/run.py train \
  --data-path outputs/local_backtest/synthetic_ohlcv.csv \
  --policy-path outputs/local_backtest/policy.json

python examples/end_to_end/local_backtest/run.py evaluate \
  --data-path outputs/local_backtest/synthetic_ohlcv.csv \
  --policy-path outputs/local_backtest/policy.json \
  --evaluation-path outputs/local_backtest/evaluation.json
```

## Customize output

```bash
python examples/end_to_end/local_backtest/run.py run-all \
  --output-dir /tmp/torchtrade-local-backtest \
  --rows 720 \
  --seed 42 \
  --train-ratio 0.75
```

The example is intentionally simple so it can be used as a stable local smoke
test before trying heavier TorchRL examples or live exchange integrations.
