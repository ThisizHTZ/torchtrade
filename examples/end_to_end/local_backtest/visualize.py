"""Create dependency-free SVG visualizations for local backtest artifacts.

The script renders two useful review artifacts without matplotlib:

* a benchmark summary chart from ``benchmark_report.json``;
* a dataset smoke-load status dashboard from ``torchtrade_datasets.py smoke-load`` JSON.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


GREEN = "#2e7d32"
RED = "#c62828"
BLUE = "#1565c0"
GRAY = "#546e7a"
LIGHT = "#eef3f7"
DARK = "#263238"
ORANGE = "#ef6c00"


def pct(value: float) -> str:
    """Format a decimal return as percentage text."""
    return f"{value * 100:.2f}%"


def bar_width(value: float, max_abs: float, width: int) -> int:
    """Scale absolute value to a positive bar width."""
    if max_abs == 0:
        return 0
    return int(abs(value) / max_abs * width)


def text(x: int, y: int, body: str, size: int = 14, fill: str = DARK, weight: str = "400") -> str:
    """Return an SVG text element."""
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(body)}</text>'


def rect(x: int, y: int, width: int, height: int, fill: str, radius: int = 4) -> str:
    """Return an SVG rounded rectangle element."""
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" />'


def write_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    """Write a complete SVG document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Inter,Arial,sans-serif}.small{font-size:12px}</style>',
        rect(0, 0, width, height, "#ffffff", 0),
        *elements,
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def render_benchmark(report_path: Path, output_path: Path) -> None:
    """Render strategy/baseline and walk-forward returns from a benchmark report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    strategy = report["evaluation_metrics"]
    series = [("strategy", strategy)] + list(report["baselines"].items())
    max_abs_return = max(abs(metrics["total_return"]) for _, metrics in series) or 1.0

    elements = [text(28, 38, "Local Backtest Benchmark", 24, DARK, "700")]
    elements.append(text(28, 64, f"Source: {report_path}", 12, GRAY))
    y = 104
    for name, metrics in series:
        value = metrics["total_return"]
        color = GREEN if value >= 0 else RED
        elements.append(text(28, y + 15, name, 14, DARK, "700"))
        elements.append(rect(170, y, 360, 20, LIGHT))
        elements.append(rect(170, y, max(2, bar_width(value, max_abs_return, 360)), 20, color))
        elements.append(text(548, y + 15, f"return {pct(value)} | maxDD {pct(metrics['max_drawdown'])}", 13, DARK))
        y += 42

    y += 16
    elements.append(text(28, y, "Walk-forward test returns", 18, DARK, "700"))
    y += 22
    folds = report["walk_forward"]
    max_fold_return = max(abs(fold["evaluation_metrics"]["total_return"]) for fold in folds) or 1.0
    for fold in folds:
        value = fold["evaluation_metrics"]["total_return"]
        color = BLUE if value >= 0 else ORANGE
        elements.append(text(28, y + 15, f"fold {fold['fold']}", 14, DARK, "700"))
        elements.append(rect(170, y, 360, 20, LIGHT))
        elements.append(rect(170, y, max(2, bar_width(value, max_fold_return, 360)), 20, color))
        elements.append(text(548, y + 15, f"{pct(value)} | policy {fold['policy']}", 13, DARK))
        y += 42

    metadata = report.get("metadata", {})
    elements.append(text(28, y + 18, f"data_sha256: {metadata.get('data_sha256', 'n/a')}", 12, GRAY))
    write_svg(output_path, 900, y + 46, elements)


def render_dataset_smoke(smoke_path: Path, output_path: Path) -> None:
    """Render pass/fail status from dataset smoke-load JSON."""
    results = json.loads(smoke_path.read_text(encoding="utf-8"))
    passed = [result for result in results if "error" not in result]
    failed = [result for result in results if "error" in result]
    height = 108 + len(results) * 28
    elements = [
        text(28, 38, "Torch-Trade Dataset Smoke Test", 24, DARK, "700"),
        text(28, 64, f"passed {len(passed)} / {len(results)} | failed {len(failed)}", 14, GRAY),
    ]
    y = 96
    for index, result in enumerate(results, start=1):
        ok = "error" not in result
        color = GREEN if ok else RED
        status = "OK" if ok else "FAIL"
        suffix = f"rows={result.get('rows')} columns={','.join(result.get('columns', []))}" if ok else result.get("error", "error")
        elements.append(rect(28, y - 16, 58, 20, color))
        elements.append(text(38, y, status, 12, "#ffffff", "700"))
        elements.append(text(100, y, f"{index:02d}. {result['dataset_id']}", 12, DARK, "700"))
        elements.append(text(500, y, str(suffix)[:80], 12, GRAY))
        y += 28
    write_svg(output_path, 980, height, elements)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render SVG visualizations for local backtest outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Render benchmark_report.json as SVG.")
    benchmark.add_argument("--report", type=Path, default=Path("outputs/local_backtest/benchmark_report.json"))
    benchmark.add_argument("--output", type=Path, default=Path("outputs/local_backtest/benchmark_summary.svg"))
    benchmark.set_defaults(func=lambda args: render_benchmark(args.report, args.output))

    datasets = subparsers.add_parser("datasets", help="Render dataset smoke-load JSON as SVG.")
    datasets.add_argument("--smoke-results", type=Path, required=True)
    datasets.add_argument("--output", type=Path, default=Path("outputs/local_backtest/dataset_smoke.svg"))
    datasets.set_defaults(func=lambda args: render_dataset_smoke(args.smoke_results, args.output))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
