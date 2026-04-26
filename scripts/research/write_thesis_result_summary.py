#!/usr/bin/env python3
"""Generate a Markdown summary from a mergekit GA result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_text_if_exists(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _model_table(final_comparison: pd.DataFrame | None) -> list[str]:
    if final_comparison is None or final_comparison.empty:
        return ["No final comparison table was found."]

    lines = [
        "| Model | Raw weighted score |",
        "| --- | ---: |",
    ]
    for _, row in final_comparison.iterrows():
        lines.append(f"| `{row['model']}` | `{_fmt(row['weighted_score'], 10)}` |")
    return lines


def _method_summary(method_history: pd.DataFrame | None) -> list[str]:
    if method_history is None or method_history.empty:
        return ["No method-history table was found."]

    latest = method_history.sort_values("generation").groupby("merge_method").tail(1)
    lines = [
        "| Method | Final probability | Final success rate | Best score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in latest.sort_values("merge_method").iterrows():
        lines.append(
            "| `{method}` | `{prob}` | `{success}` | `{best}` |".format(
                method=row["merge_method"],
                prob=_fmt(row.get("probability_after")),
                success=_fmt(row.get("success_rate")),
                best=_fmt(row.get("best_score")),
            )
        )
    return lines


def _failure_summary(failed: pd.DataFrame | None) -> list[str]:
    if failed is None or failed.empty:
        return ["No failed genotypes were recorded."]

    counts = failed["error_type"].fillna("unknown").value_counts()
    lines = [
        "| Error type | Count |",
        "| --- | ---: |",
    ]
    for error_type, count in counts.items():
        lines.append(f"| `{error_type}` | `{count}` |")
    return lines


def generate_summary(results_dir: Path, title: str | None = None) -> str:
    results_dir = results_dir.resolve()
    stop_details = _read_json_if_exists(results_dir / "ga_stop_details.json")
    final_stop = stop_details.get("final_stop", stop_details)
    final_comparison = _read_csv_if_exists(results_dir / "ga_final_comparison.csv")
    method_history = _read_csv_if_exists(results_dir / "ga_method_history.csv")
    failed = _read_csv_if_exists(results_dir / "failed_genotypes.csv")
    summary_text = _read_text_if_exists(results_dir / "ga_summary.txt")

    heading = title or f"Thesis Result Summary: {results_dir.name}"
    lines = [
        f"# {heading}",
        "",
        "Result directory:",
        f"[{results_dir.name}]({results_dir})",
        "",
        "## Stop Details",
        "",
        f"- stop reason: `{final_stop.get('reason', 'N/A')}`",
        f"- generation: `{final_stop.get('generation', 'N/A')}`",
        f"- fevals: `{final_stop.get('fevals', 'N/A')}`",
        f"- best generation: `{final_stop.get('best_generation', 'N/A')}`",
        f"- best GA objective score: `{_fmt(final_stop.get('best_score'), 10)}`",
        f"- best score source: `{final_stop.get('best_score_source', 'N/A')}`",
        "",
    ]

    if summary_text:
        lines.extend(["## GA Summary", "", "```text", summary_text.strip(), "```", ""])

    lines.extend(
        [
            "## Raw Final Comparison",
            "",
            *_model_table(final_comparison),
            "",
            "## Operator Summary",
            "",
            *_method_summary(method_history),
            "",
            "## Failure Summary",
            "",
            *_failure_summary(failed),
            "",
            "## Interpretation Note",
            "",
            "The GA objective score and the raw final comparison score are not the "
            "same metric. The GA objective may include staged evaluation, diversity "
            "terms, penalties, or bonuses used by the optimizer. Thesis claims about "
            "model quality should therefore distinguish search-objective improvement "
            "from raw benchmark improvement on the exported final model.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary from a mergekit GA result directory."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    markdown = generate_summary(args.results_dir, title=args.title)
    if args.output:
        args.output.write_text(markdown)
        print(f"Wrote {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
