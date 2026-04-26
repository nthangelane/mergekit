#!/usr/bin/env python3
"""Generate thesis-facing plots from mergekit GA result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.exists():
        print(f"Skipping {name}: file not found")
        return None
    return pd.read_csv(path)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_method_history(results_dir: Path) -> Path | None:
    data = _read_csv(results_dir, "ga_method_history.csv")
    if data is None or data.empty:
        return None

    methods = sorted(data["merge_method"].dropna().unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for method in methods:
        subset = data[data["merge_method"] == method].sort_values("generation")
        axes[0].plot(
            subset["generation"],
            subset["probability_after"],
            marker="o",
            linewidth=2,
            label=method,
        )
        axes[1].plot(
            subset["generation"],
            subset["success_rate"],
            marker="o",
            linewidth=2,
            label=method,
        )

    axes[0].set_title("Adaptive Operator Probability")
    axes[0].set_ylabel("Probability after generation")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Operator Success Rate")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    output = results_dir / "ga_method_history_plot.png"
    _save(fig, output)
    return output


def plot_candidate_scores(results_dir: Path) -> Path | None:
    data = _read_csv(results_dir, "ga_candidate_history.csv")
    if data is None or data.empty:
        return None

    scored = data.dropna(subset=["score"]).copy()
    if scored.empty:
        print("Skipping candidate score plot: no scored candidates")
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    methods = sorted(scored["merge_method"].dropna().unique())
    for method in methods:
        subset = scored[scored["merge_method"] == method]
        ax.scatter(
            subset["generation"],
            subset["score"],
            label=method,
            alpha=0.75,
            s=48,
        )

    best_by_generation = scored.groupby("generation")["score"].max()
    ax.plot(
        best_by_generation.index,
        best_by_generation.values,
        color="black",
        linewidth=2,
        marker="o",
        label="generation best",
    )
    ax.set_title("Candidate Scores By Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("GA objective score")
    ax.grid(alpha=0.3)
    ax.legend()

    output = results_dir / "ga_candidate_scores_plot.png"
    _save(fig, output)
    return output


def plot_failures(results_dir: Path) -> Path | None:
    data = _read_csv(results_dir, "failed_genotypes.csv")
    if data is None or data.empty:
        return None

    by_generation = data.groupby("generation").size()
    by_type = data["error_type"].fillna("unknown").value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(by_generation.index.astype(str), by_generation.values, color="#7a3b2e")
    axes[0].set_title("Failed Candidates By Generation")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Failure count")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(by_type.index.astype(str), by_type.values, color="#345c72")
    axes[1].set_title("Failure Type Breakdown")
    axes[1].set_xlabel("Error type")
    axes[1].set_ylabel("Failure count")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    output = results_dir / "ga_failure_breakdown_plot.png"
    _save(fig, output)
    return output


def plot_raw_comparison(results_dir: Path) -> Path | None:
    data = _read_csv(results_dir, "ga_final_comparison.csv")
    if data is None or data.empty:
        data = _read_csv(results_dir, "baseline_results.csv")
    if data is None or data.empty or "weighted_score" not in data:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = data["model"].astype(str)
    ax.bar(labels, data["weighted_score"], color=["#345c72", "#6d8c4d", "#7a3b2e"])
    ax.set_title("Raw Weighted Score Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Raw weighted score")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    output = results_dir / "ga_raw_score_comparison_plot.png"
    _save(fig, output)
    return output


def generate_plots(results_dir: Path) -> list[Path]:
    outputs = [
        plot_method_history(results_dir),
        plot_candidate_scores(results_dir),
        plot_failures(results_dir),
        plot_raw_comparison(results_dir),
    ]
    return [path for path in outputs if path is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis plots from a mergekit GA result directory."
    )
    parser.add_argument("results_dir", type=Path, help="Directory containing GA CSVs")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Result directory not found: {results_dir}")

    outputs = generate_plots(results_dir)
    if not outputs:
        raise SystemExit("No plots were generated")


if __name__ == "__main__":
    main()
