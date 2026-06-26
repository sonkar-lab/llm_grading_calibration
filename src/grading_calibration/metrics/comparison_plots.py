"""
Plotting utilities for Step 6 cross-comparison analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def plot_metrics_heatmap(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    *,
    binning_strategy: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = metrics_df[metrics_df["metric"] == metric].copy()
    if "binning_strategy" in df.columns:
        if binning_strategy is None:
            strategies = df["binning_strategy"].dropna().unique().tolist()
            if "equal_width" in strategies:
                binning_strategy = "equal_width"
            elif strategies:
                binning_strategy = str(strategies[0])
        if binning_strategy is not None:
            df = df[df["binning_strategy"].fillna(binning_strategy) == binning_strategy]
    pivot = df.pivot_table(index="model", columns="method", values="value", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax)
    title = f"{metric.upper()} by Model and Method"
    if binning_strategy is not None:
        title = f"{title} ({binning_strategy})"
    ax.set_title(title)
    plt.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"heatmap_{metric}"
    if binning_strategy is not None:
        filename += f"_{binning_strategy}"
    plt.savefig(output_path / f"{filename}.png", dpi=150)
    plt.close()


def plot_forest_plot(comparisons_df: pd.DataFrame, metric: str, output_path: Path, label_col: str) -> None:
    import matplotlib.pyplot as plt

    df = comparisons_df[comparisons_df["metric"] == metric].copy()
    if df.empty:
        return
    df = df.sort_values("diff")
    y_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.4 + 1)))
    ax.errorbar(
        df["diff"],
        y_pos,
        xerr=[df["diff"] - df["diff_ci_low"], df["diff_ci_high"] - df["diff"]],
        fmt="o",
        capsize=3,
        color="steelblue",
    )
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[label_col])
    ax.set_xlabel(f"Difference in {metric}")
    ax.set_title(f"Pairwise Comparisons: {metric.upper()}")
    plt.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f"forest_{metric}.png", dpi=150)
    plt.close()


def plot_cd_diagram(mean_ranks: Dict[str, float], cd: float, output_path: Path, title: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    if not mean_ranks:
        return
    labels = list(mean_ranks.keys())
    ranks = np.array([mean_ranks[k] for k in labels], dtype=float)
    order = np.argsort(ranks)
    labels = [labels[i] for i in order]
    ranks = ranks[order]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title(title)
    ax.set_xlabel("Mean rank (lower is better)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.errorbar(ranks, range(len(labels)), xerr=cd / 2, fmt="o", color="steelblue")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / filename, dpi=150)
    plt.close()


def plot_reliability_diagram(bin_data: Dict[str, np.ndarray], output_path: Path, title: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    mean_conf = bin_data.get("mean_conf", np.array([]))
    acc = bin_data.get("acc", np.array([]))
    if len(mean_conf) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(mean_conf, acc, marker="o", color="steelblue")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / filename, dpi=150)
    plt.close()
