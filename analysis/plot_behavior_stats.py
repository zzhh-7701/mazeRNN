"""
Plot basic maze behavior statistics.

Recommended command:
    C:\\Users\\lirui\\anaconda3\\Scripts\\conda.exe run -n rnn python analysis\\plot_behavior_stats.py

Input:
    outputs/basic_behavior_stats/trial_metrics.csv

Outputs:
    outputs/basic_behavior_stats/figures/learning_curve.png
    outputs/basic_behavior_stats/figures/hit_count.png
    outputs/basic_behavior_stats/figures/optimal_rate.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = "outputs/basic_behavior_stats/trial_metrics.csv"
DEFAULT_OUTPUT_DIR = "outputs/basic_behavior_stats/figures"

TASK_LABELS = {
    1: "Task 1",
    2: "Task 2",
    3: "Task 3",
    4: "Task 4",
}


def setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Arial",
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        },
    )


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "subid",
        "batch",
        "maze",
        "day",
        "task",
        "block",
        "trial",
        "replan",
        "n_hit",
        "clean_optimal_rate",
        "is_clean_optimal",
        "path_efficiency",
        "mean_rt",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["task_label"] = df["task"].map(TASK_LABELS).fillna("Task " + df["task"].astype(str))
    return df


def keep_plan_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the plan phase for the current four-task behavior analysis."""
    if "replan" not in df.columns:
        print("No replan column found; treating all rows as plan trials.")
        return df.copy()

    plan_df = df[df["replan"].fillna(0).eq(0)].copy()
    n_removed = len(df) - len(plan_df)
    if n_removed > 0:
        print(f"Removed {n_removed} non-plan trials where replan != 0.")
    return plan_df


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def subject_block_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average trial metrics within subject x block before plotting group trends."""
    return (
        df.groupby(["subid", "day", "task", "block"], dropna=False)
        .agg(
            mean_n_hit=("n_hit", "mean"),
            clean_optimal_rate=("is_clean_optimal", "mean"),
            mean_path_efficiency=("path_efficiency", "mean"),
            mean_rt=("mean_rt", "mean"),
        )
        .reset_index()
    )


def plot_learning_curve(df: pd.DataFrame, output_dir: Path) -> None:
    block_df = subject_block_summary(df)
    block_df = block_df.dropna(subset=["block"])
    block_df["block_label"] = block_df["block"].astype(int).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    sns.lineplot(
        data=block_df,
        x="block",
        y="mean_n_hit",
        marker="o",
        errorbar="se",
        color="#2F6F9F",
        ax=axes[0],
    )
    axes[0].set_title("Learning Curve: Hit Count")
    axes[0].set_xlabel("Block")
    axes[0].set_ylabel("Mean hit count")

    sns.lineplot(
        data=block_df,
        x="block",
        y="clean_optimal_rate",
        marker="o",
        errorbar="se",
        color="#3C8D57",
        ax=axes[1],
    )
    axes[1].set_title("Learning Curve: Optimal Rate")
    axes[1].set_xlabel("Block")
    axes[1].set_ylabel("Clean optimal rate")
    axes[1].set_ylim(0, 1)

    save_figure(fig, output_dir / "learning_curve.png")


def plot_hit_count(df: pd.DataFrame, output_dir: Path) -> None:
    task_df = (
        df.groupby(["subid", "task", "task_label"], dropna=False)
        .agg(mean_n_hit=("n_hit", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=task_df,
        x="task_label",
        y="mean_n_hit",
        errorbar="se",
        color="#4C78A8",
        ax=ax,
    )
    sns.stripplot(
        data=task_df,
        x="task_label",
        y="mean_n_hit",
        color="black",
        alpha=0.25,
        size=2.5,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("Hit Count by Task")
    ax.set_xlabel("Task")
    ax.set_ylabel("Mean hit count per trial")
    save_figure(fig, output_dir / "hit_count.png")


def plot_optimal_rate(df: pd.DataFrame, output_dir: Path) -> None:
    task_df = (
        df.groupby(["subid", "task", "task_label"], dropna=False)
        .agg(clean_optimal_rate=("is_clean_optimal", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=task_df,
        x="task_label",
        y="clean_optimal_rate",
        errorbar="se",
        color="#59A14F",
        ax=ax,
    )
    sns.stripplot(
        data=task_df,
        x="task_label",
        y="clean_optimal_rate",
        color="black",
        alpha=0.25,
        size=2.5,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("Optimal Rate by Task")
    ax.set_xlabel("Task")
    ax.set_ylabel("Clean optimal rate")
    ax.set_ylim(0, 1)
    save_figure(fig, output_dir / "optimal_rate.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot maze behavior statistics.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to trial_metrics.csv.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Figure output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_style()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    raw_df = load_metrics(input_path)
    df = keep_plan_trials(raw_df)
    plot_learning_curve(df, output_dir)
    plot_hit_count(df, output_dir)
    plot_optimal_rate(df, output_dir)

    print(f"Read {len(raw_df)} trials from {input_path}")
    print(f"Plotted {len(df)} plan trials.")
    print(f"Wrote {output_dir / 'learning_curve.png'}")
    print(f"Wrote {output_dir / 'hit_count.png'}")
    print(f"Wrote {output_dir / 'optimal_rate.png'}")


if __name__ == "__main__":
    main()
