"""
Basic descriptive statistics for maze behavior data.

Recommended command:
    C:\\Users\\lirui\\anaconda3\\Scripts\\conda.exe run -n rnn python analysis\\basic_behavior_stats.py

Optional: keep only trials that pass core decode checks:
    C:\\Users\\lirui\\anaconda3\\Scripts\\conda.exe run -n rnn python analysis\\basic_behavior_stats.py --valid-only

Outputs:
    outputs/basic_behavior_stats/trial_metrics.csv
    outputs/basic_behavior_stats/summary_by_subject.csv
    outputs/basic_behavior_stats/summary_by_subject_task.csv
    outputs/basic_behavior_stats/summary_overall_by_task.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


JSON_COLUMNS = ["short_path", "action", "hits", "rt", "true_path"]

TRIAL_ID_COLUMNS = [
    "subid",
    "batch",
    "maze",
    "day",
    "task",
    "block",
    "trial",
    "replan",
    "start",
    "goal",
]

METRIC_COLUMNS = [
    "n_action",
    "n_hit",
    "hit_rate",
    "shortest_steps",
    "actual_steps",
    "successful_moves",
    "excess_steps",
    "path_efficiency",
    "reached_goal",
    "is_optimal_by_length",
    "is_clean_optimal",
    "is_exact_short_path",
    "total_rt",
    "mean_rt",
    "median_rt",
    "first_rt",
    "mean_rt_after_first",
]

CORE_DIAGNOSTIC_COLUMNS = [
    "true_path_end_ok",
    "hit_consistent",
    "short_path_valid_in_maze",
    "short_path_is_shortest",
]


def parse_json_list(value) -> list:
    """Convert JSON arrays stored as CSV strings into Python lists."""
    if pd.isna(value) or value == "":
        return []
    return json.loads(value)


def safe_len(value) -> int:
    return len(value) if isinstance(value, list) else 0


def count_hits(hits: list) -> int:
    return int(sum(bool(x) for x in hits)) if isinstance(hits, list) else 0


def count_successful_moves(path: list) -> int:
    """Wall hits stay at the same state, so only changed states are successful moves."""
    if not isinstance(path, list) or len(path) < 2:
        return 0
    return int(sum(a != b for a, b in zip(path[:-1], path[1:])))


def last_item(value: list):
    return value[-1] if isinstance(value, list) and len(value) > 0 else np.nan


def list_sum(value: list) -> float:
    return float(np.sum(value)) if isinstance(value, list) and len(value) > 0 else np.nan


def list_mean(value: list) -> float:
    return float(np.mean(value)) if isinstance(value, list) and len(value) > 0 else np.nan


def list_median(value: list) -> float:
    return float(np.median(value)) if isinstance(value, list) and len(value) > 0 else np.nan


def first_item(value: list) -> float:
    return float(value[0]) if isinstance(value, list) and len(value) > 0 else np.nan


def mean_after_first(value: list) -> float:
    return float(np.mean(value[1:])) if isinstance(value, list) and len(value) > 1 else np.nan


def read_trial_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    for col in TRIAL_ID_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in JSON_COLUMNS:
        df[col] = df[col].map(parse_json_list)
    return df


def add_trial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = df.copy()

    metrics["n_action"] = metrics["action"].map(safe_len)
    metrics["n_hit"] = metrics["hits"].map(count_hits)
    metrics["hit_rate"] = metrics["n_hit"].div(metrics["n_action"].replace(0, np.nan))

    metrics["shortest_steps"] = metrics["short_path"].map(safe_len).sub(1).clip(lower=0)
    metrics["actual_steps"] = metrics["n_action"]
    metrics["successful_moves"] = metrics["true_path"].map(count_successful_moves)
    metrics["excess_steps"] = metrics["actual_steps"] - metrics["shortest_steps"]
    metrics["path_efficiency"] = metrics["shortest_steps"].div(
        metrics["actual_steps"].replace(0, np.nan)
    )

    metrics["final_state"] = metrics["true_path"].map(last_item)
    metrics["reached_goal"] = metrics["final_state"].eq(metrics["goal"]).astype(int)
    metrics["is_optimal_by_length"] = (
        metrics["reached_goal"].eq(1)
        & metrics["actual_steps"].eq(metrics["shortest_steps"])
    ).astype(int)
    metrics["is_clean_optimal"] = (
        metrics["is_optimal_by_length"].eq(1) & metrics["n_hit"].eq(0)
    ).astype(int)
    metrics["is_exact_short_path"] = [
        int(true_path == short_path)
        for true_path, short_path in zip(metrics["true_path"], metrics["short_path"])
    ]

    metrics["total_rt"] = metrics["rt"].map(list_sum)
    metrics["mean_rt"] = metrics["rt"].map(list_mean)
    metrics["median_rt"] = metrics["rt"].map(list_median)
    metrics["first_rt"] = metrics["rt"].map(first_item)
    metrics["mean_rt_after_first"] = metrics["rt"].map(mean_after_first)

    return metrics


def keep_valid_trials(metrics: pd.DataFrame, diagnostics_path: Path) -> pd.DataFrame:
    diagnostics = pd.read_csv(diagnostics_path)
    missing_cols = [col for col in CORE_DIAGNOSTIC_COLUMNS if col not in diagnostics.columns]
    if missing_cols:
        raise ValueError(f"Diagnostics file is missing columns: {missing_cols}")
    if len(diagnostics) != len(metrics):
        raise ValueError(
            f"Diagnostics rows ({len(diagnostics)}) do not match trial rows ({len(metrics)})."
        )

    valid_mask = diagnostics[CORE_DIAGNOSTIC_COLUMNS].eq(1).all(axis=1)
    return metrics.loc[valid_mask.to_numpy()].copy()


def summarize(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        metrics.groupby(group_cols, dropna=False)
        .agg(
            n_trials=("trial", "size"),
            mean_n_action=("n_action", "mean"),
            mean_n_hit=("n_hit", "mean"),
            mean_hit_rate=("hit_rate", "mean"),
            mean_shortest_steps=("shortest_steps", "mean"),
            mean_excess_steps=("excess_steps", "mean"),
            mean_path_efficiency=("path_efficiency", "mean"),
            reached_goal_rate=("reached_goal", "mean"),
            optimal_by_length_rate=("is_optimal_by_length", "mean"),
            clean_optimal_rate=("is_clean_optimal", "mean"),
            exact_short_path_rate=("is_exact_short_path", "mean"),
            mean_total_rt=("total_rt", "mean"),
            mean_rt=("mean_rt", "mean"),
            mean_first_rt=("first_rt", "mean"),
            mean_rt_after_first=("mean_rt_after_first", "mean"),
        )
        .reset_index()
    )
    return summary.round(6)


def save_outputs(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics[TRIAL_ID_COLUMNS + METRIC_COLUMNS].round(6).to_csv(
        output_dir / "trial_metrics.csv", index=False, encoding="utf-8-sig"
    )

    summaries = {
        "summary_by_subject.csv": summarize(metrics, ["subid", "batch", "maze"]),
        "summary_by_subject_task.csv": summarize(
            metrics, ["subid", "batch", "maze", "day", "task", "block", "replan"]
        ),
        "summary_overall_by_task.csv": summarize(
            metrics, ["batch", "maze", "day", "task", "block", "replan"]
        ),
    }

    for filename, summary in summaries.items():
        summary.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute basic maze behavior metrics.")
    parser.add_argument(
        "--input",
        default=r"C:\Users\lirui\Desktop\RNN\mazeTask\maze_healthy_batch123\trial_level.csv",
        help="Path to trial_level.csv.",
    )
    parser.add_argument(
        "--diagnostics",
        default=r"C:\Users\lirui\Desktop\RNN\mazeTask\maze_healthy_batch123\\trial_level_decode_diagnostics.csv",
        help="Path to trial_level_decode_diagnostics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/basic_behavior_stats",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only analyze trials passing core diagnostic checks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    raw = read_trial_data(Path(args.input))
    metrics = add_trial_metrics(raw)

    if args.valid_only:
        before = len(metrics)
        metrics = keep_valid_trials(metrics, Path(args.diagnostics))
        print(f"Kept {len(metrics)} / {before} trials after diagnostic filtering.")

    output_dir = Path(args.output_dir)
    save_outputs(metrics, output_dir)

    print(f"Read {len(raw)} trials.")
    print(f"Analyzed {len(metrics)} trials.")
    print(f"Wrote {output_dir / 'trial_metrics.csv'}")
    print(f"Wrote {output_dir / 'summary_by_subject.csv'}")
    print(f"Wrote {output_dir / 'summary_by_subject_task.csv'}")
    print(f"Wrote {output_dir / 'summary_overall_by_task.csv'}")


if __name__ == "__main__":
    main()
