from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MazeAnalyzer.analyze_linear_dynamics import fit_affine_dynamics


def assign_task_group(task):
    if task in {1, 2, 3}:
        return "task1_3"
    if task == 4:
        return "task4"
    return "other"


def add_task_group(table, task_col):
    table = table.copy()
    table["task_group"] = table[task_col].astype(int).map(assign_task_group)
    return table[table["task_group"].isin(["task1_3", "task4"])].copy()


def fit_task_group_dynamics(transition_table, n_pcs, min_samples, test_size, seed):
    rows = []
    for task_group, group in transition_table.groupby("task_group"):
        result = fit_affine_dynamics(
            group,
            n_pcs=n_pcs,
            min_samples=min_samples,
            test_size=test_size,
            seed=seed,
        )
        if result is None:
            continue
        rows.append({"task_group": task_group, **result})
    return pd.DataFrame(rows)


def summarize_pca_trajectory(step_table, n_pcs):
    agg = {
        "n_steps": ("PC1", "size"),
        "mean_hidden_norm": ("hidden_norm", "mean"),
        "mean_choice_confidence": ("choice_confidence", "mean"),
    }
    for idx in range(1, n_pcs + 1):
        agg[f"mean_PC{idx}"] = (f"PC{idx}", "mean")
        agg[f"sem_PC{idx}"] = (f"PC{idx}", lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    return (
        step_table.groupby(["task_group", "progress_bin"], as_index=False)
        .agg(**agg)
        .sort_values(["task_group", "progress_bin"])
    )


def plot_spectral_radius(summary, output_dir):
    plt.figure(figsize=(5.5, 4))
    colors = ["#4c78a8" if group == "task1_3" else "#f58518" for group in summary["task_group"]]
    plt.bar(summary["task_group"], summary["spectral_radius"], color=colors)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Spectral radius")
    plt.title("Task1-3 vs Task4 linear dynamics")
    plt.tight_layout()
    plt.savefig(output_dir / "task4_spectral_radius.png", dpi=220)
    plt.close()


def plot_fixed_points(summary, output_dir):
    if "fixed_PC2" not in summary.columns:
        return
    ok = summary[summary["fixed_point_note"].eq("ok")].copy()
    if ok.empty:
        return

    plt.figure(figsize=(5.5, 5))
    for task_group, group in ok.groupby("task_group"):
        plt.scatter(
            group["fixed_PC1"],
            group["fixed_PC2"],
            s=120,
            label=task_group,
        )
        for row in group.itertuples(index=False):
            plt.text(row.fixed_PC1, row.fixed_PC2, f"  {task_group}", va="center")
    plt.axhline(0, color="0.85", linewidth=1)
    plt.axvline(0, color="0.85", linewidth=1)
    plt.xlabel("Fixed point PC1")
    plt.ylabel("Fixed point PC2")
    plt.title("Fixed points in PC1-PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task4_fixed_points_pc1_pc2.png", dpi=220)
    plt.close()


def plot_pca_trajectory_2d(trajectory, output_dir):
    if "mean_PC2" not in trajectory.columns:
        return

    plt.figure(figsize=(6, 5))
    for task_group, group in trajectory.groupby("task_group"):
        group = group.sort_values("progress_bin")
        plt.plot(
            group["mean_PC1"],
            group["mean_PC2"],
            marker="o",
            linewidth=2,
            label=task_group,
        )
        plt.scatter(group["mean_PC1"].iloc[0], group["mean_PC2"].iloc[0], marker="s", s=70)
        plt.scatter(group["mean_PC1"].iloc[-1], group["mean_PC2"].iloc[-1], marker="x", s=90)
    plt.xlabel("Mean PC1")
    plt.ylabel("Mean PC2")
    plt.title("Task1-3 vs Task4 mean PCA trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task4_pca_trajectory_pc1_pc2.png", dpi=220)
    plt.close()


def plot_pca_trajectory_3d(trajectory, output_dir):
    if "mean_PC3" not in trajectory.columns:
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for task_group, group in trajectory.groupby("task_group"):
        group = group.sort_values("progress_bin")
        ax.plot(
            group["mean_PC1"],
            group["mean_PC2"],
            group["mean_PC3"],
            marker="o",
            linewidth=2,
            label=task_group,
        )
        ax.scatter(group["mean_PC1"].iloc[0], group["mean_PC2"].iloc[0], group["mean_PC3"].iloc[0], marker="s", s=70)
        ax.scatter(group["mean_PC1"].iloc[-1], group["mean_PC2"].iloc[-1], group["mean_PC3"].iloc[-1], marker="x", s=90)
    ax.set_xlabel("Mean PC1")
    ax.set_ylabel("Mean PC2")
    ax.set_zlabel("Mean PC3")
    ax.set_title("Task1-3 vs Task4 mean PCA trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task4_pca_trajectory_pc1_pc2_pc3.png", dpi=220)
    plt.close()


def write_readme(output_dir, step_path, transition_path, dynamics_summary):
    task_rows = []
    for row in dynamics_summary.itertuples(index=False):
        task_rows.append(
            f"- {row.task_group}: spectral_radius={row.spectral_radius:.6f}, "
            f"test_r2={row.test_r2:.6f}, attractor_like={row.attractor_like}, "
            f"fixed_point_note={row.fixed_point_note}"
        )

    text = f"""# Task4 comparison

This analysis compares Task1-3 against Task4 using the same hidden PCA space.

Input step table:
{step_path}

Input transition table:
{transition_path}

Linear dynamics model:
x_(t+1) = A x_t + b

Task grouping:
- task1_3: task in 1, 2, 3
- task4: task == 4

Key fitted summaries:
{chr(10).join(task_rows)}

Main outputs:
- task4_linear_dynamics_summary.csv
- task4_pca_trajectory_summary.csv
- task4_spectral_radius.png
- task4_fixed_points_pc1_pc2.png
- task4_pca_trajectory_pc1_pc2.png
- task4_pca_trajectory_pc1_pc2_pc3.png
"""
    (output_dir / "README_task4_comparison.md").write_text(text, encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Compare Task1-3 and Task4 hidden dynamics.")
    parser.add_argument(
        "--step-path",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/hidden_dynamics/dynamics_step_table.csv",
    )
    parser.add_argument(
        "--transition-path",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/hidden_dynamics/hidden_transition_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/task4_comparison",
    )
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    step_path = Path(args.step_path)
    transition_path = Path(args.transition_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_table = add_task_group(pd.read_csv(step_path), "task")
    transition_table = add_task_group(pd.read_csv(transition_path), "task_t")

    dynamics_summary = fit_task_group_dynamics(
        transition_table,
        n_pcs=args.n_pcs,
        min_samples=args.min_samples,
        test_size=args.test_size,
        seed=args.seed,
    )
    trajectory_summary = summarize_pca_trajectory(step_table, args.n_pcs)

    dynamics_summary.to_csv(output_dir / "task4_linear_dynamics_summary.csv", index=False, encoding="utf-8-sig")
    trajectory_summary.to_csv(output_dir / "task4_pca_trajectory_summary.csv", index=False, encoding="utf-8-sig")

    plot_spectral_radius(dynamics_summary, output_dir)
    plot_fixed_points(dynamics_summary, output_dir)
    plot_pca_trajectory_2d(trajectory_summary, output_dir)
    plot_pca_trajectory_3d(trajectory_summary, output_dir)
    write_readme(output_dir, step_path, transition_path, dynamics_summary)

    print(f"Task groups: {', '.join(dynamics_summary['task_group'])}")
    for row in dynamics_summary.itertuples(index=False):
        print(
            f"{row.task_group}: spectral_radius={row.spectral_radius:.4f}, "
            f"test_r2={row.test_r2:.4f}, fixed=({row.fixed_PC1:.3f}, {row.fixed_PC2:.3f}, {row.fixed_PC3:.3f})"
        )
    print(f"Wrote Task4 comparison to {output_dir}")


if __name__ == "__main__":
    main()
