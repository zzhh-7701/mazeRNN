from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


LABEL_KEYS = ["state", "goal", "prev_action", "task", "replan", "target"]


def load_hidden_arrays(hidden_path: Path):
    data = np.load(hidden_path)
    arrays = {key: data[key] for key in data.files}
    if "sample_id" not in arrays:
        arrays["sample_id"] = make_fallback_sample_id(arrays)
    return arrays


def make_fallback_sample_id(arrays):
    frame = pd.DataFrame(
        {
            "subid": arrays["subid"],
            "trial": arrays["trial"],
            "task": arrays["task"],
            "replan": arrays["replan"],
        }
    )
    return pd.factorize(list(map(tuple, frame.to_numpy())))[0]


def fit_pca(hidden, n_components):
    n_components = min(n_components, hidden.shape[0], hidden.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pc = pca.fit_transform(hidden - hidden.mean(axis=0, keepdims=True))
    return pca, pc


def build_step_table(arrays, pc):
    table = pd.DataFrame(
        {
            "sample_id": arrays["sample_id"],
            "subid": arrays["subid"],
            "trial": arrays["trial"],
            "step": arrays["step"],
        }
    )
    for key in LABEL_KEYS:
        table[key] = arrays[key]
    for idx in range(pc.shape[1]):
        table[f"PC{idx + 1}"] = pc[:, idx]
    table["hidden_norm"] = np.linalg.norm(arrays["hidden"], axis=1)
    logits = arrays["logits"]
    table["choice_confidence"] = logits.max(axis=1) - np.partition(logits, -2, axis=1)[:, -2]

    max_step = table.groupby("sample_id")["step"].transform("max").replace(0, 1)
    table["progress"] = table["step"] / max_step
    table["progress_bin"] = np.minimum((table["progress"] * 10).astype(int), 9)
    return table


def build_transition_table(arrays, pc):
    step_table = build_step_table(arrays, pc)
    hidden = arrays["hidden"]

    transitions = []
    order = np.argsort(
        np.rec.fromarrays(
            [arrays["sample_id"], arrays["step"]],
            names=["sample_id", "step"],
        )
    )

    for left, right in zip(order[:-1], order[1:]):
        if arrays["sample_id"][left] != arrays["sample_id"][right]:
            continue
        if arrays["step"][right] != arrays["step"][left] + 1:
            continue

        delta_h = hidden[right] - hidden[left]
        row = {
            "sample_id": int(arrays["sample_id"][left]),
            "subid": int(arrays["subid"][left]),
            "trial": int(arrays["trial"][left]),
            "from_step": int(arrays["step"][left]),
            "to_step": int(arrays["step"][right]),
            "delta_norm": float(np.linalg.norm(delta_h)),
            "hidden_norm_t": float(np.linalg.norm(hidden[left])),
            "hidden_norm_t1": float(np.linalg.norm(hidden[right])),
        }
        for key in LABEL_KEYS:
            row[f"{key}_t"] = int(arrays[key][left])
            row[f"{key}_t1"] = int(arrays[key][right])
        for idx in range(pc.shape[1]):
            row[f"PC{idx + 1}_t"] = float(pc[left, idx])
            row[f"PC{idx + 1}_t1"] = float(pc[right, idx])
            row[f"dPC{idx + 1}"] = float(pc[right, idx] - pc[left, idx])
        transitions.append(row)

    transition_table = pd.DataFrame(transitions)
    if transition_table.empty:
        return step_table, transition_table

    max_step = transition_table.groupby("sample_id")["from_step"].transform("max").replace(0, 1)
    transition_table["progress"] = transition_table["from_step"] / max_step
    transition_table["progress_bin"] = np.minimum((transition_table["progress"] * 10).astype(int), 9)
    return step_table, transition_table


def summarize_updates(transition_table, pc_num):
    if transition_table.empty:
        return pd.DataFrame()

    group_keys = ["task_t", "replan_t", "target_t", "progress_bin"]
    value_cols = ["delta_norm"] + [f"dPC{i}" for i in range(1, pc_num + 1)]
    summary = (
        transition_table.groupby(group_keys, as_index=False)
        .agg(
            n_transitions=("delta_norm", "size"),
            mean_delta_norm=("delta_norm", "mean"),
            std_delta_norm=("delta_norm", "std"),
            **{f"mean_{col}": (col, "mean") for col in value_cols[1:]},
        )
        .sort_values(group_keys)
    )
    return summary


def summarize_trajectories(step_table, pc_num):
    group_keys = ["task", "replan", "progress_bin"]
    agg = {
        "n_steps": ("PC1", "size"),
        "mean_hidden_norm": ("hidden_norm", "mean"),
        "mean_choice_confidence": ("choice_confidence", "mean"),
    }
    for idx in range(1, pc_num + 1):
        agg[f"mean_PC{idx}"] = (f"PC{idx}", "mean")
    return step_table.groupby(group_keys, as_index=False).agg(**agg).sort_values(group_keys)


def save_pca_summary(pca, output_dir):
    summary = pd.DataFrame(
        {
            "pc": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "variance_ratio": pca.explained_variance_ratio_,
            "variance_cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    summary.to_csv(output_dir / "dynamics_pca_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def subsample_index(n_rows, max_points, seed):
    if n_rows <= max_points:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))


def plot_pca_scatter(step_table, output_dir, max_points, seed):
    if "PC2" not in step_table:
        return
    index = subsample_index(len(step_table), max_points, seed)
    plot_data = step_table.iloc[index]

    plt.figure(figsize=(7, 5))
    for (task, replan), group in plot_data.groupby(["task", "replan"]):
        plt.scatter(group["PC1"], group["PC2"], s=8, alpha=0.35, label=f"T{task} R{replan}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Hidden states in PCA space")
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_scatter_task_replan.png", dpi=200)
    plt.close()


def plot_mean_trajectories(trajectory_summary, output_dir):
    if trajectory_summary.empty or "mean_PC2" not in trajectory_summary:
        return

    plt.figure(figsize=(7, 5))
    for (task, replan), group in trajectory_summary.groupby(["task", "replan"]):
        group = group.sort_values("progress_bin")
        plt.plot(group["mean_PC1"], group["mean_PC2"], marker="o", linewidth=1.5, label=f"T{task} R{replan}")
    plt.xlabel("Mean PC1")
    plt.ylabel("Mean PC2")
    plt.title("Mean hidden trajectory by task/replan")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_pca_trajectory_task_replan.png", dpi=200)
    plt.close()


def plot_delta_norm(transition_table, output_dir):
    if transition_table.empty:
        return

    summary = (
        transition_table.groupby(["task_t", "replan_t", "progress_bin"], as_index=False)
        .agg(mean_delta_norm=("delta_norm", "mean"))
    )

    plt.figure(figsize=(7, 5))
    for (task, replan), group in summary.groupby(["task_t", "replan_t"]):
        group = group.sort_values("progress_bin")
        plt.plot(group["progress_bin"], group["mean_delta_norm"], marker="o", label=f"T{task} R{replan}")
    plt.xlabel("Progress bin")
    plt.ylabel("Mean ||delta h||")
    plt.title("Hidden update magnitude")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "delta_norm_by_progress_task_replan.png", dpi=200)
    plt.close()


def write_readme(output_dir, hidden_path, pca_summary, transition_table):
    text = f"""# Hidden dynamics analysis

Input hidden file:
{hidden_path}

This analysis defines the neural dynamics object as adjacent hidden-state pairs
within the same maze sequence:

h_t, h_(t+1), and delta_h_t = h_(t+1) - h_t.

Main outputs:
- dynamics_step_table.csv: every valid step with PCA coordinates and labels.
- hidden_transition_table.csv: adjacent transitions with delta_h and dPC values.
- hidden_update_summary.csv: mean update direction/magnitude by task, replan, action, and progress bin.
- pca_trajectory_summary.csv: mean PCA trajectory by task/replan/progress bin.
- dynamics_pca_summary.csv: PCA explained variance.
- pca_scatter_task_replan.png: sampled hidden states in PC1/PC2.
- mean_pca_trajectory_task_replan.png: mean trajectories in PC1/PC2.
- delta_norm_by_progress_task_replan.png: hidden update magnitude over trial progress.

N transitions: {len(transition_table)}
PC1 cumulative variance: {pca_summary['variance_cumulative_ratio'].iloc[0]:.6f}
PC1-3 cumulative variance: {pca_summary['variance_cumulative_ratio'].iloc[min(2, len(pca_summary) - 1)]:.6f}
"""
    (output_dir / "README_hidden_dynamics.md").write_text(text, encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Analyze hidden-state dynamics for a trained maze RNN.")
    parser.add_argument(
        "--hidden-path",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/hidden_analysis/hidden_states.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/hidden_dynamics",
    )
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--max-plot-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    hidden_path = Path(args.hidden_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_hidden_arrays(hidden_path)
    pca, pc = fit_pca(arrays["hidden"], args.n_pcs)
    pc_num = pc.shape[1]

    step_table, transition_table = build_transition_table(arrays, pc)
    pca_summary = save_pca_summary(pca, output_dir)
    trajectory_summary = summarize_trajectories(step_table, pc_num)
    update_summary = summarize_updates(transition_table, pc_num)

    step_table.to_csv(output_dir / "dynamics_step_table.csv", index=False, encoding="utf-8-sig")
    transition_table.to_csv(output_dir / "hidden_transition_table.csv", index=False, encoding="utf-8-sig")
    trajectory_summary.to_csv(output_dir / "pca_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    update_summary.to_csv(output_dir / "hidden_update_summary.csv", index=False, encoding="utf-8-sig")

    plot_pca_scatter(step_table, output_dir, args.max_plot_points, args.seed)
    plot_mean_trajectories(trajectory_summary, output_dir)
    plot_delta_norm(transition_table, output_dir)
    write_readme(output_dir, hidden_path, pca_summary, transition_table)

    print(f"Steps: {len(step_table)}")
    print(f"Transitions: {len(transition_table)}")
    print(f"Hidden dim: {arrays['hidden'].shape[1]}")
    print(f"PC1-3 cumulative variance: {pca_summary['variance_cumulative_ratio'].iloc[min(2, len(pca_summary) - 1)]:.4f}")
    print(f"Wrote hidden dynamics analysis to {output_dir}")


if __name__ == "__main__":
    main()
