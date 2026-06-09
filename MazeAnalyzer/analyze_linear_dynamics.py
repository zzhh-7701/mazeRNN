from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_GROUPS = [
    "global",
    "task_t",
    "replan_t",
    "target_t",
    "task_t,replan_t",
    "task_t,replan_t,target_t",
]


def parse_group_specs(value: str):
    return [item.strip() for item in value.split(";") if item.strip()]


def pc_columns(table, suffix):
    cols = []
    idx = 1
    while f"PC{idx}_{suffix}" in table.columns:
        cols.append(f"PC{idx}_{suffix}")
        idx += 1
    return cols


def group_iterator(table, group_spec):
    if group_spec == "global":
        yield {"group_type": "global", "group_label": "all"}, table
        return

    keys = [key.strip() for key in group_spec.split(",") if key.strip()]
    groupby_keys = keys[0] if len(keys) == 1 else keys
    for values, group in table.groupby(groupby_keys):
        if not isinstance(values, tuple):
            values = (values,)
        label = ",".join(f"{key}={value}" for key, value in zip(keys, values))
        yield {"group_type": group_spec, "group_label": label}, group


def fit_affine_dynamics(group, n_pcs, min_samples, test_size, seed):
    pc_t_cols = [f"PC{i}_t" for i in range(1, n_pcs + 1)]
    pc_t1_cols = [f"PC{i}_t1" for i in range(1, n_pcs + 1)]
    needed = pc_t_cols + pc_t1_cols
    group = group.dropna(subset=needed)
    if len(group) < min_samples:
        return None

    x = group[pc_t_cols].to_numpy(dtype=float)
    y = group[pc_t1_cols].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_z = x_scaler.fit_transform(x_train)
    y_train_z = y_scaler.fit_transform(y_train)
    x_test_z = x_scaler.transform(x_test)

    model = RidgeCV(alphas=np.logspace(-6, 3, 10))
    model.fit(x_train_z, y_train_z)
    y_pred_z = model.predict(x_test_z)
    y_pred = y_scaler.inverse_transform(y_pred_z)

    # Convert standardized coefficients back into original PC coordinates:
    # y = A x + b
    coef_z = model.coef_
    a = (y_scaler.scale_[:, None] * coef_z) / x_scaler.scale_[None, :]
    b = y_scaler.mean_ - a @ x_scaler.mean_
    y_hat = x @ a.T + b

    fixed_point, fixed_point_note = solve_fixed_point(a, b)
    eigvals = np.linalg.eigvals(a)
    spectral_radius = float(np.max(np.abs(eigvals)))
    attractor_like = bool(np.isfinite(spectral_radius) and spectral_radius < 1.0)

    result = {
        "n_samples": int(len(group)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "alpha": float(model.alpha_),
        "train_r2": float(r2_score(y, y_hat, multioutput="variance_weighted")),
        "test_r2": float(r2_score(y_test, y_pred, multioutput="variance_weighted")),
        "spectral_radius": spectral_radius,
        "attractor_like": attractor_like,
        "fixed_point_note": fixed_point_note,
    }
    for idx, value in enumerate(b, start=1):
        result[f"b_PC{idx}"] = float(value)
    for idx, value in enumerate(fixed_point, start=1):
        result[f"fixed_PC{idx}"] = float(value)
    for row_idx in range(n_pcs):
        for col_idx in range(n_pcs):
            result[f"A_PC{row_idx + 1}_from_PC{col_idx + 1}"] = float(a[row_idx, col_idx])
    for idx, value in enumerate(eigvals, start=1):
        result[f"eig{idx}_real"] = float(np.real(value))
        result[f"eig{idx}_imag"] = float(np.imag(value))
        result[f"eig{idx}_abs"] = float(np.abs(value))

    return result


def solve_fixed_point(a, b):
    n_dim = a.shape[0]
    system = np.eye(n_dim) - a
    try:
        condition_number = np.linalg.cond(system)
        if not np.isfinite(condition_number) or condition_number > 1e8:
            return np.full(n_dim, np.nan), f"ill_conditioned_cond={condition_number:.3g}"
        fixed_point = np.linalg.solve(system, b)
    except np.linalg.LinAlgError:
        return np.full(n_dim, np.nan), "singular"
    return fixed_point, "ok"


def fit_all_groups(table, n_pcs, group_specs, min_samples, test_size, seed):
    rows = []
    for group_spec in group_specs:
        for meta, group in group_iterator(table, group_spec):
            result = fit_affine_dynamics(group, n_pcs, min_samples, test_size, seed)
            if result is None:
                continue
            rows.append({**meta, **result})
    return pd.DataFrame(rows)


def plot_fixed_points(results, output_dir):
    if results.empty or "fixed_PC2" not in results.columns:
        return
    ok = results[results["fixed_point_note"].eq("ok")].copy()
    if ok.empty:
        return

    plt.figure(figsize=(7, 5))
    global_row = ok[ok["group_type"].eq("global")]
    non_global = ok[~ok["group_type"].eq("global")]

    plt.scatter(
        non_global["fixed_PC1"],
        non_global["fixed_PC2"],
        c=non_global["spectral_radius"],
        cmap="viridis",
        s=45,
        alpha=0.75,
    )
    if not global_row.empty:
        plt.scatter(
            global_row["fixed_PC1"],
            global_row["fixed_PC2"],
            marker="x",
            s=100,
            color="red",
            label="global",
        )
        plt.legend()
    plt.colorbar(label="spectral radius")
    plt.xlabel("Fixed point PC1")
    plt.ylabel("Fixed point PC2")
    plt.title("Linear-dynamics fixed points")
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_points_pc1_pc2.png", dpi=200)
    plt.close()


def plot_spectral_radius(results, output_dir):
    if results.empty:
        return
    plot_data = results.sort_values(["group_type", "spectral_radius"]).copy()
    labels = plot_data["group_type"] + " | " + plot_data["group_label"]
    height = max(5, min(18, len(plot_data) * 0.23))

    plt.figure(figsize=(9, height))
    colors = np.where(plot_data["attractor_like"], "#2a9d8f", "#e76f51")
    plt.barh(np.arange(len(plot_data)), plot_data["spectral_radius"], color=colors)
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.yticks(np.arange(len(plot_data)), labels, fontsize=7)
    plt.xlabel("Spectral radius of A")
    plt.title("Attractor criterion for fitted linear dynamics")
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_radius_by_condition.png", dpi=200)
    plt.close()


def write_readme(output_dir, transition_path, results):
    global_result = results[results["group_type"].eq("global")]
    if not global_result.empty:
        global_text = (
            f"Global test R2: {global_result['test_r2'].iloc[0]:.6f}\n"
            f"Global spectral radius: {global_result['spectral_radius'].iloc[0]:.6f}\n"
            f"Global attractor-like: {global_result['attractor_like'].iloc[0]}\n"
        )
    else:
        global_text = "Global fit was not available.\n"

    text = f"""# Linear hidden dynamics

Input transition table:
{transition_path}

The fitted model is an affine linear dynamical system in PCA coordinates:

x_(t+1) = A x_t + b

For each fitted condition, the script computes:
- train/test R2 for predicting x_(t+1)
- A and b coefficients
- eigenvalues of A
- spectral radius max(abs(eigenvalues))
- fixed point x* = (I - A)^(-1)b when numerically stable
- attractor-like flag, defined as spectral radius < 1

{global_text}
Main outputs:
- linear_dynamics_summary.csv
- fixed_points_pc1_pc2.png
- spectral_radius_by_condition.png
"""
    (output_dir / "README_linear_dynamics.md").write_text(text, encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fit linear dynamics and fixed points in hidden PCA space.")
    parser.add_argument(
        "--transition-path",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/hidden_dynamics/hidden_transition_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/linear_dynamics",
    )
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-specs",
        default=";".join(DEFAULT_GROUPS),
        help="Semicolon-separated group specs. Use global or comma-separated transition-table columns.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    transition_path = Path(args.transition_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(transition_path)
    available_n_pcs = len(pc_columns(table, "t"))
    n_pcs = min(args.n_pcs, available_n_pcs)
    if n_pcs < 1:
        raise ValueError("No PC columns were found in the transition table.")

    group_specs = parse_group_specs(args.group_specs)
    results = fit_all_groups(
        table,
        n_pcs=n_pcs,
        group_specs=group_specs,
        min_samples=args.min_samples,
        test_size=args.test_size,
        seed=args.seed,
    )
    results.to_csv(output_dir / "linear_dynamics_summary.csv", index=False, encoding="utf-8-sig")
    plot_fixed_points(results, output_dir)
    plot_spectral_radius(results, output_dir)
    write_readme(output_dir, transition_path, results)

    print(f"Fitted groups: {len(results)}")
    if not results.empty:
        global_row = results[results["group_type"].eq("global")]
        if not global_row.empty:
            print(f"Global test R2: {global_row['test_r2'].iloc[0]:.4f}")
            print(f"Global spectral radius: {global_row['spectral_radius'].iloc[0]:.4f}")
    print(f"Wrote linear dynamics analysis to {output_dir}")


if __name__ == "__main__":
    main()
