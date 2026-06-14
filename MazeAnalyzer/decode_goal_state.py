from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_hidden_arrays(hidden_path: Path):
    data = np.load(hidden_path)
    return {key: data[key] for key in data.files}


def stratified_subsample(x, y, meta, max_samples, seed):
    if max_samples is None or len(y) <= max_samples:
        return x, y, meta

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        rng = np.random.default_rng(seed)
        index = rng.choice(len(y), size=max_samples, replace=False)
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=max_samples,
            random_state=seed,
        )
        index, _ = next(splitter.split(x, y))
    return x[index], y[index], meta.iloc[index].reset_index(drop=True)


def build_features(arrays, feature_space, n_pcs, seed):
    hidden = arrays["hidden"]
    if feature_space == "hidden":
        return hidden
    if feature_space == "pca":
        n_components = min(n_pcs, hidden.shape[0], hidden.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        return pca.fit_transform(hidden - hidden.mean(axis=0, keepdims=True))
    raise ValueError(f"Unknown feature space: {feature_space}")


def build_meta_table(arrays):
    keys = ["sample_id", "subid", "trial", "step", "task", "replan", "target", "state", "goal"]
    available = [key for key in keys if key in arrays]
    return pd.DataFrame({key: arrays[key] for key in available})


def decode_one_label(x, labels, meta, label_name, output_dir, max_samples, test_size, seed):
    labels = np.asarray(labels)
    keep = labels >= 0
    x = x[keep]
    labels = labels[keep]
    meta = meta.loc[keep].reset_index(drop=True)

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        raise ValueError(f"{label_name} has too few classes or samples for decoding.")

    x, labels, meta = stratified_subsample(x, labels, meta, max_samples, seed)
    x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
        x,
        labels,
        np.arange(len(labels)),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    model = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=np.logspace(-4, 4, 9)),
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    class_labels = np.unique(labels)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    pred_table = meta.iloc[test_idx].copy()
    pred_table[f"true_{label_name}"] = y_test
    pred_table[f"pred_{label_name}"] = y_pred
    pred_table[f"correct_{label_name}"] = y_pred == y_test

    per_class = pd.DataFrame(
        {
            label_name: class_labels,
            "n_test": cm.sum(axis=1),
            "accuracy": np.diag(cm) / np.maximum(cm.sum(axis=1), 1),
        }
    )

    summary = {
        "label": label_name,
        "n_samples": int(len(labels)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(class_labels)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "chance": float(np.max(np.bincount(labels.astype(int))) / len(labels)),
    }

    pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(
        output_dir / f"{label_name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(cm_norm, index=class_labels, columns=class_labels).to_csv(
        output_dir / f"{label_name}_confusion_matrix_normalized.csv",
        encoding="utf-8-sig",
    )
    pred_table.to_csv(output_dir / f"{label_name}_predictions.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(output_dir / f"{label_name}_per_class_accuracy.csv", index=False, encoding="utf-8-sig")
    plot_confusion_matrix(cm_norm, class_labels, label_name, output_dir)
    return summary


def plot_confusion_matrix(cm_norm, labels, label_name, output_dir):
    fig_size = max(6, min(12, len(labels) * 0.22))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Row-normalized accuracy")
    plt.xlabel(f"Predicted {label_name}")
    plt.ylabel(f"True {label_name}")
    plt.title(f"{label_name.capitalize()} decoding confusion matrix")
    if len(labels) <= 60:
        plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
        plt.yticks(np.arange(len(labels)), labels, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / f"{label_name}_confusion_matrix.png", dpi=220)
    plt.close()


def write_readme(output_dir, hidden_path, feature_space, summaries):
    lines = [
        "# Goal and state decoding",
        "",
        f"Input hidden file: {hidden_path}",
        f"Feature space: {feature_space}",
        "",
        "Each decoder uses a standardized RidgeClassifierCV model.",
        "Outputs include summary accuracy, per-class accuracy, confusion matrices, and test-set predictions.",
        "",
    ]
    for summary in summaries:
        lines.append(
            f"- {summary['label']}: accuracy={summary['accuracy']:.6f}, "
            f"balanced_accuracy={summary['balanced_accuracy']:.6f}, chance={summary['chance']:.6f}"
        )
    (output_dir / "README_goal_state_decoding.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Decode goal and state from maze RNN hidden states.")
    parser.add_argument(
        "--hidden-path",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/hidden_analysis/hidden_states.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rnn_hidden_dim_sweep/hidden_dim_008/goal_state_decoding",
    )
    parser.add_argument("--labels", default="goal,state", help="Comma-separated labels to decode.")
    parser.add_argument("--feature-space", choices=["hidden", "pca"], default="hidden")
    parser.add_argument("--n-pcs", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    hidden_path = Path(args.hidden_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_hidden_arrays(hidden_path)
    x = build_features(arrays, args.feature_space, args.n_pcs, args.seed)
    meta = build_meta_table(arrays)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]

    summaries = []
    for label_name in labels:
        if label_name not in arrays:
            raise ValueError(f"{label_name} is not present in {hidden_path}")
        summary = decode_one_label(
            x,
            arrays[label_name],
            meta,
            label_name,
            output_dir,
            max_samples=args.max_samples,
            test_size=args.test_size,
            seed=args.seed,
        )
        summaries.append(summary)
        print(
            f"{label_name}: accuracy={summary['accuracy']:.4f}, "
            f"balanced_accuracy={summary['balanced_accuracy']:.4f}, "
            f"chance={summary['chance']:.4f}"
        )

    pd.DataFrame(summaries).to_csv(output_dir / "goal_state_decoding_summary.csv", index=False, encoding="utf-8-sig")
    write_readme(output_dir, hidden_path, args.feature_space, summaries)
    print(f"Wrote goal/state decoding outputs to {output_dir}")


if __name__ == "__main__":
    main()
