from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from MazeDataset import MazeSequenceDataset, collate_maze_sequences
from MazeRNNAgent import MazeActionRNN
from MazeTrainer.train_action_rnn import move_batch_to_device


def masked_numpy(batch, key):
    return batch[key][batch["mask"]].detach().cpu().numpy()


def collect_hidden_states(model, dataloader, device):
    model.eval()
    chunks = {
        "hidden": [],
        "logits": [],
        "state": [],
        "goal": [],
        "prev_action": [],
        "task": [],
        "replan": [],
        "target": [],
        "subid": [],
        "trial": [],
        "step": [],
    }

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            model_pass = model(batch, return_hidden=True)
            mask = batch["mask"]

            chunks["hidden"].append(model_pass["hidden"][mask].detach().cpu().numpy())
            chunks["logits"].append(model_pass["logits"][mask].detach().cpu().numpy())
            for key in ["state", "goal", "prev_action", "task", "replan", "target"]:
                chunks[key].append(masked_numpy(batch, key))

            batch_size, max_len = mask.shape
            step_index = torch.arange(max_len, device=device).expand(batch_size, max_len)
            subid = batch["subid"].unsqueeze(1).expand(batch_size, max_len)
            trial = batch["trial"].unsqueeze(1).expand(batch_size, max_len)
            chunks["step"].append(step_index[mask].detach().cpu().numpy())
            chunks["subid"].append(subid[mask].detach().cpu().numpy())
            chunks["trial"].append(trial[mask].detach().cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def fit_pca(hidden, n_components):
    n_components = min(n_components, hidden.shape[0], hidden.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pc = pca.fit_transform(hidden - hidden.mean(axis=0, keepdims=True))
    return pca, pc


def subsample_for_decoding(hidden, labels, max_samples, seed):
    if max_samples is None or hidden.shape[0] <= max_samples:
        return hidden, labels
    rng = np.random.default_rng(seed)
    index = rng.choice(hidden.shape[0], size=max_samples, replace=False)
    return hidden[index], labels[index]


def decode_label(hidden, labels, max_samples, seed):
    labels = np.asarray(labels)
    keep = labels >= 0
    hidden = hidden[keep]
    labels = labels[keep]

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 5:
        return {
            "n_samples": int(len(labels)),
            "n_classes": int(len(classes)),
            "accuracy": np.nan,
            "chance": np.nan,
            "note": "too_few_classes_or_samples",
        }

    hidden, labels = subsample_for_decoding(hidden, labels, max_samples, seed)
    classes, counts = np.unique(labels, return_counts=True)
    n_splits = min(5, int(counts.min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=np.logspace(-4, 4, 9)),
    )

    scores = []
    for train_idx, test_idx in splitter.split(hidden, labels):
        model.fit(hidden[train_idx], labels[train_idx])
        scores.append(model.score(hidden[test_idx], labels[test_idx]))

    return {
        "n_samples": int(len(labels)),
        "n_classes": int(len(classes)),
        "accuracy": float(np.mean(scores)),
        "chance": float(counts.max() / counts.sum()),
        "note": "",
    }


def write_outputs(arrays, output_dir, n_pcs, max_decode_samples, seed):
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "hidden_states.npz", **arrays)

    step_table = pd.DataFrame(
        {
            key: arrays[key]
            for key in [
                "subid",
                "trial",
                "step",
                "state",
                "goal",
                "prev_action",
                "task",
                "replan",
                "target",
            ]
        }
    )
    step_table.to_csv(output_dir / "hidden_step_table.csv", index=False, encoding="utf-8-sig")

    pca, pc = fit_pca(arrays["hidden"], n_pcs)
    pca_summary = pd.DataFrame(
        {
            "pc": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "variance_ratio": pca.explained_variance_ratio_,
            "variance_cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    pca_summary.to_csv(output_dir / "hidden_pca_summary.csv", index=False, encoding="utf-8-sig")

    pc_table = step_table.copy()
    for idx in range(pc.shape[1]):
        pc_table[f"PC{idx + 1}"] = pc[:, idx]
    pc_table.to_csv(output_dir / "hidden_pca_scores.csv", index=False, encoding="utf-8-sig")

    decoding_rows = []
    for label_name in ["state", "goal", "task", "replan", "target", "prev_action"]:
        result = decode_label(
            arrays["hidden"],
            arrays[label_name],
            max_samples=max_decode_samples,
            seed=seed,
        )
        decoding_rows.append({"label": label_name, **result})
    decoding = pd.DataFrame(decoding_rows)
    decoding.to_csv(output_dir / "hidden_decoding_summary.csv", index=False, encoding="utf-8-sig")

    return {
        "n_steps": int(arrays["hidden"].shape[0]),
        "hidden_dim": int(arrays["hidden"].shape[1]),
        "pca_cumulative_3": float(pca_summary["variance_cumulative_ratio"].iloc[min(2, len(pca_summary) - 1)]),
        "pca_cumulative_10": float(pca_summary["variance_cumulative_ratio"].iloc[min(9, len(pca_summary) - 1)]),
        "decoding": decoding,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract and analyze maze RNN hidden states.")
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--diagnostics", default="maze_healthy_batch123/trial_level_decode_diagnostics.csv")
    parser.add_argument("--checkpoint", default="outputs/rnn_action_model/best_model.pt")
    parser.add_argument("--output-dir", default="outputs/rnn_action_model/hidden_analysis")
    parser.add_argument("--split", default="val", choices=["train", "val", "valid", "validation", "all"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--max-decode-samples", type=int, default=5000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    seed = train_args.get("seed", 42)

    dataset = MazeSequenceDataset(
        csv_path=args.input,
        split=args.split,
        val_ratio=train_args.get("val_ratio", 0.2),
        seed=seed,
        max_trials=train_args.get("max_trials"),
        valid_only=args.valid_only,
        diagnostics_path=args.diagnostics,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_maze_sequences,
    )

    model = MazeActionRNN(hidden_dim=train_args.get("hidden_dim", 64)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    arrays = collect_hidden_states(model, dataloader, device)
    summary = write_outputs(
        arrays,
        Path(args.output_dir),
        n_pcs=args.n_pcs,
        max_decode_samples=args.max_decode_samples,
        seed=seed,
    )

    print(f"Extracted {summary['n_steps']} valid steps")
    print(f"Hidden dimension: {summary['hidden_dim']}")
    print(f"PC1-3 cumulative variance: {summary['pca_cumulative_3']:.4f}")
    print(f"PC1-10 cumulative variance: {summary['pca_cumulative_10']:.4f}")
    print(f"Wrote hidden analysis to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
