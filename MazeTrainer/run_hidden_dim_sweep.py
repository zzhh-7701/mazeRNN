from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


def parse_hidden_dims(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_command(command, cwd: Path):
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def add_optional_flag(command, flag, enabled):
    if enabled:
        command.append(flag)


def train_one_dim(args, hidden_dim: int, output_dir: Path, repo_root: Path):
    command = [
        sys.executable,
        "-m",
        "MazeTrainer.train_action_rnn",
        "--input",
        args.input,
        "--diagnostics",
        args.diagnostics,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(hidden_dim),
        "--lr",
        str(args.lr),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.max_trials is not None:
        command.extend(["--max-trials", str(args.max_trials)])
    add_optional_flag(command, "--valid-only", args.valid_only)
    run_command(command, repo_root)


def evaluate_one_dim(args, output_dir: Path, repo_root: Path):
    command = [
        sys.executable,
        "-m",
        "MazeAnalyzer.evaluate_action_rnn",
        "--input",
        args.input,
        "--diagnostics",
        args.diagnostics,
        "--checkpoint",
        str(output_dir / "best_model.pt"),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
    ]
    add_optional_flag(command, "--valid-only", args.valid_only)
    run_command(command, repo_root)


def extract_hidden_one_dim(args, output_dir: Path, repo_root: Path):
    command = [
        sys.executable,
        "-m",
        "MazeAnalyzer.extract_hidden_states",
        "--input",
        args.input,
        "--diagnostics",
        args.diagnostics,
        "--checkpoint",
        str(output_dir / "best_model.pt"),
        "--output-dir",
        str(output_dir / "hidden_analysis"),
        "--split",
        args.hidden_split,
        "--batch-size",
        str(args.batch_size),
        "--n-pcs",
        str(args.n_pcs),
        "--max-decode-samples",
        str(args.max_decode_samples),
    ]
    add_optional_flag(command, "--valid-only", args.valid_only)
    run_command(command, repo_root)


def load_dim_summary(hidden_dim: int, output_dir: Path):
    row = {
        "hidden_dim": hidden_dim,
        "output_dir": str(output_dir),
    }

    log_path = output_dir / "training_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        best_idx = log["val_accuracy"].idxmax()
        best = log.loc[best_idx]
        last = log.iloc[-1]
        row.update(
            {
                "best_epoch": int(best["epoch"]),
                "best_train_accuracy": float(best["train_accuracy"]),
                "best_val_accuracy": float(best["val_accuracy"]),
                "last_train_accuracy": float(last["train_accuracy"]),
                "last_val_accuracy": float(last["val_accuracy"]),
                "last_train_loss": float(last["train_loss"]),
                "last_val_loss": float(last["val_loss"]),
            }
        )

    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        row["checkpoint_best_val_accuracy"] = float(checkpoint.get("best_val_accuracy", float("nan")))

    pca_path = output_dir / "hidden_analysis" / "hidden_pca_summary.csv"
    if pca_path.exists():
        pca = pd.read_csv(pca_path)
        row["pca_cumulative_3"] = float(
            pca["variance_cumulative_ratio"].iloc[min(2, len(pca) - 1)]
        )
        row["pca_cumulative_10"] = float(
            pca["variance_cumulative_ratio"].iloc[min(9, len(pca) - 1)]
        )

    decoding_path = output_dir / "hidden_analysis" / "hidden_decoding_summary.csv"
    if decoding_path.exists():
        decoding = pd.read_csv(decoding_path)
        for decode_row in decoding.itertuples(index=False):
            label = decode_row.label
            row[f"decode_{label}_accuracy"] = float(decode_row.accuracy)
            row[f"decode_{label}_chance"] = float(decode_row.chance)

    return row


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train and compare maze RNNs with different hidden dimensions."
    )
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--diagnostics", default="maze_healthy_batch123/trial_level_decode_diagnostics.csv")
    parser.add_argument("--output-root", default="outputs/rnn_hidden_dim_sweep")
    parser.add_argument("--hidden-dims", default="1,2,4,8,16,64")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-hidden-analysis", action="store_true")
    parser.add_argument("--hidden-split", default="val", choices=["train", "val", "valid", "validation", "all"])
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--max-decode-samples", type=int, default=5000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for hidden_dim in parse_hidden_dims(args.hidden_dims):
        output_dir = output_root / f"hidden_dim_{hidden_dim:03d}"
        checkpoint_path = output_dir / "best_model.pt"

        if args.skip_existing and checkpoint_path.exists():
            print(f"Skipping hidden_dim={hidden_dim}; checkpoint already exists.")
        else:
            train_one_dim(args, hidden_dim, output_dir, repo_root)

        evaluate_one_dim(args, output_dir, repo_root)
        if not args.no_hidden_analysis:
            extract_hidden_one_dim(args, output_dir, repo_root)

        summary_rows.append(load_dim_summary(hidden_dim, output_dir))
        pd.DataFrame(summary_rows).to_csv(
            output_root / "hidden_dim_sweep_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Wrote {output_root / 'hidden_dim_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
