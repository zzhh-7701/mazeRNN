from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from MazeDataset import SubjectSequenceDataset, collate_subject_sequences
from MazeRNNAgent import MazeActionRNN, MinimalMazeActionRNN
from MazeTrainer.train_action_rnn import compute_loss_and_accuracy, move_batch_to_device


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct_weighted = 0.0
    total_steps = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            loss, accuracy, n_steps = compute_loss_and_accuracy(model, batch, criterion)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * n_steps
        total_correct_weighted += accuracy * n_steps
        total_steps += n_steps

    mean_loss = total_loss / total_steps if total_steps else 0.0
    mean_accuracy = total_correct_weighted / total_steps if total_steps else 0.0
    return mean_loss, mean_accuracy, total_steps


def save_log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize_dataset(dataset: SubjectSequenceDataset, split: str) -> dict:
    n_steps = int(sum(len(sample["target"]) for sample in dataset.samples))
    first_trial = min(sample["first_trial_index"] for sample in dataset.samples)
    last_trial = max(sample["last_trial_index"] for sample in dataset.samples)
    return {
        "split": split,
        "subject_id": dataset.subject_id,
        "n_sequences": len(dataset),
        "n_steps": n_steps,
        "first_trial_index": first_trial,
        "last_trial_index": last_trial,
        "val_every": dataset.val_every,
        "val_offset": dataset.val_offset,
    }


def save_grouping_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train a GRU on one subject with interspersed within-subject trial splits."
    )
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--maze-wall", default="maze_healthy_batch123/maze_wall.csv")
    parser.add_argument(
        "--diagnostics",
        default="maze_healthy_batch123/trial_level_decode_diagnostics.csv",
    )
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/rnn_subject_action_model")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--model-kind",
        default="standard",
        choices=["standard", "minimal"],
        help="standard uses task/replan context; minimal uses only the strict feature set.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=4)
    parser.add_argument("--val-offset", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser


def build_model(args):
    if args.model_kind == "minimal":
        return MinimalMazeActionRNN(hidden_dim=args.hidden_dim)
    return MazeActionRNN(hidden_dim=args.hidden_dim)


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "training_log.csv"
    if log_path.exists():
        log_path.unlink()

    train_dataset = SubjectSequenceDataset(
        csv_path=args.input,
        maze_wall_path=args.maze_wall,
        subject_id=args.subject_id,
        split="train",
        val_every=args.val_every,
        val_offset=args.val_offset,
        valid_only=args.valid_only,
        diagnostics_path=args.diagnostics,
        max_trials=args.max_trials,
    )
    val_dataset = SubjectSequenceDataset(
        csv_path=args.input,
        maze_wall_path=args.maze_wall,
        subject_id=args.subject_id,
        split="val",
        val_every=args.val_every,
        val_offset=args.val_offset,
        valid_only=args.valid_only,
        diagnostics_path=args.diagnostics,
        max_trials=args.max_trials,
    )

    grouping_rows = [
        summarize_dataset(train_dataset, "train"),
        summarize_dataset(val_dataset, "val"),
    ]
    save_grouping_summary(output_dir / "grouping_summary.csv", grouping_rows)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_subject_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_subject_sequences,
    )

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_accuracy = -1.0
    best_path = output_dir / "best_model.pt"

    print(f"Device: {device}")
    for row in grouping_rows:
        print(
            f"{row['split']}: {row['n_sequences']} sequences, "
            f"{row['n_steps']} steps, trial index "
            f"{row['first_trial_index']}..{row['last_trial_index']}"
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_steps = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_acc, val_steps = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "train_steps": train_steps,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_steps": val_steps,
        }
        save_log_row(log_path, row)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "model_kind": args.model_kind,
                    "input_features": (
                        [
                            "current_state",
                            "prev_action",
                            "goal",
                            "prev_reward",
                            "maze_wall",
                            "trial_start_flag",
                            "wall_hit",
                        ]
                        if args.model_kind == "minimal"
                        else ["current_state", "goal", "prev_action", "task", "replan"]
                    ),
                    "best_val_accuracy": best_val_accuracy,
                    "grouping": grouping_rows,
                },
                best_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved best model to {best_path}")
    print(f"Saved training log to {log_path}")
    print(f"Saved grouping summary to {output_dir / 'grouping_summary.csv'}")


if __name__ == "__main__":
    main()
