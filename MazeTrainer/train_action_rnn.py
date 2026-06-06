from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from MazeDataset import MazeSequenceDataset, collate_maze_sequences
from MazeRNNAgent import MazeActionRNN


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_loss_and_accuracy(model, batch, criterion):
    logits = model(batch)
    targets = batch["target"]
    loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))

    mask = batch["mask"]
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets).logical_and(mask).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total else 0.0
    return loss, accuracy, total


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

    mean_loss = total_loss / total_steps
    mean_accuracy = total_correct_weighted / total_steps
    return mean_loss, mean_accuracy


def save_log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a GRU on maze action sequences.")
    parser.add_argument(
        "--input",
        default="maze_healthy_batch123/trial_level.csv",
        help="Path to trial_level.csv.",
    )
    parser.add_argument(
        "--diagnostics",
        default="maze_healthy_batch123/trial_level_decode_diagnostics.csv",
        help="Path to diagnostics CSV.",
    )
    parser.add_argument("--output-dir", default="outputs/rnn_action_model")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    if log_path.exists():
        log_path.unlink()

    train_dataset = MazeSequenceDataset(
        csv_path=args.input,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_trials=args.max_trials,
        valid_only=args.valid_only,
        diagnostics_path=args.diagnostics,
    )
    val_dataset = MazeSequenceDataset(
        csv_path=args.input,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_trials=args.max_trials,
        valid_only=args.valid_only,
        diagnostics_path=args.diagnostics,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_maze_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_maze_sequences,
    )

    model = MazeActionRNN(hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_accuracy = -1.0
    best_path = output_dir / "best_model.pt"

    print(f"Device: {device}")
    print(f"Train trials: {len(train_dataset)}")
    print(f"Validation trials: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
        }
        save_log_row(log_path, row)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_accuracy": best_val_accuracy,
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


if __name__ == "__main__":
    main()
