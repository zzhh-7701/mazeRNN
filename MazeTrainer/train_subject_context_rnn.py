from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from MazeDataset import FullSubjectSequenceDataset
from MazeRNNAgent import MinimalMazeActionRNN


INPUT_FEATURES = [
    "current_state",
    "prev_action",
    "goal",
    "prev_reward",
    "maze_wall",
    "trial_start_flag",
    "wall_hit",
]


def move_sample_to_device(sample, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def slice_chunk(sample, start: int, end: int):
    chunk = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == sample["target"].shape[0]:
            chunk[key] = value[start:end].unsqueeze(0)
        else:
            chunk[key] = value
    return chunk


def forward_chunk(model, chunk, hidden=None):
    x = model.build_input(chunk)
    output, hidden = model.rnn(x, hidden)
    logits = model.action_head(output)
    return logits, hidden


def masked_loss_accuracy(logits, targets, mask, criterion):
    flat_logits = logits.reshape(-1, 4)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    n_steps = int(flat_mask.sum().item())
    if n_steps == 0:
        return None, 0.0, 0

    loss = criterion(flat_logits[flat_mask], flat_targets[flat_mask])
    predictions = flat_logits.argmax(dim=-1)
    correct = predictions[flat_mask].eq(flat_targets[flat_mask]).sum().item()
    return loss, correct / n_steps, n_steps


def run_train_epoch(model, sample, criterion, optimizer, chunk_len: int):
    model.train()
    hidden = None
    total_loss = 0.0
    total_correct_weighted = 0.0
    total_steps = 0
    n_total = int(sample["target"].shape[0])

    for start in range(0, n_total, chunk_len):
        end = min(start + chunk_len, n_total)
        chunk = slice_chunk(sample, start, end)

        logits, hidden = forward_chunk(model, chunk, hidden)
        loss, accuracy, n_steps = masked_loss_accuracy(
            logits,
            chunk["target"],
            chunk["train_mask"],
            criterion,
        )

        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * n_steps
            total_correct_weighted += accuracy * n_steps
            total_steps += n_steps

        if hidden is not None:
            hidden = hidden.detach()

    mean_loss = total_loss / total_steps if total_steps else 0.0
    mean_accuracy = total_correct_weighted / total_steps if total_steps else 0.0
    return mean_loss, mean_accuracy, total_steps


def run_eval(model, sample, criterion, chunk_len: int, mask_key: str):
    model.eval()
    hidden = None
    total_loss = 0.0
    total_correct_weighted = 0.0
    total_steps = 0
    n_total = int(sample["target"].shape[0])

    with torch.no_grad():
        for start in range(0, n_total, chunk_len):
            end = min(start + chunk_len, n_total)
            chunk = slice_chunk(sample, start, end)
            logits, hidden = forward_chunk(model, chunk, hidden)
            loss, accuracy, n_steps = masked_loss_accuracy(
                logits,
                chunk["target"],
                chunk[mask_key],
                criterion,
            )
            if loss is not None:
                total_loss += float(loss.item()) * n_steps
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


def save_grouping_summary(path: Path, dataset: FullSubjectSequenceDataset) -> None:
    sample = dataset.sample
    row = {
        "subject_id": dataset.subject_id,
        "n_sequences": 1,
        "n_trials": len(dataset.trial_table),
        "n_steps": int(sample["target"].shape[0]),
        "train_steps": int(sample["train_mask"].sum().item()),
        "val_steps": int(sample["val_mask"].sum().item()),
        "first_trial_index": int(sample["first_trial_index"]),
        "last_trial_index": int(sample["last_trial_index"]),
        "val_every": dataset.val_every,
        "val_offset": dataset.val_offset,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train minimal RNN on a full chronological subject sequence with interspersed masks."
    )
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--maze-wall", default="maze_healthy_batch123/maze_wall.csv")
    parser.add_argument(
        "--diagnostics",
        default="maze_healthy_batch123/trial_level_decode_diagnostics.csv",
    )
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/rnn_subject_context_minimal")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=4)
    parser.add_argument("--val-offset", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FullSubjectSequenceDataset(
        csv_path=args.input,
        maze_wall_path=args.maze_wall,
        subject_id=args.subject_id,
        val_every=args.val_every,
        val_offset=args.val_offset,
        valid_only=args.valid_only,
        diagnostics_path=args.diagnostics,
        max_trials=args.max_trials,
    )
    sample = move_sample_to_device(dataset.sample, device)
    save_grouping_summary(output_dir / "grouping_summary.csv", dataset)

    log_path = output_dir / "training_log.csv"
    if log_path.exists():
        log_path.unlink()

    model = MinimalMazeActionRNN(hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_accuracy = -1.0
    best_path = output_dir / "best_model.pt"

    print(f"Device: {device}")
    print(
        f"Full sequence: {len(dataset.trial_table)} trials, "
        f"{sample['target'].shape[0]} steps "
        f"({int(sample['train_mask'].sum().item())} train, "
        f"{int(sample['val_mask'].sum().item())} val)"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_steps = run_train_epoch(
            model,
            sample,
            criterion,
            optimizer,
            chunk_len=args.chunk_len,
        )
        val_loss, val_acc, val_steps = run_eval(
            model,
            sample,
            criterion,
            chunk_len=args.chunk_len,
            mask_key="val_mask",
        )

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
                    "model_kind": "minimal",
                    "training_protocol": "full_subject_interspersed_context",
                    "input_features": INPUT_FEATURES,
                    "best_val_accuracy": best_val_accuracy,
                    "grouping": row,
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
