from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from MazeDataset import MazeSequenceDataset, collate_maze_sequences
from MazeRNNAgent import MazeActionRNN
from MazeTrainer.train_action_rnn import move_batch_to_device


def evaluate_by_task(model, dataloader, device):
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_steps = 0
    by_task = defaultdict(lambda: {"correct": 0, "steps": 0})

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            targets = batch["target"]
            mask = batch["mask"]
            predictions = logits.argmax(dim=-1)

            total_loss += criterion(logits.reshape(-1, 4), targets.reshape(-1)).item()
            total_correct += predictions.eq(targets).logical_and(mask).sum().item()
            total_steps += mask.sum().item()

            for task in torch.unique(batch["task"][mask]).tolist():
                task_mask = mask & batch["task"].eq(task)
                by_task[int(task)]["correct"] += (
                    predictions.eq(targets).logical_and(task_mask).sum().item()
                )
                by_task[int(task)]["steps"] += task_mask.sum().item()

    rows = []
    for task, values in sorted(by_task.items()):
        rows.append(
            {
                "task": task,
                "n_steps": values["steps"],
                "accuracy": values["correct"] / values["steps"],
            }
        )

    overall = {
        "loss": total_loss / total_steps,
        "accuracy": total_correct / total_steps,
        "n_steps": total_steps,
    }
    return overall, pd.DataFrame(rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained maze action RNN.")
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--diagnostics", default="maze_healthy_batch123/trial_level_decode_diagnostics.csv")
    parser.add_argument("--checkpoint", default="outputs/rnn_action_model/best_model.pt")
    parser.add_argument("--output-dir", default="outputs/rnn_action_model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--valid-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})

    dataset = MazeSequenceDataset(
        csv_path=args.input,
        split="val",
        val_ratio=train_args.get("val_ratio", 0.2),
        seed=train_args.get("seed", 42),
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

    overall, by_task = evaluate_by_task(model, dataloader, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task.to_csv(output_dir / "eval_by_task.csv", index=False, encoding="utf-8-sig")

    print(f"Validation steps: {overall['n_steps']}")
    print(f"Validation loss: {overall['loss']:.4f}")
    print(f"Validation accuracy: {overall['accuracy']:.4f}")
    print(f"Wrote {output_dir / 'eval_by_task.csv'}")


if __name__ == "__main__":
    main()
