from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from MazeAnalyzer.evaluate_subject_behavior_recovery import (
    ACTION_NAMES,
    compute_confusion_outputs,
    legal_actions,
    load_base_walls,
    load_model,
    model_rollout_trial,
    parse_trial_walls,
    shortest_path_action,
    simulate_teacher_forced_predicted_trials,
    summarize_behavior,
)
from MazeDataset import FullSubjectSequenceDataset
from MazeTrainer.train_subject_context_rnn import forward_chunk, move_sample_to_device, slice_chunk
from analysis.basic_behavior_stats import add_trial_metrics
from MazeDataset.maze_sequence_dataset import parse_json_list


def collect_context_predictions(model, sample, chunk_len: int) -> pd.DataFrame:
    model.eval()
    rows = []
    hidden = None
    n_total = int(sample["target"].shape[0])

    with torch.no_grad():
        for start in range(0, n_total, chunk_len):
            end = min(start + chunk_len, n_total)
            chunk = slice_chunk(sample, start, end)
            logits, hidden = forward_chunk(model, chunk, hidden)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            val_mask = chunk["val_mask"].squeeze(0)

            for local_idx in torch.nonzero(val_mask, as_tuple=False).flatten().tolist():
                true_action = int(chunk["target"][0, local_idx].item())
                pred_action = int(predictions[0, local_idx].item())
                rows.append(
                    {
                        "trial_index": int(chunk["trial_index"][0, local_idx].item()),
                        "step_index": int(chunk["step_index"][0, local_idx].item()),
                        "state": int(chunk["state"][0, local_idx].item()),
                        "goal": int(chunk["goal"][0, local_idx].item()),
                        "task": int(chunk["task"][0, local_idx].item()),
                        "prev_action": int(chunk["prev_action"][0, local_idx].item()),
                        "true_action": true_action,
                        "pred_action": pred_action,
                        "pred_prob": float(probs[0, local_idx, pred_action].item()),
                        "true_prob": float(probs[0, local_idx, true_action].item()),
                    }
                )
    return pd.DataFrame(rows)


def compute_context_baselines(eval_steps, eval_rows, sample, base_walls):
    train_targets = sample["target"][sample["train_mask"]].detach().cpu().tolist()
    train_majority_action = Counter(train_targets).most_common(1)[0][0]
    eval_majority_action = Counter(eval_steps["true_action"].tolist()).most_common(1)[0][0]

    row_by_trial = {
        int(row.sequence_trial_index): row for row in eval_rows.itertuples(index=False)
    }
    legal_random_probs = []
    shortest_preds = []
    for step in eval_steps.itertuples(index=False):
        trial_row = row_by_trial[int(step.trial_index)]
        walls = parse_trial_walls(trial_row, base_walls)
        legal = legal_actions(walls, int(step.state))
        legal_random_probs.append(1.0 / len(legal) if legal else 0.0)
        shortest_preds.append(shortest_path_action(trial_row, int(step.state)))

    previous_action_pred = eval_steps["prev_action"].where(
        eval_steps["prev_action"].between(0, 3),
        train_majority_action,
    )
    shortest_pred = pd.Series(shortest_preds, index=eval_steps.index)
    shortest_valid = shortest_pred.notna()

    def acc(values):
        return float(values.mean()) if len(values) else float("nan")

    return pd.DataFrame(
        [
            {
                "method": "model_argmax_context",
                "accuracy": acc(eval_steps["pred_action"].eq(eval_steps["true_action"])),
                "n_steps": len(eval_steps),
                "note": "trained RNN argmax with full-subject hidden context",
            },
            {
                "method": "uniform_random_expected",
                "accuracy": 0.25,
                "n_steps": len(eval_steps),
                "note": "expected accuracy for four equally likely actions",
            },
            {
                "method": "legal_random_expected",
                "accuracy": float(np.mean(legal_random_probs)),
                "n_steps": len(eval_steps),
                "note": "expected accuracy if guessing uniformly among legal moves",
            },
            {
                "method": "train_majority_action",
                "accuracy": acc(eval_steps["true_action"].eq(train_majority_action)),
                "n_steps": len(eval_steps),
                "note": f"always predicts {ACTION_NAMES[train_majority_action]} from train mask",
            },
            {
                "method": "eval_majority_action_oracle",
                "accuracy": acc(eval_steps["true_action"].eq(eval_majority_action)),
                "n_steps": len(eval_steps),
                "note": f"cheating baseline: always predicts eval majority {ACTION_NAMES[eval_majority_action]}",
            },
            {
                "method": "repeat_previous_action",
                "accuracy": acc(previous_action_pred.eq(eval_steps["true_action"])),
                "n_steps": len(eval_steps),
                "note": "uses previous action; first step falls back to train majority",
            },
            {
                "method": "shortest_path_next_step_oracle",
                "accuracy": acc(
                    shortest_pred[shortest_valid].eq(
                        eval_steps.loc[shortest_valid, "true_action"]
                    )
                ),
                "n_steps": int(shortest_valid.sum()),
                "note": "uses stored shortest path at the current true state",
            },
        ]
    ).round(6)


def warm_hidden_until(model, sample, end_step: int, chunk_len: int):
    hidden = None
    with torch.no_grad():
        for start in range(0, end_step, chunk_len):
            end = min(start + chunk_len, end_step)
            chunk = slice_chunk(sample, start, end)
            _, hidden = forward_chunk(model, chunk, hidden)
    return hidden


def simulate_context_rollout_trials(eval_rows, model, sample, device, base_walls, chunk_len: int):
    step_trial_indices = sample["trial_index"].detach().cpu().numpy()
    trial_start_step = {}
    for step_idx, trial_idx in enumerate(step_trial_indices):
        trial_start_step.setdefault(int(trial_idx), step_idx)

    rows = []
    with torch.no_grad():
        for row in eval_rows.itertuples(index=False):
            trial_index = int(row.sequence_trial_index)
            start_step = trial_start_step[trial_index]
            initial_hidden = warm_hidden_until(model, sample, start_step, chunk_len)
            walls = parse_trial_walls(row, base_walls)
            true_actions = parse_json_list(row.action)
            short_path = parse_json_list(row.short_path)
            actual_hits = parse_json_list(row.hits)
            actual_path = parse_json_list(row.true_path)
            actual_rt = parse_json_list(row.rt)

            max_rollout_steps = max(
                50,
                3 * max(len(true_actions), len(short_path) - 1, 1),
            )
            predicted_actions, predicted_hits, predicted_path = model_rollout_trial(
                model,
                row,
                walls,
                device,
                max_rollout_steps=max_rollout_steps,
                initial_hidden=initial_hidden,
            )

            predicted = row._asdict()
            predicted["source"] = "model_context_rollout"
            predicted["action"] = predicted_actions
            predicted["hits"] = predicted_hits
            predicted["true_path"] = predicted_path
            predicted["rt"] = [np.nan] * len(predicted_actions)
            rows.append(predicted)

            actual = row._asdict()
            actual["source"] = "actual_subject"
            actual["action"] = true_actions
            actual["hits"] = actual_hits
            actual["true_path"] = actual_path
            actual["rt"] = actual_rt
            rows.append(actual)

    behavior_df = pd.DataFrame(rows)
    behavior_df["short_path"] = behavior_df["short_path"].map(parse_json_list)
    return behavior_df


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate full-subject-context interspersed teacher forcing and rollout recovery."
    )
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--maze-wall", default="maze_healthy_batch123/maze_wall.csv")
    parser.add_argument(
        "--diagnostics",
        default="maze_healthy_batch123/trial_level_decode_diagnostics.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/rnn_subject_13015195273_context_minimal_h512/best_model.pt",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--chunk-len", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    model, train_args = load_model(checkpoint_path, device)

    chunk_len = args.chunk_len if args.chunk_len is not None else int(train_args.get("chunk_len", 256))
    subject_id = int(train_args["subject_id"])
    val_every = int(train_args.get("val_every", 4))
    val_offset = int(train_args.get("val_offset", 3))
    valid_only = bool(train_args.get("valid_only", False))
    max_trials = train_args.get("max_trials")

    dataset = FullSubjectSequenceDataset(
        csv_path=args.input,
        maze_wall_path=args.maze_wall,
        subject_id=subject_id,
        val_every=val_every,
        val_offset=val_offset,
        valid_only=valid_only,
        diagnostics_path=args.diagnostics,
        max_trials=max_trials,
    )
    sample = move_sample_to_device(dataset.sample, device)
    eval_rows = dataset.trial_table[dataset.trial_table["is_validation_trial"]].copy()
    base_walls = load_base_walls(Path(args.maze_wall))
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "context_recovery"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_steps = collect_context_predictions(model, sample, chunk_len)
    baselines = compute_context_baselines(eval_steps, eval_rows, sample, base_walls)
    confusion_counts, confusion_normalized, per_action = compute_confusion_outputs(eval_steps)

    behavior_trials = simulate_context_rollout_trials(
        eval_rows,
        model,
        sample,
        device,
        base_walls,
        chunk_len,
    )
    teacher_forced_trials = simulate_teacher_forced_predicted_trials(
        eval_rows,
        eval_steps,
        base_walls,
    )
    behavior_metrics = add_trial_metrics(behavior_trials)
    teacher_forced_metrics = add_trial_metrics(teacher_forced_trials)
    behavior_summary, behavior_diff = summarize_behavior(behavior_metrics)

    eval_steps.to_csv(output_dir / "action_step_predictions.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(output_dir / "action_baseline_accuracy.csv", index=False, encoding="utf-8-sig")
    confusion_counts.to_csv(output_dir / "confusion_matrix_counts.csv", encoding="utf-8-sig")
    confusion_normalized.to_csv(output_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig")
    per_action.to_csv(output_dir / "per_action_accuracy.csv", index=False, encoding="utf-8-sig")
    behavior_metrics.to_csv(output_dir / "behavior_recovery_trial_metrics.csv", index=False, encoding="utf-8-sig")
    teacher_forced_metrics.to_csv(
        output_dir / "behavior_recovery_teacher_forced_trial_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    behavior_summary.to_csv(output_dir / "behavior_recovery_by_task.csv", index=False, encoding="utf-8-sig")
    behavior_diff.to_csv(output_dir / "behavior_recovery_model_minus_actual.csv", index=False, encoding="utf-8-sig")

    print(f"Subject: {subject_id}")
    print(f"Protocol: full-subject interspersed context, valid_only={valid_only}")
    print(f"Evaluation steps: {len(eval_steps)}")
    print("\nAction baseline accuracy:")
    print(baselines.to_string(index=False))
    print("\nPer-action accuracy:")
    print(per_action.to_string(index=False))
    print("\nBehavior recovery by task:")
    print(behavior_summary.to_string(index=False))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
