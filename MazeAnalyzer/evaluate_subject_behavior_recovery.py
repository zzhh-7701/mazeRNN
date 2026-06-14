from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from MazeDataset import SubjectSequenceDataset, collate_subject_sequences
from MazeDataset.maze_sequence_dataset import START_ACTION, parse_json_list
from MazeDataset.subject_sequence_dataset import interspersed_trial_split
from MazeRNNAgent import MazeActionRNN, MinimalMazeActionRNN
from MazeTrainer.train_action_rnn import move_batch_to_device
from analysis.basic_behavior_stats import add_trial_metrics


ACTION_NAMES = {
    0: "left",
    1: "up",
    2: "right",
    3: "down",
}
ACTION_DELTAS = {
    0: -7,
    1: -1,
    2: 7,
    3: 1,
}
WALL_COLUMNS = ["walls_l", "walls_u", "walls_r", "walls_d"]
CORE_DIAGNOSTIC_COLUMNS = [
    "true_path_end_ok",
    "hit_consistent",
    "short_path_valid_in_maze",
    "short_path_is_shortest",
]


def load_base_walls(path: Path) -> dict[int, list[set[int]]]:
    wall_df = pd.read_csv(path)
    walls = {}
    for row in wall_df.itertuples(index=False):
        walls[int(row.maze)] = [
            set(parse_json_list(getattr(row, col))) for col in WALL_COLUMNS
        ]
    return walls


def parse_trial_walls(row, base_walls: dict[int, list[set[int]]]) -> list[set[int]]:
    if pd.notna(row.maze_wall) and row.maze_wall != "":
        return [set(x) for x in json.loads(row.maze_wall)]
    return base_walls[int(row.maze)]


def legal_actions(walls: list[set[int]], state: int) -> list[int]:
    return [action for action, blocked in enumerate(walls) if state not in blocked]


def apply_action(state: int, action: int, walls: list[set[int]]) -> tuple[int, bool]:
    if state in walls[action]:
        return state, True

    next_state = state + ACTION_DELTAS[action]
    if not 0 <= next_state < 49:
        return state, True
    return next_state, False


def wall_feature_tensor(walls: list[set[int]], state: int, device):
    values = [1.0 if state in direction_walls else 0.0 for direction_walls in walls]
    return torch.tensor([[values]], dtype=torch.float32, device=device)


def state_transition_to_action(current_state: int, next_state: int):
    delta = next_state - current_state
    for action, action_delta in ACTION_DELTAS.items():
        if delta == action_delta:
            return action
    return np.nan


def shortest_path_action(row, state: int):
    short_path = parse_json_list(row.short_path)
    for idx, path_state in enumerate(short_path[:-1]):
        if int(path_state) == int(state):
            return state_transition_to_action(int(path_state), int(short_path[idx + 1]))
    return np.nan


def prepare_subject_rows(args) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    if args.valid_only:
        diagnostics = pd.read_csv(args.diagnostics)
        valid_mask = diagnostics[CORE_DIAGNOSTIC_COLUMNS].eq(1).all(axis=1)
        df = df.loc[valid_mask.to_numpy()].copy()

    df = df[df["subid"].eq(args.subject_id)].copy()
    if df.empty:
        raise ValueError(f"No rows found for subject_id={args.subject_id}")

    df = df.sort_values(["day", "block", "trial", "replan", "createdat"], na_position="last")
    df = df.reset_index(drop=True)
    if args.max_trials is not None:
        df = df.head(args.max_trials)

    df["sequence_trial_index"] = np.arange(len(df))
    df["is_validation_trial"] = (
        df["sequence_trial_index"].to_numpy() % args.val_every == args.val_offset
    )
    keep_mask = interspersed_trial_split(
        len(df),
        split=args.split,
        val_every=args.val_every,
        val_offset=args.val_offset,
    )
    return df.loc[keep_mask].copy()


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model_kind = checkpoint.get("model_kind", train_args.get("model_kind", "standard"))
    if model_kind == "minimal":
        model = MinimalMazeActionRNN(hidden_dim=train_args.get("hidden_dim", 512)).to(device)
    else:
        model = MazeActionRNN(hidden_dim=train_args.get("hidden_dim", 64)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, train_args


def collect_step_predictions(model, dataloader, device) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            mask = batch["mask"]

            batch_size, max_steps = predictions.shape
            for batch_idx in range(batch_size):
                for step_pos in range(max_steps):
                    if not bool(mask[batch_idx, step_pos].item()):
                        continue
                    true_action = int(batch["target"][batch_idx, step_pos].item())
                    pred_action = int(predictions[batch_idx, step_pos].item())
                    rows.append(
                        {
                            "trial_index": int(
                                batch["trial_index"][batch_idx, step_pos].item()
                            ),
                            "step_index": int(
                                batch["step_index"][batch_idx, step_pos].item()
                            ),
                            "state": int(batch["state"][batch_idx, step_pos].item()),
                            "goal": int(batch["goal"][batch_idx, step_pos].item()),
                            "task": int(batch["task"][batch_idx, step_pos].item()),
                            "prev_action": int(
                                batch["prev_action"][batch_idx, step_pos].item()
                            ),
                            "true_action": true_action,
                            "pred_action": pred_action,
                            "pred_prob": float(
                                probs[batch_idx, step_pos, pred_action].item()
                            ),
                            "true_prob": float(
                                probs[batch_idx, step_pos, true_action].item()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def accuracy(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def compute_baseline_accuracy(
    eval_steps: pd.DataFrame,
    eval_rows: pd.DataFrame,
    train_dataset: SubjectSequenceDataset,
    base_walls: dict[int, list[set[int]]],
) -> pd.DataFrame:
    train_targets = []
    for sample in train_dataset.samples:
        train_targets.extend(sample["target"].tolist())
    train_counts = Counter(train_targets)
    train_majority_action = train_counts.most_common(1)[0][0]

    eval_counts = Counter(eval_steps["true_action"].tolist())
    eval_majority_action = eval_counts.most_common(1)[0][0]

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

    rows = [
        {
            "method": "model_argmax",
            "accuracy": accuracy(eval_steps["pred_action"].eq(eval_steps["true_action"])),
            "n_steps": len(eval_steps),
            "note": "trained RNN argmax",
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
            "accuracy": accuracy(eval_steps["true_action"].eq(train_majority_action)),
            "n_steps": len(eval_steps),
            "note": f"always predicts {ACTION_NAMES[train_majority_action]} from train split",
        },
        {
            "method": "eval_majority_action_oracle",
            "accuracy": accuracy(eval_steps["true_action"].eq(eval_majority_action)),
            "n_steps": len(eval_steps),
            "note": f"cheating baseline: always predicts eval majority {ACTION_NAMES[eval_majority_action]}",
        },
        {
            "method": "repeat_previous_action",
            "accuracy": accuracy(previous_action_pred.eq(eval_steps["true_action"])),
            "n_steps": len(eval_steps),
            "note": "uses previous action; first step falls back to train majority",
        },
        {
            "method": "shortest_path_next_step_oracle",
            "accuracy": accuracy(shortest_pred[shortest_valid].eq(eval_steps.loc[shortest_valid, "true_action"])),
            "n_steps": int(shortest_valid.sum()),
            "note": "uses stored shortest path at the current true state",
        },
    ]
    return pd.DataFrame(rows).round(6)


def compute_confusion_outputs(eval_steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = list(range(4))
    counts = pd.crosstab(
        eval_steps["true_action"],
        eval_steps["pred_action"],
    ).reindex(index=labels, columns=labels, fill_value=0)
    counts.index = [ACTION_NAMES[x] for x in counts.index]
    counts.columns = [ACTION_NAMES[x] for x in counts.columns]

    normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    per_action_rows = []
    for action_id, action_name in ACTION_NAMES.items():
        action_mask = eval_steps["true_action"].eq(action_id)
        per_action_rows.append(
            {
                "true_action": action_name,
                "n_steps": int(action_mask.sum()),
                "accuracy": accuracy(
                    eval_steps.loc[action_mask, "pred_action"].eq(action_id)
                ),
                "mean_true_prob": float(eval_steps.loc[action_mask, "true_prob"].mean()),
            }
        )
    return counts, normalized.round(6), pd.DataFrame(per_action_rows).round(6)


def model_rollout_trial(
    model,
    row,
    walls: list[set[int]],
    device,
    max_rollout_steps: int,
) -> tuple[list[int], list[bool], list[int]]:
    state = int(row.start)
    goal = int(row.goal)
    prev_action = START_ACTION
    prev_reward = 0.0
    prev_hit = 0.0
    hidden = None
    actions = []
    hits = []
    path = [state]

    for step_idx in range(max_rollout_steps):
        batch = {
            "state": torch.tensor([[state]], dtype=torch.long, device=device),
            "goal": torch.tensor([[goal]], dtype=torch.long, device=device),
            "prev_action": torch.tensor([[prev_action]], dtype=torch.long, device=device),
            "prev_reward": torch.tensor([[prev_reward]], dtype=torch.float32, device=device),
            "maze_wall": wall_feature_tensor(walls, state, device),
            "trial_start": torch.tensor(
                [[1.0 if step_idx == 0 else 0.0]],
                dtype=torch.float32,
                device=device,
            ),
            "prev_hit": torch.tensor([[prev_hit]], dtype=torch.float32, device=device),
            "task": torch.tensor([[int(row.task)]], dtype=torch.long, device=device),
            "replan": torch.tensor([[int(row.replan)]], dtype=torch.long, device=device),
        }
        if isinstance(model, MinimalMazeActionRNN):
            x = model.build_input(batch)
        else:
            x = torch.cat(
                [
                    model.state_embedding(batch["state"]),
                    model.goal_embedding(batch["goal"]),
                    model.prev_action_embedding(batch["prev_action"]),
                    model.task_embedding(batch["task"]),
                    model.replan_embedding(batch["replan"]),
                ],
                dim=-1,
            )
        output, hidden = model.rnn(x, hidden)
        logits = model.action_head(output)
        action = int(logits.argmax(dim=-1).item())

        state, hit = apply_action(state, action, walls)
        actions.append(action)
        hits.append(bool(hit))
        path.append(state)
        prev_action = action
        prev_hit = 1.0 if hit else 0.0
        if hit:
            prev_reward = -1.0
        elif state == goal:
            prev_reward = 1.0
        else:
            prev_reward = 0.0

        if state == goal:
            break

    return actions, hits, path


def simulate_predicted_trials(
    eval_rows: pd.DataFrame,
    model,
    device,
    base_walls: dict[int, list[set[int]]],
    rollout_max_multiplier: int = 3,
    rollout_min_max_steps: int = 50,
) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for row in eval_rows.itertuples(index=False):
            true_actions = parse_json_list(row.action)
            walls = parse_trial_walls(row, base_walls)
            short_path = parse_json_list(row.short_path)
            max_rollout_steps = max(
                rollout_min_max_steps,
                rollout_max_multiplier * max(len(true_actions), len(short_path) - 1, 1),
            )
            predicted_actions, predicted_hits, predicted_path = model_rollout_trial(
                model,
                row,
                walls,
                device,
                max_rollout_steps=max_rollout_steps,
            )

            row_dict = row._asdict()
            row_dict["source"] = "model_rollout"
            row_dict["action"] = predicted_actions
            row_dict["hits"] = predicted_hits
            row_dict["true_path"] = predicted_path
            row_dict["rt"] = [np.nan] * len(predicted_actions)
            rows.append(row_dict)

            actual_dict = row._asdict()
            actual_dict["source"] = "actual_subject"
            actual_dict["action"] = true_actions
            actual_dict["hits"] = parse_json_list(row.hits)
            actual_dict["true_path"] = parse_json_list(row.true_path)
            actual_dict["rt"] = parse_json_list(row.rt)
            rows.append(actual_dict)

    behavior_df = pd.DataFrame(rows)
    behavior_df["short_path"] = behavior_df["short_path"].map(parse_json_list)
    return behavior_df


def simulate_teacher_forced_predicted_trials(
    eval_rows: pd.DataFrame,
    eval_steps: pd.DataFrame,
    base_walls: dict[int, list[set[int]]],
) -> pd.DataFrame:
    step_groups = {
        int(trial_idx): group.sort_values("step_index")
        for trial_idx, group in eval_steps.groupby("trial_index")
    }

    rows = []
    for row in eval_rows.itertuples(index=False):
        trial_index = int(row.sequence_trial_index)
        if trial_index not in step_groups:
            continue

        true_actions = parse_json_list(row.action)
        predicted_actions = step_groups[trial_index]["pred_action"].astype(int).tolist()
        walls = parse_trial_walls(row, base_walls)

        state = int(row.start)
        predicted_path = [state]
        predicted_hits = []
        for action in predicted_actions:
            state, hit = apply_action(state, int(action), walls)
            predicted_path.append(state)
            predicted_hits.append(bool(hit))

        row_dict = row._asdict()
        row_dict["source"] = "model_teacher_forced"
        row_dict["action"] = predicted_actions
        row_dict["hits"] = predicted_hits
        row_dict["true_path"] = predicted_path
        row_dict["rt"] = [np.nan] * len(predicted_actions)
        rows.append(row_dict)

        actual_dict = row._asdict()
        actual_dict["source"] = "actual_subject"
        actual_dict["action"] = true_actions
        actual_dict["hits"] = parse_json_list(row.hits)
        actual_dict["true_path"] = parse_json_list(row.true_path)
        actual_dict["rt"] = parse_json_list(row.rt)
        rows.append(actual_dict)

    behavior_df = pd.DataFrame(rows)
    behavior_df["short_path"] = behavior_df["short_path"].map(parse_json_list)
    return behavior_df


def summarize_behavior(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_metrics = metrics[metrics["task"].isin([1, 2, 3, 4])].copy()
    summary = (
        task_metrics.groupby(["source", "task"], dropna=False)
        .agg(
            n_trials=("trial", "size"),
            mean_path_length=("actual_steps", "mean"),
            mean_hit_rate=("hit_rate", "mean"),
            optimal_path_rate=("is_optimal_by_length", "mean"),
            clean_optimal_rate=("is_clean_optimal", "mean"),
            reached_goal_rate=("reached_goal", "mean"),
            mean_path_efficiency=("path_efficiency", "mean"),
        )
        .reset_index()
        .round(6)
    )

    pivot = summary.pivot(index="task", columns="source")
    diff_rows = []
    for task in sorted(task_metrics["task"].dropna().unique()):
        row = {"task": int(task)}
        for metric in [
            "mean_path_length",
            "mean_hit_rate",
            "optimal_path_rate",
            "clean_optimal_rate",
            "reached_goal_rate",
            "mean_path_efficiency",
        ]:
            try:
                row[f"{metric}_model_minus_actual"] = (
                    pivot.loc[task, (metric, "model_rollout")]
                    - pivot.loc[task, (metric, "actual_subject")]
                )
            except KeyError:
                row[f"{metric}_model_minus_actual"] = np.nan
        diff_rows.append(row)

    return summary, pd.DataFrame(diff_rows).round(6)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate single-subject action baselines, confusion matrix, and behavior recovery."
    )
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--maze-wall", default="maze_healthy_batch123/maze_wall.csv")
    parser.add_argument(
        "--diagnostics",
        default="maze_healthy_batch123/trial_level_decode_diagnostics.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/rnn_subject_13015195273_h512/best_model.pt",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--subject-id", type=int, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "valid", "validation", "all"])
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--val-offset", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    model, train_args = load_model(checkpoint_path, device)

    subject_id = args.subject_id or int(train_args["subject_id"])
    val_every = args.val_every if args.val_every is not None else int(train_args.get("val_every", 4))
    val_offset = args.val_offset if args.val_offset is not None else int(train_args.get("val_offset", 3))
    valid_only = args.valid_only or bool(train_args.get("valid_only", False))
    max_trials = args.max_trials if args.max_trials is not None else train_args.get("max_trials")

    args.subject_id = subject_id
    args.val_every = val_every
    args.val_offset = val_offset
    args.valid_only = valid_only
    args.max_trials = max_trials

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "behavior_recovery"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SubjectSequenceDataset(
        csv_path=args.input,
        maze_wall_path=args.maze_wall,
        subject_id=subject_id,
        split=args.split,
        val_every=val_every,
        val_offset=val_offset,
        valid_only=valid_only,
        diagnostics_path=args.diagnostics,
        max_trials=max_trials,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_subject_sequences,
    )
    train_dataset = SubjectSequenceDataset(
        csv_path=args.input,
        maze_wall_path=args.maze_wall,
        subject_id=subject_id,
        split="train",
        val_every=val_every,
        val_offset=val_offset,
        valid_only=valid_only,
        diagnostics_path=args.diagnostics,
        max_trials=max_trials,
    )

    eval_rows = prepare_subject_rows(args)
    base_walls = load_base_walls(Path(args.maze_wall))
    eval_steps = collect_step_predictions(model, dataloader, device)

    baselines = compute_baseline_accuracy(eval_steps, eval_rows, train_dataset, base_walls)
    confusion_counts, confusion_normalized, per_action = compute_confusion_outputs(eval_steps)

    behavior_trials = simulate_predicted_trials(eval_rows, model, device, base_walls)
    teacher_forced_behavior_trials = simulate_teacher_forced_predicted_trials(
        eval_rows,
        eval_steps,
        base_walls,
    )
    behavior_metrics = add_trial_metrics(behavior_trials)
    teacher_forced_behavior_metrics = add_trial_metrics(teacher_forced_behavior_trials)
    behavior_summary, behavior_diff = summarize_behavior(behavior_metrics)

    eval_steps.to_csv(output_dir / "action_step_predictions.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(output_dir / "action_baseline_accuracy.csv", index=False, encoding="utf-8-sig")
    confusion_counts.to_csv(output_dir / "confusion_matrix_counts.csv", encoding="utf-8-sig")
    confusion_normalized.to_csv(output_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig")
    per_action.to_csv(output_dir / "per_action_accuracy.csv", index=False, encoding="utf-8-sig")
    behavior_metrics.to_csv(output_dir / "behavior_recovery_trial_metrics.csv", index=False, encoding="utf-8-sig")
    behavior_summary.to_csv(output_dir / "behavior_recovery_by_task.csv", index=False, encoding="utf-8-sig")
    behavior_diff.to_csv(output_dir / "behavior_recovery_model_minus_actual.csv", index=False, encoding="utf-8-sig")
    teacher_forced_behavior_metrics.to_csv(
        output_dir / "behavior_recovery_teacher_forced_trial_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Subject: {subject_id}")
    print(f"Split: {args.split}, valid_only={valid_only}")
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
