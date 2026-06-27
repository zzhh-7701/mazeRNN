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
from MazeDataset.memory_state import (
    ACTION_DELTAS,
    ACTION_ENCODING,
    DEFAULT_HEADING,
    MazeMemoryState,
    WALL_COLUMN_ORDER,
    compute_relative_action,
    update_heading,
)
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
WALL_COLUMNS = list(WALL_COLUMN_ORDER)
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


def parse_maybe_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return parse_json_list(value)


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


def memory_feature_tensor(
    memory: MazeMemoryState,
    state: int,
    goal: int,
    heading: int,
    walls: list[set[int]],
    variant: str,
    step_idx: int,
    device,
):
    return torch.tensor(
        memory.get_features(
            current_pos=state,
            goal_pos=goal,
            heading=heading,
            walls=walls,
            variant=variant,
            t=step_idx,
        ).reshape(1, 1, -1),
        dtype=torch.float32,
        device=device,
    )


def human_biased_logits(
    logits: torch.Tensor,
    memory: MazeMemoryState,
    state: int,
    goal: int,
    heading: int,
    args,
) -> torch.Tensor:
    if not getattr(args, "use_human_bias", False):
        return logits
    adjusted = logits.clone()
    current_goal_distance = abs((int(goal) // 7) - (int(state) // 7)) + abs(
        (int(goal) % 7) - (int(state) % 7)
    )
    for action in range(4):
        known_wall = float(memory.known_wall_mask[int(state), action])
        known_edge = float(memory.known_edge_mask[int(state), action])
        unknown_edge = max(0.0, 1.0 - min(1.0, known_wall + known_edge))
        adjusted[action] -= float(args.known_wall_penalty) * known_wall
        adjusted[action] -= float(args.revisit_penalty) * float(
            memory.visited_edge_count[int(state), action]
        )
        adjusted[action] += float(args.unexplored_bonus) * unknown_edge
        if heading >= 0 and action == (int(heading) + 2) % 4:
            adjusted[action] -= float(args.backtrack_penalty)

        candidate = int(state) + ACTION_DELTAS[action]
        if 0 <= candidate < 49:
            next_goal_distance = abs((int(goal) // 7) - (candidate // 7)) + abs(
                (int(goal) % 7) - (candidate % 7)
            )
            if next_goal_distance < current_goal_distance:
                adjusted[action] += float(args.goal_progress_bonus)
            if candidate in memory.recent_positions:
                adjusted[action] -= float(args.loop_penalty)
    return adjusted


def memory_soft_logits(
    logits: torch.Tensor,
    memory: MazeMemoryState,
    state: int,
    goal: int,
    heading: int,
    args,
) -> torch.Tensor:
    adjusted = logits.clone()
    current_goal_distance = abs((int(goal) // 7) - (int(state) // 7)) + abs(
        (int(goal) % 7) - (int(state) % 7)
    )
    for action in range(4):
        known_wall = float(memory.known_wall_mask[int(state), action])
        known_edge = float(memory.known_edge_mask[int(state), action])
        unknown_edge = max(0.0, 1.0 - min(1.0, known_wall + known_edge))
        adjusted[action] -= float(args.known_wall_penalty) * known_wall
        adjusted[action] -= float(args.revisit_penalty) * float(
            memory.visited_edge_count[int(state), action]
        )
        adjusted[action] += float(args.unexplored_bonus) * unknown_edge
        if heading >= 0 and action == (int(heading) + 2) % 4:
            adjusted[action] -= float(args.backtrack_penalty)
        candidate = int(state) + ACTION_DELTAS[action]
        if 0 <= candidate < 49:
            next_goal_distance = abs((int(goal) // 7) - (candidate // 7)) + abs(
                (int(goal) % 7) - (candidate % 7)
            )
            if next_goal_distance < current_goal_distance:
                adjusted[action] += float(args.goal_progress_bonus)
            if candidate in memory.recent_positions:
                adjusted[action] -= float(args.loop_penalty)
    return adjusted


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
        model = MinimalMazeActionRNN(
            hidden_dim=train_args.get("hidden_dim", 512),
            cognitive_feature_dim=train_args.get("cognitive_feature_dim", 0),
            include_maze_wall=train_args.get("include_maze_wall", True),
            use_relative_action_head=train_args.get("use_relative_action_head", False),
            use_auxiliary_heads=train_args.get("use_auxiliary_heads", False),
        ).to(device)
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
    initial_hidden=None,
    hard_illegal_mask: bool = False,
    rollout_mask_policy: str = "none",
    model_variant: str = "A",
    args=None,
) -> tuple[list[int], list[bool], list[int]]:
    state = int(row.start)
    goal = int(row.goal)
    prev_action = START_ACTION
    prev_reward = 0.0
    prev_hit = 0.0
    hidden = initial_hidden
    actions = []
    hits = []
    path = [state]
    hit_counts = Counter()
    memory = MazeMemoryState()
    memory.reset(task_id=int(row.task), trial_id=int(row.sequence_trial_index))
    heading = DEFAULT_HEADING

    for step_idx in range(max_rollout_steps):
        batch = {
            "state": torch.tensor([[state]], dtype=torch.long, device=device),
            "goal": torch.tensor([[goal]], dtype=torch.long, device=device),
            "prev_action": torch.tensor([[prev_action]], dtype=torch.long, device=device),
            "prev_reward": torch.tensor([[prev_reward]], dtype=torch.float32, device=device),
            "maze_wall": wall_feature_tensor(walls, state, device),
            "cognitive_features": memory_feature_tensor(
                memory,
                state,
                goal,
                heading,
                walls,
                model_variant,
                step_idx,
                device,
            ),
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
        logits_1d = logits.squeeze(0).squeeze(0)
        if args is not None and rollout_mask_policy != "memory_soft":
            logits_1d = human_biased_logits(logits_1d, memory, state, goal, heading, args)
            logits = logits_1d.reshape(1, 1, -1)
        policy = rollout_mask_policy
        if hard_illegal_mask and policy == "none":
            policy = "hard"
        if policy == "task_specific":
            policy = "hard" if int(row.task) in {2, 3} else "repeated_hit"
        if policy == "memory_soft":
            logits = memory_soft_logits(logits_1d, memory, state, goal, heading, args).reshape(1, 1, -1)
        elif policy == "hard":
            illegal = [
                action
                for action in range(4)
                if apply_action(state, action, walls)[1]
            ]
            if illegal:
                logits = logits.clone()
                logits[..., illegal] = -1e9
        elif policy == "repeated_hit":
            illegal = [
                action
                for action in range(4)
                if apply_action(state, action, walls)[1]
                and hit_counts[(state, action)] > 0
            ]
            if illegal:
                logits = logits.clone()
                logits[..., illegal] = -1e9
        action = int(logits.argmax(dim=-1).item())

        previous_state = state
        state, hit = apply_action(state, action, walls)
        if hit:
            hit_counts[(previous_state, action)] += 1
        actions.append(action)
        hits.append(bool(hit))
        path.append(state)
        memory.update(previous_state, action, state, hit, t=step_idx)
        memory.observe_local_walls(previous_state, walls)
        heading = update_heading(heading, action, hit)
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
    hard_illegal_mask: bool = False,
    rollout_mask_policy: str = "none",
    model_variant: str = "A",
    args=None,
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
                hard_illegal_mask=hard_illegal_mask,
                rollout_mask_policy=rollout_mask_policy,
                model_variant=model_variant,
                args=args,
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
    model_source = (
        "model_context_rollout"
        if "model_context_rollout" in summary["source"].unique()
        else "model_rollout"
    )
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
                    pivot.loc[task, (metric, model_source)]
                    - pivot.loc[task, (metric, "actual_subject")]
                )
            except KeyError:
                row[f"{metric}_model_minus_actual"] = np.nan
        diff_rows.append(row)

    return summary, pd.DataFrame(diff_rows).round(6)


def manhattan_distance(a: int, b: int) -> int:
    return abs((int(a) // 7) - (int(b) // 7)) + abs((int(a) % 7) - (int(b) % 7))


def summarize_metric_group(
    trial_features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if trial_features.empty:
        return pd.DataFrame(columns=["source", "task", *columns])
    return (
        trial_features.groupby(["source", "task"], dropna=False)[columns]
        .mean()
        .reset_index()
        .round(6)
    )


def compute_memory_strategy_outputs(
    behavior_trials: pd.DataFrame,
    base_walls: dict[int, list[set[int]]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    trial_rows = []
    step_rows = []

    for row in behavior_trials.itertuples(index=False):
        actions = [int(x) for x in parse_maybe_list(row.action)]
        hits = [bool(x) for x in parse_maybe_list(row.hits)]
        path = [int(x) for x in parse_maybe_list(row.true_path)]
        if not actions:
            continue
        walls = parse_trial_walls(row, base_walls)
        memory = MazeMemoryState()
        trial_index = int(getattr(row, "sequence_trial_index", getattr(row, "trial", -1)))
        task = int(row.task)
        memory.reset(task_id=task, trial_id=trial_index)
        heading = DEFAULT_HEADING
        collision_edges = Counter()

        counters = Counter()
        distance_deltas = []
        last_state = int(row.start)
        for step_idx, action in enumerate(actions):
            state = path[step_idx] if step_idx < len(path) else last_state
            local_features = memory.get_features(
                current_pos=state,
                goal_pos=int(row.goal),
                heading=heading,
                walls=walls,
                variant="D",
                t=step_idx,
            )
            known_wall = float(memory.known_wall_mask[state, action])
            known_edge = float(memory.known_edge_mask[state, action])
            unknown_edge = max(0.0, 1.0 - min(1.0, known_wall + known_edge))
            next_state = (
                path[step_idx + 1]
                if step_idx + 1 < len(path)
                else apply_action(state, action, walls)[0]
            )
            hit = hits[step_idx] if step_idx < len(hits) else next_state == state
            relative_action = (
                np.nan
                if heading == DEFAULT_HEADING
                else compute_relative_action(heading, action)
            )
            was_known_node = float(memory.known_node_mask[next_state])
            would_loop = float(next_state in memory.recent_positions)

            counters["collision"] += int(hit)
            counters["known_wall_collision"] += int(hit and known_wall > 0)
            counters["unknown_wall_collision"] += int(hit and unknown_edge > 0)
            counters["unknown_edge_try"] += int(unknown_edge > 0)
            counters["frontier_action"] += int(unknown_edge > 0)
            counters["new_node_discovery"] += int((not hit) and was_known_node == 0)
            counters["node_revisit"] += int((not hit) and memory.known_node_mask[next_state] > 0)
            counters["edge_revisit"] += int(memory.visited_edge_count[state, action] > 0)
            counters["recent_loop"] += int(would_loop)
            counters["backtrack"] += int(relative_action == 3)
            counters["forward"] += int(relative_action == 0)
            counters["left_turn"] += int(relative_action == 1)
            counters["right_turn"] += int(relative_action == 2)
            counters["backward"] += int(relative_action == 3)
            counters["wall_following"] += int(
                heading != DEFAULT_HEADING
                and (
                    memory.known_wall_mask[state, (heading - 1) % 4] > 0
                    or memory.known_wall_mask[state, (heading + 1) % 4] > 0
                )
            )
            counters["right_hand_rule_match"] += int(
                heading != DEFAULT_HEADING
                and (
                    (
                        memory.known_wall_mask[state, (heading + 1) % 4] == 0
                        and relative_action == 2
                    )
                    or (
                        memory.known_wall_mask[state, (heading + 1) % 4] > 0
                        and memory.known_wall_mask[state, heading] == 0
                        and relative_action == 0
                    )
                )
            )
            counters["left_hand_rule_match"] += int(
                heading != DEFAULT_HEADING
                and (
                    (
                        memory.known_wall_mask[state, (heading - 1) % 4] == 0
                        and relative_action == 1
                    )
                    or (
                        memory.known_wall_mask[state, (heading - 1) % 4] > 0
                        and memory.known_wall_mask[state, heading] == 0
                        and relative_action == 0
                    )
                )
            )
            distance_deltas.append(
                manhattan_distance(state, int(row.goal))
                - manhattan_distance(next_state, int(row.goal))
            )
            if hit:
                collision_edges[(state, action)] += 1

            step_rows.append(
                {
                    "source": row.source,
                    "task": task,
                    "trial": int(row.trial) if not pd.isna(row.trial) else trial_index,
                    "sequence_trial_index": trial_index,
                    "step_index": step_idx,
                    "state": state,
                    "goal": int(row.goal),
                    "action": action,
                    "relative_action": relative_action,
                    "heading": heading,
                    "collision": int(hit),
                    "known_wall_before": known_wall,
                    "known_edge_before": known_edge,
                    "unknown_edge_before": unknown_edge,
                    "visited_node_count_before": float(memory.visited_node_count[state]),
                    "visited_edge_count_before": float(memory.visited_edge_count[state, action]),
                    "recent_loop_before": would_loop,
                    "backtrack": int(relative_action == 3),
                    "local_feature_dim": len(local_features),
                }
            )
            memory.update(state, action, next_state, hit, t=step_idx)
            memory.observe_local_walls(state, walls)
            heading = update_heading(heading, action, hit)
            last_state = next_state

        n_steps = max(len(actions), 1)
        shortest_steps = max(len(parse_maybe_list(row.short_path)) - 1, 1)
        repeat_collisions = sum(max(0, count - 1) for count in collision_edges.values())
        trial_rows.append(
            {
                "source": row.source,
                "task": task,
                "trial": int(row.trial) if not pd.isna(row.trial) else trial_index,
                "sequence_trial_index": trial_index,
                "n_steps": n_steps,
                "collision_rate": counters["collision"] / n_steps,
                "repeat_collision_rate": repeat_collisions / n_steps,
                "known_wall_collision_rate": counters["known_wall_collision"] / n_steps,
                "unknown_wall_collision_rate": counters["unknown_wall_collision"] / n_steps,
                "node_revisit_rate": counters["node_revisit"] / n_steps,
                "edge_revisit_rate": counters["edge_revisit"] / n_steps,
                "recent_loop_rate": counters["recent_loop"] / n_steps,
                "backtrack_rate": counters["backtrack"] / n_steps,
                "unknown_edge_try_rate": counters["unknown_edge_try"] / n_steps,
                "new_node_discovery_rate": counters["new_node_discovery"] / n_steps,
                "frontier_action_rate": counters["frontier_action"] / n_steps,
                "forward_rate": counters["forward"] / n_steps,
                "left_turn_rate": counters["left_turn"] / n_steps,
                "right_turn_rate": counters["right_turn"] / n_steps,
                "backward_rate": counters["backward"] / n_steps,
                "right_hand_rule_match": counters["right_hand_rule_match"] / n_steps,
                "left_hand_rule_match": counters["left_hand_rule_match"] / n_steps,
                "wall_following_rate": counters["wall_following"] / n_steps,
                "distance_to_goal_delta_mean": float(np.mean(distance_deltas)),
                "shortest_path_deviation": n_steps - shortest_steps,
                "path_efficiency": shortest_steps / n_steps,
            }
        )

    trial_features = pd.DataFrame(trial_rows).round(6)
    outputs = {
        "trial_memory_features": pd.DataFrame(step_rows).round(6),
        "collision_metrics_by_task": summarize_metric_group(
            trial_features,
            [
                "collision_rate",
                "repeat_collision_rate",
                "known_wall_collision_rate",
                "unknown_wall_collision_rate",
            ],
        ),
        "loop_metrics_by_task": summarize_metric_group(
            trial_features,
            [
                "node_revisit_rate",
                "edge_revisit_rate",
                "recent_loop_rate",
                "backtrack_rate",
            ],
        ),
        "strategy_metrics_by_task": summarize_metric_group(
            trial_features,
            [
                "forward_rate",
                "left_turn_rate",
                "right_turn_rate",
                "backward_rate",
                "right_hand_rule_match",
                "left_hand_rule_match",
                "wall_following_rate",
            ],
        ),
        "memory_metrics_by_task": summarize_metric_group(
            trial_features,
            [
                "unknown_edge_try_rate",
                "new_node_discovery_rate",
                "frontier_action_rate",
                "distance_to_goal_delta_mean",
                "shortest_path_deviation",
                "path_efficiency",
            ],
        ),
    }
    outputs["trial_strategy_features"] = trial_features
    return trial_features, outputs


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
    parser.add_argument(
        "--hard-illegal-mask",
        action="store_true",
        help="Mask illegal wall actions during free rollout.",
    )
    parser.add_argument(
        "--rollout-mask-policy",
        default=None,
        choices=["none", "hard", "repeated_hit", "task_specific", "memory_soft"],
    )
    parser.add_argument("--model-variant", default=None, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--use-human-bias", action="store_true")
    parser.add_argument("--known-wall-penalty", type=float, default=None)
    parser.add_argument("--revisit-penalty", type=float, default=None)
    parser.add_argument("--loop-penalty", type=float, default=None)
    parser.add_argument("--unexplored-bonus", type=float, default=None)
    parser.add_argument("--goal-progress-bonus", type=float, default=None)
    parser.add_argument("--backtrack-penalty", type=float, default=None)
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
    hard_illegal_mask = bool(
        args.hard_illegal_mask
        or train_args.get("hard_illegal_action_mask", False)
        or train_args.get("training_recipe") == "minimal_rollout_constrained"
    )
    rollout_mask_policy = args.rollout_mask_policy
    if rollout_mask_policy is None:
        rollout_mask_policy = train_args.get("rollout_mask_policy", "none")
        if rollout_mask_policy == "none" and hard_illegal_mask:
            rollout_mask_policy = "hard"
    model_variant = args.model_variant or train_args.get("model_variant", "A")
    for name, default in [
        ("known_wall_penalty", 3.0),
        ("revisit_penalty", 0.15),
        ("loop_penalty", 0.5),
        ("unexplored_bonus", 0.25),
        ("goal_progress_bonus", 0.15),
        ("backtrack_penalty", 0.1),
    ]:
        if getattr(args, name) is None:
            setattr(args, name, train_args.get(name, default))
    args.use_human_bias = bool(args.use_human_bias or train_args.get("use_human_bias", False))

    args.subject_id = subject_id
    args.val_every = val_every
    args.val_offset = val_offset
    args.valid_only = valid_only
    args.max_trials = max_trials
    args.model_variant = model_variant

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
        model_variant=model_variant,
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
        model_variant=model_variant,
    )

    eval_rows = prepare_subject_rows(args)
    base_walls = load_base_walls(Path(args.maze_wall))
    eval_steps = collect_step_predictions(model, dataloader, device)

    baselines = compute_baseline_accuracy(eval_steps, eval_rows, train_dataset, base_walls)
    confusion_counts, confusion_normalized, per_action = compute_confusion_outputs(eval_steps)

    behavior_trials = simulate_predicted_trials(
        eval_rows,
        model,
        device,
        base_walls,
        hard_illegal_mask=hard_illegal_mask,
        rollout_mask_policy=rollout_mask_policy,
        model_variant=model_variant,
        args=args,
    )
    teacher_forced_behavior_trials = simulate_teacher_forced_predicted_trials(
        eval_rows,
        eval_steps,
        base_walls,
    )
    behavior_metrics = add_trial_metrics(behavior_trials)
    teacher_forced_behavior_metrics = add_trial_metrics(teacher_forced_behavior_trials)
    behavior_summary, behavior_diff = summarize_behavior(behavior_metrics)
    _, memory_outputs = compute_memory_strategy_outputs(behavior_trials, base_walls)
    variant_summary = pd.DataFrame(
        [
            {
                "checkpoint": str(checkpoint_path),
                "model_kind": train_args.get("model_kind", "standard"),
                "model_variant": model_variant,
                "cognitive_feature_dim": int(train_args.get("cognitive_feature_dim", 0)),
                "cognitive_feature_names": json.dumps(
                    train_args.get("cognitive_feature_names", []),
                    ensure_ascii=False,
                ),
                "include_maze_wall": bool(train_args.get("include_maze_wall", True)),
                "use_relative_action_head": bool(
                    train_args.get("use_relative_action_head", False)
                ),
                "use_auxiliary_heads": bool(train_args.get("use_auxiliary_heads", False)),
                "action_encoding": json.dumps(
                    train_args.get("action_encoding", ACTION_ENCODING),
                    ensure_ascii=False,
                ),
                "action_deltas": json.dumps(
                    train_args.get("action_deltas", ACTION_DELTAS),
                    ensure_ascii=False,
                ),
                "wall_column_order": json.dumps(
                    train_args.get("wall_column_order", WALL_COLUMNS),
                    ensure_ascii=False,
                ),
                "heading_first_step_ignored": bool(
                    train_args.get("heading_first_step_ignored", False)
                ),
                "hard_mask_used": bool(train_args.get("hard_mask_used", False)),
                "shortest_recovery_oracle_used": bool(
                    train_args.get("shortest_recovery_oracle_used", False)
                ),
                "rollout_mask_policy": rollout_mask_policy,
                "use_human_bias": bool(args.use_human_bias),
                "known_wall_penalty": float(args.known_wall_penalty),
                "revisit_penalty": float(args.revisit_penalty),
                "loop_penalty": float(args.loop_penalty),
                "unexplored_bonus": float(args.unexplored_bonus),
                "goal_progress_bonus": float(args.goal_progress_bonus),
                "backtrack_penalty": float(args.backtrack_penalty),
            }
        ]
    )

    eval_steps.to_csv(output_dir / "action_step_predictions.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(output_dir / "action_baseline_accuracy.csv", index=False, encoding="utf-8-sig")
    confusion_counts.to_csv(output_dir / "confusion_matrix_counts.csv", encoding="utf-8-sig")
    confusion_normalized.to_csv(output_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig")
    per_action.to_csv(output_dir / "per_action_accuracy.csv", index=False, encoding="utf-8-sig")
    variant_summary.to_csv(output_dir / "model_variant_summary.csv", index=False, encoding="utf-8-sig")
    behavior_metrics.to_csv(output_dir / "behavior_recovery_trial_metrics.csv", index=False, encoding="utf-8-sig")
    behavior_summary.to_csv(output_dir / "behavior_recovery_by_task.csv", index=False, encoding="utf-8-sig")
    behavior_diff.to_csv(output_dir / "behavior_recovery_model_minus_actual.csv", index=False, encoding="utf-8-sig")
    teacher_forced_behavior_metrics.to_csv(
        output_dir / "behavior_recovery_teacher_forced_trial_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for name, frame in memory_outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print(f"Subject: {subject_id}")
    print(
        f"Split: {args.split}, valid_only={valid_only}, "
        f"hard_illegal_mask={hard_illegal_mask}, rollout_mask_policy={rollout_mask_policy}, "
        f"model_variant={model_variant}, use_human_bias={args.use_human_bias}"
    )
    print(f"Evaluation steps: {len(eval_steps)}")
    print("\nAction baseline accuracy:")
    print(baselines.to_string(index=False))
    print("\nPer-action accuracy:")
    print(per_action.to_string(index=False))
    print("\nBehavior recovery by task:")
    print(behavior_summary.to_string(index=False))
    print("\nMemory/strategy metrics by task:")
    print(memory_outputs["memory_metrics_by_task"].to_string(index=False))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
