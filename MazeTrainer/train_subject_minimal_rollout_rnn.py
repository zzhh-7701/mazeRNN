from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from MazeDataset import SubjectSequenceDataset, collate_subject_sequences
from MazeDataset.maze_sequence_dataset import IGNORE_INDEX, START_ACTION, parse_json_list
from MazeDataset.subject_sequence_dataset import interspersed_trial_split
from MazeRNNAgent import MinimalMazeActionRNN
from MazeTrainer.train_action_rnn import move_batch_to_device


ACTION_DELTAS = {0: -7, 1: -1, 2: 7, 3: 1}
WALL_COLUMNS = ["walls_l", "walls_u", "walls_r", "walls_d"]
CORE_DIAGNOSTIC_COLUMNS = [
    "true_path_end_ok",
    "hit_consistent",
    "short_path_valid_in_maze",
    "short_path_is_shortest",
]


def load_base_walls(path: str | Path) -> dict[int, list[set[int]]]:
    wall_df = pd.read_csv(path)
    return {
        int(row.maze): [
            set(parse_json_list(getattr(row, col))) for col in WALL_COLUMNS
        ]
        for row in wall_df.itertuples(index=False)
    }


def parse_trial_walls(row, base_walls: dict[int, list[set[int]]]) -> list[set[int]]:
    maze_wall = getattr(row, "maze_wall", "")
    if pd.notna(maze_wall) and maze_wall != "":
        return [set(x) for x in json.loads(maze_wall)]
    return base_walls[int(row.maze)]


def wall_features(walls: list[set[int]], state: int) -> list[float]:
    return [1.0 if state in direction_walls else 0.0 for direction_walls in walls]


def legal_actions(walls: list[set[int]], state: int) -> list[int]:
    return [
        action
        for action in range(4)
        if state not in walls[action]
        and 0 <= state + ACTION_DELTAS[action] < 49
    ]


def apply_action(state: int, action: int, walls: list[set[int]]) -> tuple[int, bool]:
    if state in walls[action]:
        return state, True
    next_state = state + ACTION_DELTAS[action]
    if not 0 <= next_state < 49:
        return state, True
    return next_state, False


def shortest_distances_to_goal(walls: list[set[int]], goal: int) -> dict[int, int]:
    distances = {int(goal): 0}
    queue = deque([int(goal)])
    while queue:
        current = queue.popleft()
        for state in range(49):
            if state in distances:
                continue
            for action in legal_actions(walls, state):
                next_state, hit = apply_action(state, action, walls)
                if not hit and next_state == current:
                    distances[state] = distances[current] + 1
                    queue.append(state)
                    break
    return distances


def shortest_recovery_action(
    walls: list[set[int]],
    state: int,
    goal: int,
    cache: dict[tuple[int, int], int],
) -> int:
    key = (int(state), int(goal))
    if key in cache:
        return cache[key]
    if state == goal:
        cache[key] = IGNORE_INDEX
        return IGNORE_INDEX

    queue = deque([(state, None)])
    visited = {state}
    while queue:
        current, first_action = queue.popleft()
        for action in legal_actions(walls, current):
            next_state, hit = apply_action(current, action, walls)
            if hit or next_state in visited:
                continue
            next_first = action if first_action is None else first_action
            if next_state == goal:
                cache[key] = next_first
                return next_first
            visited.add(next_state)
            queue.append((next_state, next_first))

    cache[key] = IGNORE_INDEX
    return IGNORE_INDEX


def mask_illegal_logits(logits: torch.Tensor, walls: list[set[int]], state: int) -> torch.Tensor:
    illegal = [
        action
        for action in range(4)
        if state in walls[action] or not 0 <= state + ACTION_DELTAS[action] < 49
    ]
    if not illegal:
        return logits
    masked = logits.clone()
    masked[illegal] = -1e9
    return masked


def stratified_sample_rows_by_task(rows: list, n_rows: int) -> list:
    if n_rows <= 0 or not rows:
        return []
    rows_by_task: dict[int, list] = {}
    for row in rows:
        rows_by_task.setdefault(int(row.task), []).append(row)

    tasks = sorted(rows_by_task)
    per_task = max(1, n_rows // len(tasks))
    remainder = max(0, n_rows - per_task * len(tasks))
    sampled = []
    for task in tasks:
        task_rows = rows_by_task[task]
        k = per_task + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        if len(task_rows) >= k:
            sampled.extend(random.sample(task_rows, k=k))
        else:
            sampled.extend(random.choices(task_rows, k=k))
    random.shuffle(sampled)
    return sampled[:n_rows]


def prepare_subject_rows(args, split: str) -> pd.DataFrame:
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
        split=split,
        val_every=args.val_every,
        val_offset=args.val_offset,
    )
    return df.loc[keep_mask].copy()


def forward_loss_and_accuracy(model, batch, behavior_weight, illegal_weight):
    logits = model(batch)
    targets = batch["target"]
    behavior_loss = F.cross_entropy(
        logits.reshape(-1, 4),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )

    probs = torch.softmax(logits, dim=-1)
    illegal_mask = batch["maze_wall"].bool()
    valid = batch["mask"].unsqueeze(-1)
    illegal_mass = probs.masked_fill(~illegal_mask, 0.0).sum(dim=-1)
    illegal_loss = illegal_mass[batch["mask"]].mean()
    loss = behavior_weight * behavior_loss + illegal_weight * illegal_loss

    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets).logical_and(batch["mask"]).sum().item()
    total = batch["mask"].sum().item()
    accuracy = correct / total if total else 0.0
    return loss, accuracy, total, {
        "behavior_loss": float(behavior_loss.detach().item()),
        "illegal_loss": float(illegal_loss.detach().item()),
    }


def rollout_constraint_loss(
    model,
    rows: list,
    base_walls: dict[int, list[set[int]]],
    recovery_cache: dict[tuple[int, int], int],
    device,
    max_steps: int,
    recovery_weight: float,
    no_progress_weight: float,
) -> tuple[torch.Tensor, dict]:
    recovery_losses = []
    no_progress_losses = []
    n_steps = 0
    n_prevented_illegal = 0
    n_revisit = 0
    n_recovery_targets = 0

    for row in rows:
        walls = parse_trial_walls(row, base_walls)
        goal = int(row.goal)
        distances = shortest_distances_to_goal(walls, goal)
        state = int(row.start)
        prev_action = START_ACTION
        prev_hit = 0.0
        prev_reward = 0.0
        hidden = None
        visited = {state}
        visit_counts = Counter([state])

        true_actions = parse_json_list(row.action)
        true_path = set(parse_json_list(row.true_path))
        short_path = parse_json_list(row.short_path)
        trial_max_steps = min(
            max_steps,
            max(8, 3 * max(len(true_actions), len(short_path) - 1, 1)),
        )

        for step_idx in range(trial_max_steps):
            batch = {
                "state": torch.tensor([[state]], dtype=torch.long, device=device),
                "goal": torch.tensor([[goal]], dtype=torch.long, device=device),
                "prev_action": torch.tensor([[prev_action]], dtype=torch.long, device=device),
                "prev_reward": torch.tensor([[prev_reward]], dtype=torch.float32, device=device),
                "maze_wall": torch.tensor(
                    [[wall_features(walls, state)]],
                    dtype=torch.float32,
                    device=device,
                ),
                "trial_start": torch.tensor(
                    [[1.0 if step_idx == 0 else 0.0]],
                    dtype=torch.float32,
                    device=device,
                ),
                "prev_hit": torch.tensor([[prev_hit]], dtype=torch.float32, device=device),
            }
            x = model.build_input(batch)
            output, hidden = model.rnn(x, hidden)
            logits = model.action_head(output).squeeze(0).squeeze(0)
            masked_logits = mask_illegal_logits(logits, walls, state)
            probs = torch.softmax(masked_logits, dim=-1)

            no_progress_actions = []
            current_distance = distances.get(state)
            for action in range(4):
                next_state, hit = apply_action(state, action, walls)
                if current_distance is not None:
                    next_distance = distances.get(next_state)
                    if (
                        not hit
                        and (next_distance is None or next_distance >= current_distance)
                    ):
                        no_progress_actions.append(action)
            if no_progress_actions:
                no_progress_losses.append(
                    no_progress_weight * probs[no_progress_actions].sum()
                )

            recovery_target = shortest_recovery_action(
                walls,
                state,
                goal,
                recovery_cache,
            )
            is_off_trajectory = state not in true_path or visit_counts[state] > 1
            if is_off_trajectory and recovery_target != IGNORE_INDEX:
                recovery_losses.append(
                    recovery_weight
                    * F.cross_entropy(
                        masked_logits.unsqueeze(0),
                        torch.tensor([recovery_target], dtype=torch.long, device=device),
                    )
                )
                n_recovery_targets += 1

            raw_action = int(logits.argmax().detach().item())
            action = int(masked_logits.argmax().detach().item())
            n_prevented_illegal += int(raw_action != action)
            next_state, hit = apply_action(state, action, walls)
            n_steps += 1
            n_revisit += int((not hit) and next_state in visited)

            prev_action = action
            prev_hit = 1.0 if hit else 0.0
            if hit:
                prev_reward = -1.0
            elif next_state == goal:
                prev_reward = 1.0
            else:
                prev_reward = 0.0
            state = next_state
            visited.add(state)
            visit_counts[state] += 1
            if state == goal:
                break

    losses = recovery_losses + no_progress_losses
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    stats = {
        "rollout_constraint_steps": n_steps,
        "rollout_prevented_illegal_rate": n_prevented_illegal / n_steps if n_steps else 0.0,
        "rollout_constraint_revisit_rate": n_revisit / n_steps if n_steps else 0.0,
        "rollout_recovery_targets": n_recovery_targets,
    }
    return loss, stats


def rollout_metrics(model, rows: pd.DataFrame, base_walls, device, max_steps: int) -> dict:
    model.eval()
    records = []
    with torch.no_grad():
        for row in rows.itertuples(index=False):
            walls = parse_trial_walls(row, base_walls)
            goal = int(row.goal)
            state = int(row.start)
            prev_action = START_ACTION
            prev_hit = 0.0
            prev_reward = 0.0
            hidden = None
            visited = {state}
            n_hit = 0
            n_revisit = 0
            true_actions = parse_json_list(row.action)
            short_path = parse_json_list(row.short_path)
            shortest_steps = max(len(short_path) - 1, 1)
            trial_max_steps = max(50, 3 * max(len(true_actions), shortest_steps, 1))
            trial_max_steps = min(max_steps, trial_max_steps)

            for step_idx in range(trial_max_steps):
                batch = {
                    "state": torch.tensor([[state]], dtype=torch.long, device=device),
                    "goal": torch.tensor([[goal]], dtype=torch.long, device=device),
                    "prev_action": torch.tensor([[prev_action]], dtype=torch.long, device=device),
                    "prev_reward": torch.tensor([[prev_reward]], dtype=torch.float32, device=device),
                    "maze_wall": torch.tensor(
                        [[wall_features(walls, state)]],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "trial_start": torch.tensor(
                        [[1.0 if step_idx == 0 else 0.0]],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "prev_hit": torch.tensor([[prev_hit]], dtype=torch.float32, device=device),
                }
                x = model.build_input(batch)
                output, hidden = model.rnn(x, hidden)
                logits = model.action_head(output).squeeze(0).squeeze(0)
                masked_logits = mask_illegal_logits(logits, walls, state)
                action = int(masked_logits.argmax().item())
                next_state, hit = apply_action(state, action, walls)
                n_hit += int(hit)
                n_revisit += int((not hit) and next_state in visited)
                prev_action = action
                prev_hit = 1.0 if hit else 0.0
                if hit:
                    prev_reward = -1.0
                elif next_state == goal:
                    prev_reward = 1.0
                else:
                    prev_reward = 0.0
                state = next_state
                visited.add(state)
                if state == goal:
                    break

            steps = step_idx + 1
            reached = int(state == goal)
            excess = max(0, steps - shortest_steps)
            records.append(
                {
                    "task": int(row.task),
                    "steps": steps,
                    "shortest_steps": shortest_steps,
                    "excess": excess,
                    "hit_rate": n_hit / steps if steps else 0.0,
                    "revisit_rate": n_revisit / steps if steps else 0.0,
                    "reached": reached,
                    "efficiency": shortest_steps / steps if steps else 0.0,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return {
            "rollout_score": -float("inf"),
            "reached_goal_rate": 0.0,
            "mean_excess_steps": 0.0,
            "mean_hit_rate": 0.0,
            "mean_revisit_rate": 0.0,
            "mean_efficiency": 0.0,
        }

    def score_frame(frame: pd.DataFrame) -> dict:
        reached = float(frame["reached"].mean())
        excess = float(frame["excess"].mean())
        hit = float(frame["hit_rate"].mean())
        revisit = float(frame["revisit_rate"].mean())
        efficiency = float(frame["efficiency"].mean())
        score = (
            2.0 * reached
            + 1.0 * efficiency
            - 0.03 * excess
            - 0.8 * hit
            - 0.4 * revisit
        )
        return {
            "score": score,
            "reached_goal_rate": reached,
            "mean_excess_steps": excess,
            "mean_hit_rate": hit,
            "mean_revisit_rate": revisit,
            "mean_efficiency": efficiency,
        }

    task_scores = {
        int(task): score_frame(task_df)
        for task, task_df in df.groupby("task", sort=True)
    }
    overall = score_frame(df)
    balanced_score = float(np.mean([values["score"] for values in task_scores.values()]))
    balanced_reached = float(
        np.mean([values["reached_goal_rate"] for values in task_scores.values()])
    )
    balanced_excess = float(
        np.mean([values["mean_excess_steps"] for values in task_scores.values()])
    )
    balanced_hit = float(
        np.mean([values["mean_hit_rate"] for values in task_scores.values()])
    )
    balanced_revisit = float(
        np.mean([values["mean_revisit_rate"] for values in task_scores.values()])
    )
    balanced_efficiency = float(
        np.mean([values["mean_efficiency"] for values in task_scores.values()])
    )
    result = {
        "rollout_score": balanced_score,
        "trial_weighted_rollout_score": overall["score"],
        "reached_goal_rate": balanced_reached,
        "mean_excess_steps": balanced_excess,
        "mean_hit_rate": balanced_hit,
        "mean_revisit_rate": balanced_revisit,
        "mean_efficiency": balanced_efficiency,
    }
    for task, values in task_scores.items():
        result[f"task{task}_rollout_score"] = values["score"]
        result[f"task{task}_reached_goal_rate"] = values["reached_goal_rate"]
        result[f"task{task}_mean_hit_rate"] = values["mean_hit_rate"]
        result[f"task{task}_mean_excess_steps"] = values["mean_excess_steps"]
    return result


def run_behavior_epoch(model, dataloader, device, args, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_behavior_loss = 0.0
    total_illegal_loss = 0.0
    total_correct_weighted = 0.0
    total_steps = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            loss, accuracy, n_steps, stats = forward_loss_and_accuracy(
                model,
                batch,
                behavior_weight=args.behavior_weight,
                illegal_weight=args.illegal_weight,
            )
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * n_steps
        total_behavior_loss += stats["behavior_loss"] * n_steps
        total_illegal_loss += stats["illegal_loss"] * n_steps
        total_correct_weighted += accuracy * n_steps
        total_steps += n_steps

    return {
        "loss": total_loss / total_steps if total_steps else 0.0,
        "behavior_loss": total_behavior_loss / total_steps if total_steps else 0.0,
        "illegal_loss": total_illegal_loss / total_steps if total_steps else 0.0,
        "accuracy": total_correct_weighted / total_steps if total_steps else 0.0,
        "steps": total_steps,
    }


def run_rollout_constraint_step(
    model,
    rows,
    base_walls,
    recovery_cache,
    device,
    args,
    optimizer,
):
    if args.rollout_weight <= 0.0 or args.rollout_trials_per_epoch <= 0:
        return {
            "rollout_constraint_loss": 0.0,
            "rollout_constraint_steps": 0,
            "rollout_constraint_hit_rate": 0.0,
            "rollout_constraint_revisit_rate": 0.0,
        }
    sampled_rows = stratified_sample_rows_by_task(rows, args.rollout_trials_per_epoch)
    optimizer.zero_grad()
    loss, stats = rollout_constraint_loss(
        model,
        sampled_rows,
        base_walls,
        recovery_cache,
        device,
        max_steps=args.rollout_constraint_max_steps,
        recovery_weight=args.off_trajectory_recovery_weight,
        no_progress_weight=args.rollout_no_progress_weight,
    )
    scaled_loss = args.rollout_weight * loss
    scaled_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        "rollout_constraint_loss": float(loss.detach().item()),
        **stats,
    }


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
    return {
        "split": split,
        "subject_id": dataset.subject_id,
        "n_sequences": len(dataset),
        "n_steps": n_steps,
        "first_trial_index": min(sample["first_trial_index"] for sample in dataset.samples),
        "last_trial_index": max(sample["last_trial_index"] for sample in dataset.samples),
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
        description=(
            "Train a strict minimal subject GRU using behavior cloning plus "
            "human-like rollout constraints and rollout-score model selection."
        )
    )
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--maze-wall", default="maze_healthy_batch123/maze_wall.csv")
    parser.add_argument(
        "--diagnostics",
        default="maze_healthy_batch123/trial_level_decode_diagnostics.csv",
    )
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/rnn_subject_minimal_rollout_h512")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=4)
    parser.add_argument("--val-offset", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--behavior-weight", type=float, default=1.0)
    parser.add_argument("--illegal-weight", type=float, default=0.03)
    parser.add_argument("--rollout-weight", type=float, default=0.08)
    parser.add_argument("--rollout-trials-per-epoch", type=int, default=24)
    parser.add_argument("--rollout-constraint-max-steps", type=int, default=40)
    parser.add_argument("--rollout-eval-max-steps", type=int, default=80)
    parser.add_argument("--off-trajectory-recovery-weight", type=float, default=1.0)
    parser.add_argument("--rollout-no-progress-weight", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric",
        default="rollout_score",
        choices=["rollout_score", "val_accuracy"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
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
    train_rows = list(prepare_subject_rows(args, "train").itertuples(index=False))
    val_rows = prepare_subject_rows(args, "val")
    base_walls = load_base_walls(args.maze_wall)
    recovery_cache: dict[tuple[int, int], int] = {}

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

    model = MinimalMazeActionRNN(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_value = -float("inf")
    best_path = output_dir / "best_model.pt"

    print(f"Device: {device}")
    print(f"Selection metric: {args.selection_metric}")
    for row in grouping_rows:
        print(
            f"{row['split']}: {row['n_sequences']} sequences, "
            f"{row['n_steps']} steps, trial index "
            f"{row['first_trial_index']}..{row['last_trial_index']}"
        )

    for epoch in range(1, args.epochs + 1):
        train_stats = run_behavior_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer=optimizer,
        )
        rollout_train_stats = run_rollout_constraint_step(
            model,
            train_rows,
            base_walls,
            recovery_cache,
            device,
            args,
            optimizer,
        )
        val_stats = run_behavior_epoch(model, val_loader, device, args, optimizer=None)
        rollout_stats = rollout_metrics(
            model,
            val_rows,
            base_walls,
            device,
            max_steps=args.rollout_eval_max_steps,
        )

        selection_value = (
            rollout_stats["rollout_score"]
            if args.selection_metric == "rollout_score"
            else val_stats["accuracy"]
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "train_behavior_loss": round(train_stats["behavior_loss"], 6),
            "train_illegal_loss": round(train_stats["illegal_loss"], 6),
            "train_accuracy": round(train_stats["accuracy"], 6),
            "train_steps": train_stats["steps"],
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in rollout_train_stats.items()
            },
            "val_loss": round(val_stats["loss"], 6),
            "val_behavior_loss": round(val_stats["behavior_loss"], 6),
            "val_illegal_loss": round(val_stats["illegal_loss"], 6),
            "val_accuracy": round(val_stats["accuracy"], 6),
            "val_steps": val_stats["steps"],
            **{
                key: round(value, 6)
                for key, value in rollout_stats.items()
            },
            "selection_metric": args.selection_metric,
            "selection_value": round(selection_value, 6),
        }
        save_log_row(log_path, row)

        if selection_value > best_value:
            best_value = selection_value
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args)
                    | {
                        "model_kind": "minimal",
                        "hard_illegal_action_mask": True,
                        "training_recipe": "minimal_rollout_constrained",
                    },
                    "model_kind": "minimal",
                    "training_recipe": "minimal_rollout_constrained",
                    "hard_illegal_action_mask": True,
                    "selection_metric": args.selection_metric,
                    "best_selection_value": best_value,
                    "best_rollout_metrics": rollout_stats,
                    "input_features": [
                        "current_state",
                        "prev_action",
                        "goal",
                        "prev_reward",
                        "maze_wall",
                        "trial_start_flag",
                        "wall_hit",
                    ],
                    "grouping": grouping_rows,
                },
                best_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train acc {train_stats['accuracy']:.4f} | "
            f"val acc {val_stats['accuracy']:.4f} | "
            f"rollout score {rollout_stats['rollout_score']:.4f}, "
            f"reach {rollout_stats['reached_goal_rate']:.3f}, "
            f"hit {rollout_stats['mean_hit_rate']:.3f}, "
            f"revisit {rollout_stats['mean_revisit_rate']:.3f}"
        )

    print(f"Best {args.selection_metric}: {best_value:.4f}")
    print(f"Saved best model to {best_path}")
    print(f"Saved training log to {log_path}")
    print(f"Saved grouping summary to {output_dir / 'grouping_summary.csv'}")


if __name__ == "__main__":
    main()
