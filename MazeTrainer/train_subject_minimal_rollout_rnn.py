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
from MazeDataset.memory_state import (
    ACTION_DELTAS,
    ACTION_ENCODING,
    DEFAULT_HEADING,
    MazeMemoryState,
    SubjectMazeMemoryStore,
    WALL_COLUMN_ORDER,
    feature_names_for_variant,
    update_heading,
)
from MazeDataset.maze_sequence_dataset import IGNORE_INDEX, START_ACTION, parse_json_list
from MazeDataset.subject_sequence_dataset import interspersed_trial_split
from MazeRNNAgent import MinimalMazeActionRNN
from MazeTrainer.train_action_rnn import move_batch_to_device


WALL_COLUMNS = list(WALL_COLUMN_ORDER)
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


def mask_logits_for_policy(
    logits: torch.Tensor,
    walls: list[set[int]],
    state: int,
    task: int,
    hit_counts: Counter,
    mask_policy: str,
) -> torch.Tensor:
    if mask_policy == "none":
        return logits
    if mask_policy == "hard":
        return mask_illegal_logits(logits, walls, state)

    effective_policy = mask_policy
    if mask_policy == "task_specific":
        effective_policy = "hard" if int(task) in {2, 3} else "repeated_hit"
    if effective_policy != "repeated_hit":
        return logits

    masked = logits.clone()
    for action in range(4):
        if apply_action(state, action, walls)[1] and hit_counts[(state, action)] > 0:
            masked[action] = -1e9
    return masked


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
        is_known_wall = float(memory.known_wall_mask[int(state), action])
        is_known_edge = float(memory.known_edge_mask[int(state), action])
        is_unknown_edge = max(0.0, 1.0 - min(1.0, is_known_wall + is_known_edge))
        adjusted[action] -= float(args.known_wall_penalty) * is_known_wall
        adjusted[action] -= float(args.revisit_penalty) * float(
            memory.visited_edge_count[int(state), action]
        )
        adjusted[action] += float(args.unexplored_bonus) * is_unknown_edge
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


def _copy_memory_store(store: SubjectMazeMemoryStore) -> SubjectMazeMemoryStore:
    return store.copy()


def build_incremental_memory_stores(
    all_rows: list,
    base_walls: dict[int, list[set[int]]],
) -> dict[int, SubjectMazeMemoryStore]:
    """Return a snapshot of the long-term memory store at the START of each trial.

    Processes all rows in chronological order (they must be pre-sorted) using
    the true trajectories so that each trial inherits the accumulated wall
    knowledge from all preceding trials.
    """
    snapshots: dict[int, SubjectMazeMemoryStore] = {}
    store = SubjectMazeMemoryStore()

    for row in all_rows:
        trial_idx = int(row.sequence_trial_index)
        snapshots[trial_idx] = _copy_memory_store(store)

        walls = parse_trial_walls(row, base_walls)
        maze_id = int(getattr(row, "maze", 1))
        task_id = int(row.task)
        is_hidden_task = task_id == 4
        memory = store.start_trial(maze_id=maze_id, task_id=task_id, trial_id=trial_idx)

        true_path = [int(x) for x in parse_json_list(row.true_path)]
        actions = [int(x) for x in parse_json_list(row.action)]
        hits_raw = parse_json_list(row.hits)

        n_steps = min(len(actions), len(true_path))
        for step_idx in range(n_steps):
            state = true_path[step_idx]
            action = actions[step_idx]
            hit = bool(hits_raw[step_idx]) if step_idx < len(hits_raw) else False
            next_state = true_path[step_idx + 1] if step_idx + 1 < len(true_path) else state
            memory.update(state, action, next_state, hit, t=step_idx)
            if not is_hidden_task:
                memory.observe_local_walls(state, walls)

        store.finish_trial(maze_id, memory)

    return snapshots


def parse_task_float_map(value: str, default: float) -> dict[int, float]:
    mapping = {task: default for task in [1, 2, 3, 4]}
    if not value:
        return mapping
    for item in value.split(","):
        if not item.strip():
            continue
        task, weight = item.split(":", 1)
        mapping[int(task)] = float(weight)
    return mapping


def build_action_priors(rows: list, alpha: float = 0.25) -> dict:
    global_counts = np.full(4, alpha, dtype=np.float64)
    task_counts = {task: np.full(4, alpha, dtype=np.float64) for task in [1, 2, 3, 4]}
    state_counts: dict[tuple[int, int], np.ndarray] = {}

    for row in rows:
        task = int(row.task)
        actions = [int(x) for x in parse_json_list(row.action)]
        path = [int(x) for x in parse_json_list(row.true_path)]
        for step_idx, action in enumerate(actions):
            state = int(path[step_idx]) if step_idx < len(path) else int(row.start)
            global_counts[action] += 1
            task_counts.setdefault(task, np.full(4, alpha, dtype=np.float64))[action] += 1
            state_counts.setdefault((task, state), np.full(4, alpha, dtype=np.float64))[action] += 1

    def normalize(counts):
        return counts / counts.sum()

    return {
        "global": normalize(global_counts),
        "task": {task: normalize(counts) for task, counts in task_counts.items()},
        "state": {key: normalize(counts) for key, counts in state_counts.items()},
    }


def local_action_prior(action_priors: dict, task: int, state: int) -> np.ndarray:
    return action_priors["state"].get(
        (int(task), int(state)),
        action_priors["task"].get(int(task), action_priors["global"]),
    )


def soft_recovery_target(
    walls: list[set[int]],
    state: int,
    task: int,
    shortest_action: int,
    action_priors: dict,
    shortest_weight_by_task: dict[int, float],
) -> torch.Tensor:
    prior = local_action_prior(action_priors, task, state).copy()
    for action in range(4):
        if apply_action(state, action, walls)[1]:
            prior[action] = 0.0
    if prior.sum() <= 0:
        legal = legal_actions(walls, state)
        prior = np.zeros(4, dtype=np.float64)
        for action in legal:
            prior[action] = 1.0 / len(legal)
    else:
        prior = prior / prior.sum()

    shortest_weight = shortest_weight_by_task.get(int(task), 0.6)
    target = (1.0 - shortest_weight) * prior
    if shortest_action != IGNORE_INDEX:
        target[int(shortest_action)] += shortest_weight
    return torch.tensor(target, dtype=torch.float32)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target.to(logits.device) * F.log_softmax(logits, dim=-1)).sum()


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


def forward_loss_and_accuracy(model, batch, behavior_weight, illegal_weight, args):
    output = model.forward_with_aux(batch) if hasattr(model, "forward_with_aux") else None
    logits = output["logits"] if output is not None else model(batch)
    targets = batch["target"]
    behavior_loss = F.cross_entropy(
        logits.reshape(-1, 4),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    relative_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    if (
        output is not None
        and "relative_logits" in output
        and "relative_target" in batch
        and args.lambda_relative > 0.0
    ):
        relative_loss = F.cross_entropy(
            output["relative_logits"].reshape(-1, 4),
            batch["relative_target"].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    probs = torch.softmax(logits, dim=-1)
    illegal_mask = batch["maze_wall"].bool()
    valid = batch["mask"].unsqueeze(-1)
    illegal_mass = probs.masked_fill(~illegal_mask, 0.0).sum(dim=-1)
    illegal_loss = illegal_mass[batch["mask"]].mean()
    next_pos_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    collision_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    distance_delta_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    loop_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    newly_seen_wall_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    if output is not None and "next_position_logits" in output:
        if args.lambda_next_pos > 0.0:
            next_pos_loss = F.cross_entropy(
                output["next_position_logits"].reshape(-1, output["next_position_logits"].shape[-1]),
                batch["next_position_target"].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        valid = batch["mask"]
        if args.lambda_collision > 0.0:
            collision_loss = F.binary_cross_entropy_with_logits(
                output["collision_logits"][valid],
                batch["collision_target"][valid].float(),
            )
        if args.lambda_dist > 0.0:
            distance_delta_loss = F.mse_loss(
                output["distance_delta"][valid],
                batch["distance_delta_target"][valid].float(),
            )
        if args.lambda_loop > 0.0:
            loop_loss = F.binary_cross_entropy_with_logits(
                output["loop_logits"][valid],
                batch["loop_target"][valid].float(),
            )
        if args.lambda_new_wall > 0.0:
            newly_seen_wall_loss = F.binary_cross_entropy_with_logits(
                output["newly_seen_wall_logits"][valid],
                batch["newly_seen_wall_target"][valid].float(),
            )

    loss = behavior_weight * behavior_loss + illegal_weight * illegal_loss
    loss = loss + float(args.lambda_relative) * relative_loss
    loss = loss + float(args.lambda_next_pos) * next_pos_loss
    loss = loss + float(args.lambda_collision) * collision_loss
    loss = loss + float(args.lambda_dist) * distance_delta_loss
    loss = loss + float(args.lambda_loop) * loop_loss
    loss = loss + float(args.lambda_new_wall) * newly_seen_wall_loss

    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets).logical_and(batch["mask"]).sum().item()
    total = batch["mask"].sum().item()
    accuracy = correct / total if total else 0.0
    return loss, accuracy, total, {
        "behavior_loss": float(behavior_loss.detach().item()),
        "illegal_loss": float(illegal_loss.detach().item()),
        "relative_loss": float(relative_loss.detach().item()),
        "next_pos_loss": float(next_pos_loss.detach().item()),
        "collision_loss": float(collision_loss.detach().item()),
        "distance_delta_loss": float(distance_delta_loss.detach().item()),
        "loop_loss": float(loop_loss.detach().item()),
        "newly_seen_wall_loss": float(newly_seen_wall_loss.detach().item()),
    }


def rollout_constraint_loss(
    model,
    rows: list,
    base_walls: dict[int, list[set[int]]],
    recovery_cache: dict[tuple[int, int], int],
    action_priors: dict,
    shortest_weight_by_task: dict[int, float],
    device,
    max_steps: int,
    recovery_weight: float,
    no_progress_weight: float,
    mask_policy: str,
    model_variant: str,
    args,
    memory_snapshots: dict[int, SubjectMazeMemoryStore] | None = None,
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
        hit_counts = Counter()
        task = int(row.task)

        true_actions = parse_json_list(row.action)
        trial_idx = int(getattr(row, "sequence_trial_index", -1))
        maze_id = int(getattr(row, "maze", 1))
        is_hidden_task = task == 4
        snapshot_store = memory_snapshots.get(trial_idx) if memory_snapshots else None
        if snapshot_store is not None:
            memory = snapshot_store.start_trial(maze_id=maze_id, task_id=task, trial_id=trial_idx)
        else:
            memory = MazeMemoryState()
            memory.reset(task_id=task, trial_id=trial_idx)
        heading = DEFAULT_HEADING
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
                "cognitive_features": torch.tensor(
                    memory.get_features(
                        current_pos=state,
                        goal_pos=goal,
                        heading=heading,
                        walls=None if is_hidden_task else walls,
                        variant=model_variant,
                        t=step_idx,
                    ).reshape(1, 1, -1),
                    dtype=torch.float32,
                    device=device,
                ),
                "trial_start": torch.tensor(
                    [[1.0 if step_idx == 0 else 0.0]],
                    dtype=torch.float32,
                    device=device,
                ),
                "prev_hit": torch.tensor([[prev_hit]], dtype=torch.float32, device=device),
                "task": torch.tensor([[task]], dtype=torch.long, device=device),
                "replan": torch.tensor([[int(row.replan)]], dtype=torch.long, device=device),
            }
            x = model.build_input(batch)
            output, hidden = model.rnn(x, hidden)
            logits = model.action_head(output).squeeze(0).squeeze(0)
            if mask_policy == "memory_soft":
                masked_logits = memory_soft_logits(logits, memory, state, goal, heading, args)
            else:
                biased_logits = human_biased_logits(logits, memory, state, goal, heading, args)
                masked_logits = mask_logits_for_policy(
                    biased_logits,
                    walls,
                    state,
                    task,
                    hit_counts,
                    mask_policy,
                )
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

            is_off_trajectory = state not in true_path or visit_counts[state] > 1
            recovery_target = IGNORE_INDEX
            if args.use_shortest_recovery_loss:
                recovery_target = shortest_recovery_action(
                    walls,
                    state,
                    goal,
                    recovery_cache,
                )
            if args.use_shortest_recovery_loss and is_off_trajectory and recovery_target != IGNORE_INDEX:
                target = soft_recovery_target(
                    walls,
                    state,
                    task,
                    recovery_target,
                    action_priors,
                    shortest_weight_by_task,
                ).to(device)
                recovery_losses.append(
                    recovery_weight
                    * soft_cross_entropy(masked_logits, target)
                )
                n_recovery_targets += 1

            raw_action = int(logits.argmax().detach().item())
            action = int(masked_logits.argmax().detach().item())
            n_prevented_illegal += int(raw_action != action)
            next_state, hit = apply_action(state, action, walls)
            if hit:
                hit_counts[(state, action)] += 1
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
            memory.update(state, action, next_state, hit, t=step_idx)
            if not is_hidden_task:
                memory.observe_local_walls(state, walls)
            heading = update_heading(heading, action, hit)
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


def rollout_metrics(
    model,
    rows: pd.DataFrame,
    base_walls,
    device,
    max_steps: int,
    mask_policy: str,
    model_variant: str,
    args,
    memory_snapshots: dict[int, SubjectMazeMemoryStore] | None = None,
) -> dict:
    model.eval()
    records = []
    actual_records = []
    with torch.no_grad():
        for row in rows.itertuples(index=False):
            walls = parse_trial_walls(row, base_walls)
            goal = int(row.goal)
            state = int(row.start)
            prev_action = START_ACTION
            prev_hit = 0.0
            prev_reward = 0.0
            hidden = None
            trial_idx = int(row.sequence_trial_index)
            maze_id = int(getattr(row, "maze", 1))
            is_hidden_task = int(row.task) == 4
            snapshot_store = memory_snapshots.get(trial_idx) if memory_snapshots else None
            if snapshot_store is not None:
                memory = snapshot_store.start_trial(maze_id=maze_id, task_id=int(row.task), trial_id=trial_idx)
            else:
                memory = MazeMemoryState()
                memory.reset(task_id=int(row.task), trial_id=trial_idx)
            visited = {state}
            hit_counts = Counter()
            n_hit = 0
            n_revisit = 0
            true_actions = parse_json_list(row.action)
            heading = DEFAULT_HEADING
            true_path = parse_json_list(row.true_path)
            true_hits = parse_json_list(row.hits)
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
                    "cognitive_features": torch.tensor(
                        memory.get_features(
                            current_pos=state,
                            goal_pos=goal,
                            heading=heading,
                            walls=None if is_hidden_task else walls,
                            variant=model_variant,
                            t=step_idx,
                        ).reshape(1, 1, -1),
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
                if mask_policy == "memory_soft":
                    masked_logits = memory_soft_logits(logits, memory, state, goal, heading, args)
                else:
                    biased_logits = human_biased_logits(logits, memory, state, goal, heading, args)
                    masked_logits = mask_logits_for_policy(
                        biased_logits,
                        walls,
                        state,
                        int(row.task),
                        hit_counts,
                        mask_policy,
                    )
                action = int(masked_logits.argmax().item())
                next_state, hit = apply_action(state, action, walls)
                if hit:
                    hit_counts[(state, action)] += 1
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
                memory.update(state, action, next_state, hit, t=step_idx)
                if not is_hidden_task:
                    memory.observe_local_walls(state, walls)
                heading = update_heading(heading, action, hit)
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
            actual_state_counts = Counter(int(x) for x in true_path)
            actual_revisits = sum(max(0, count - 1) for count in actual_state_counts.values())
            actual_steps = max(len(true_actions), 1)
            actual_records.append(
                {
                    "task": int(row.task),
                    "steps": actual_steps,
                    "shortest_steps": shortest_steps,
                    "excess": max(0, actual_steps - shortest_steps),
                    "hit_rate": sum(bool(x) for x in true_hits) / actual_steps,
                    "revisit_rate": actual_revisits / len(true_path) if true_path else 0.0,
                    "reached": 1.0,
                    "efficiency": shortest_steps / actual_steps,
                }
            )

    df = pd.DataFrame(records)
    actual_df = pd.DataFrame(actual_records)
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

    def human_similarity_frame(model_frame: pd.DataFrame, actual_frame: pd.DataFrame) -> float:
        if model_frame.empty or actual_frame.empty:
            return 0.0
        actual_steps = max(float(actual_frame["steps"].mean()), 1.0)
        actual_excess = max(float(actual_frame["excess"].mean()), 1.0)
        distance = (
            1.2 * abs(float(model_frame["steps"].mean()) - actual_steps) / actual_steps
            + 1.0 * abs(float(model_frame["excess"].mean()) - float(actual_frame["excess"].mean())) / actual_excess
            + 1.0 * abs(float(model_frame["hit_rate"].mean()) - float(actual_frame["hit_rate"].mean()))
            + 0.8 * abs(float(model_frame["revisit_rate"].mean()) - float(actual_frame["revisit_rate"].mean()))
            + 0.6 * abs(float(model_frame["efficiency"].mean()) - float(actual_frame["efficiency"].mean()))
            + 1.5 * max(0.0, float(actual_frame["reached"].mean()) - float(model_frame["reached"].mean()))
        )
        return 1.0 / (1.0 + distance)

    task_scores = {
        int(task): score_frame(task_df)
        for task, task_df in df.groupby("task", sort=True)
    }
    task_human_similarity = {
        int(task): human_similarity_frame(task_df, actual_df[actual_df["task"].eq(task)])
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
        "human_similarity_score": float(np.mean(list(task_human_similarity.values()))),
        "trial_weighted_rollout_score": overall["score"],
        "reached_goal_rate": balanced_reached,
        "mean_excess_steps": balanced_excess,
        "mean_hit_rate": balanced_hit,
        "mean_revisit_rate": balanced_revisit,
        "mean_efficiency": balanced_efficiency,
    }
    for task, values in task_scores.items():
        result[f"task{task}_rollout_score"] = values["score"]
        result[f"task{task}_human_similarity_score"] = task_human_similarity.get(task, 0.0)
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
    total_relative_loss = 0.0
    total_next_pos_loss = 0.0
    total_collision_loss = 0.0
    total_distance_delta_loss = 0.0
    total_loop_loss = 0.0
    total_newly_seen_wall_loss = 0.0
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
                args=args,
            )
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * n_steps
        total_behavior_loss += stats["behavior_loss"] * n_steps
        total_illegal_loss += stats["illegal_loss"] * n_steps
        total_relative_loss += stats["relative_loss"] * n_steps
        total_next_pos_loss += stats["next_pos_loss"] * n_steps
        total_collision_loss += stats["collision_loss"] * n_steps
        total_distance_delta_loss += stats["distance_delta_loss"] * n_steps
        total_loop_loss += stats["loop_loss"] * n_steps
        total_newly_seen_wall_loss += stats["newly_seen_wall_loss"] * n_steps
        total_correct_weighted += accuracy * n_steps
        total_steps += n_steps

    return {
        "loss": total_loss / total_steps if total_steps else 0.0,
        "behavior_loss": total_behavior_loss / total_steps if total_steps else 0.0,
        "illegal_loss": total_illegal_loss / total_steps if total_steps else 0.0,
        "relative_loss": total_relative_loss / total_steps if total_steps else 0.0,
        "next_pos_loss": total_next_pos_loss / total_steps if total_steps else 0.0,
        "collision_loss": total_collision_loss / total_steps if total_steps else 0.0,
        "distance_delta_loss": total_distance_delta_loss / total_steps if total_steps else 0.0,
        "loop_loss": total_loop_loss / total_steps if total_steps else 0.0,
        "newly_seen_wall_loss": total_newly_seen_wall_loss / total_steps if total_steps else 0.0,
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
    memory_snapshots: dict[int, SubjectMazeMemoryStore] | None = None,
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
        args.action_priors,
        args.shortest_weight_by_task,
        device,
        max_steps=args.rollout_constraint_max_steps,
        recovery_weight=args.off_trajectory_recovery_weight,
        no_progress_weight=args.rollout_no_progress_weight,
        mask_policy=args.rollout_mask_policy,
        model_variant=args.model_variant,
        args=args,
        memory_snapshots=memory_snapshots,
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
    parser.add_argument("--illegal-weight", type=float, default=0.0)
    parser.add_argument(
        "--model-variant",
        default="A",
        choices=["A", "B", "C", "D", "E"],
        help="A=baseline, B=local visual/topology, C=explicit memory, D=heading/relative action.",
    )
    parser.add_argument("--lambda-relative", type=float, default=0.0)
    parser.add_argument("--lambda-next-pos", type=float, default=0.0)
    parser.add_argument("--lambda-collision", type=float, default=0.0)
    parser.add_argument("--lambda-dist", type=float, default=0.0)
    parser.add_argument("--lambda-loop", type=float, default=0.0)
    parser.add_argument("--lambda-new-wall", type=float, default=0.0)
    parser.add_argument("--rollout-weight", type=float, default=0.08)
    parser.add_argument("--rollout-trials-per-epoch", type=int, default=24)
    parser.add_argument("--rollout-constraint-max-steps", type=int, default=40)
    parser.add_argument("--rollout-eval-max-steps", type=int, default=80)
    parser.add_argument("--off-trajectory-recovery-weight", type=float, default=1.0)
    parser.add_argument(
        "--shortest-recovery-weight-by-task",
        default="1:0.50,2:0.70,3:0.75,4:0.35",
        help="Comma-separated task:weight map for soft recovery targets.",
    )
    parser.add_argument(
        "--rollout-mask-policy",
        default="memory_soft",
        choices=["none", "hard", "repeated_hit", "task_specific", "memory_soft"],
    )
    parser.add_argument("--use-shortest-recovery-loss", action="store_true")
    parser.add_argument("--use-human-bias", action="store_true")
    parser.add_argument("--known-wall-penalty", type=float, default=3.0)
    parser.add_argument("--revisit-penalty", type=float, default=0.15)
    parser.add_argument("--loop-penalty", type=float, default=0.5)
    parser.add_argument("--unexplored-bonus", type=float, default=0.25)
    parser.add_argument("--goal-progress-bonus", type=float, default=0.15)
    parser.add_argument("--backtrack-penalty", type=float, default=0.1)
    parser.add_argument("--rollout-no-progress-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        default="human_similarity",
        choices=["human_similarity", "rollout_score", "val_accuracy"],
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

    # Build memory snapshots first — datasets and rollout functions all depend on them.
    base_walls = load_base_walls(args.maze_wall)
    all_rows = list(prepare_subject_rows(args, "all").itertuples(index=False))
    memory_snapshots = build_incremental_memory_stores(all_rows, base_walls)
    print(f"Built memory snapshots for {len(memory_snapshots)} trials.")
    train_rows = list(prepare_subject_rows(args, "train").itertuples(index=False))
    val_rows = prepare_subject_rows(args, "val")

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
        model_variant=args.model_variant,
        memory_snapshots=memory_snapshots,
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
        model_variant=args.model_variant,
        memory_snapshots=memory_snapshots,
    )
    recovery_cache: dict[tuple[int, int], int] = {}
    args.action_priors = build_action_priors(train_rows)
    args.shortest_weight_by_task = parse_task_float_map(
        args.shortest_recovery_weight_by_task,
        default=0.6,
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

    cognitive_feature_names = feature_names_for_variant(args.model_variant)
    include_maze_wall = False
    use_auxiliary_heads = args.model_variant == "E" and any(
        value > 0.0
        for value in [
            args.lambda_next_pos,
            args.lambda_collision,
            args.lambda_dist,
            args.lambda_loop,
            args.lambda_new_wall,
        ]
    )
    model = MinimalMazeActionRNN(
        hidden_dim=args.hidden_dim,
        cognitive_feature_dim=len(cognitive_feature_names),
        include_maze_wall=include_maze_wall,
        use_relative_action_head=args.model_variant in {"D", "E"} and args.lambda_relative > 0,
        use_auxiliary_heads=use_auxiliary_heads,
    ).to(device)
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
            memory_snapshots=memory_snapshots,
        )
        val_stats = run_behavior_epoch(model, val_loader, device, args, optimizer=None)
        rollout_stats = rollout_metrics(
            model,
            val_rows,
            base_walls,
            device,
            max_steps=args.rollout_eval_max_steps,
            mask_policy=args.rollout_mask_policy,
            model_variant=args.model_variant,
            args=args,
            memory_snapshots=memory_snapshots,
        )

        selection_value = (
            rollout_stats["human_similarity_score"]
            if args.selection_metric == "human_similarity"
            else (
                rollout_stats["rollout_score"]
                if args.selection_metric == "rollout_score"
                else val_stats["accuracy"]
            )
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "train_behavior_loss": round(train_stats["behavior_loss"], 6),
            "train_illegal_loss": round(train_stats["illegal_loss"], 6),
            "train_relative_loss": round(train_stats["relative_loss"], 6),
            "train_next_pos_loss": round(train_stats["next_pos_loss"], 6),
            "train_collision_loss": round(train_stats["collision_loss"], 6),
            "train_distance_delta_loss": round(train_stats["distance_delta_loss"], 6),
            "train_loop_loss": round(train_stats["loop_loss"], 6),
            "train_newly_seen_wall_loss": round(train_stats["newly_seen_wall_loss"], 6),
            "train_accuracy": round(train_stats["accuracy"], 6),
            "train_steps": train_stats["steps"],
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in rollout_train_stats.items()
            },
            "val_loss": round(val_stats["loss"], 6),
            "val_behavior_loss": round(val_stats["behavior_loss"], 6),
            "val_illegal_loss": round(val_stats["illegal_loss"], 6),
            "val_relative_loss": round(val_stats["relative_loss"], 6),
            "val_next_pos_loss": round(val_stats["next_pos_loss"], 6),
            "val_collision_loss": round(val_stats["collision_loss"], 6),
            "val_distance_delta_loss": round(val_stats["distance_delta_loss"], 6),
            "val_loop_loss": round(val_stats["loop_loss"], 6),
            "val_newly_seen_wall_loss": round(val_stats["newly_seen_wall_loss"], 6),
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
                        "rollout_mask_policy": args.rollout_mask_policy,
                        "training_recipe": "minimal_human_mask_soft_recovery",
                        "model_variant": args.model_variant,
                        "cognitive_feature_dim": len(cognitive_feature_names),
                        "cognitive_feature_names": cognitive_feature_names,
                        "include_maze_wall": include_maze_wall,
                        "use_relative_action_head": model.use_relative_action_head,
                        "use_auxiliary_heads": model.use_auxiliary_heads,
                        "action_encoding": ACTION_ENCODING,
                        "action_deltas": ACTION_DELTAS,
                        "wall_column_order": WALL_COLUMNS,
                        "heading_first_step_ignored": True,
                        "hard_mask_used": args.rollout_mask_policy in {"hard", "task_specific"},
                        "shortest_recovery_oracle_used": bool(args.use_shortest_recovery_loss),
                    },
                    "model_kind": "minimal",
                    "training_recipe": "minimal_human_mask_soft_recovery",
                    "rollout_mask_policy": args.rollout_mask_policy,
                    "model_variant": args.model_variant,
                    "cognitive_feature_dim": len(cognitive_feature_names),
                    "cognitive_feature_names": cognitive_feature_names,
                    "include_maze_wall": include_maze_wall,
                    "use_relative_action_head": model.use_relative_action_head,
                    "use_auxiliary_heads": model.use_auxiliary_heads,
                    "action_encoding": ACTION_ENCODING,
                    "action_deltas": ACTION_DELTAS,
                    "wall_column_order": WALL_COLUMNS,
                    "heading_first_step_ignored": True,
                    "hard_mask_used": args.rollout_mask_policy in {"hard", "task_specific"},
                    "shortest_recovery_oracle_used": bool(args.use_shortest_recovery_loss),
                    "selection_metric": args.selection_metric,
                    "best_selection_value": best_value,
                    "best_rollout_metrics": rollout_stats,
                    "input_features": [
                        "current_state",
                        "prev_action",
                        "goal",
                        "prev_reward",
                        "trial_start_flag",
                        "wall_hit",
                    ]
                    + (["maze_wall"] if include_maze_wall else [])
                    + cognitive_feature_names
                    + (["relative_action_head"] if model.use_relative_action_head else []),
                    "grouping": grouping_rows,
                },
                best_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train acc {train_stats['accuracy']:.4f} | "
            f"val acc {val_stats['accuracy']:.4f} | "
            f"rollout score {rollout_stats['rollout_score']:.4f}, "
            f"human {rollout_stats['human_similarity_score']:.4f}, "
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
