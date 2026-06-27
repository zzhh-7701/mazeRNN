from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .memory_state import (
    ACTION_DELTAS,
    DEFAULT_HEADING,
    MazeMemoryState,
    SubjectMazeMemoryStore,
    compute_relative_action,
    feature_names_for_variant,
    local_wall_flags,
    update_heading,
)
from .maze_sequence_dataset import IGNORE_INDEX, PAD_ACTION, START_ACTION, parse_json_list


WALL_DIRECTIONS = ("l", "u", "r", "d")


def _safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def _parse_wall_value(value):
    if pd.isna(value) or value == "":
        return None
    return json.loads(value)


def _load_base_walls(maze_wall_path: str | Path) -> dict[int, list[set[int]]]:
    wall_df = pd.read_csv(maze_wall_path)
    walls = {}
    for row in wall_df.itertuples(index=False):
        walls[int(row.maze)] = [
            set(parse_json_list(getattr(row, f"walls_{direction}")))
            for direction in WALL_DIRECTIONS
        ]
    return walls


def _row_walls(row, base_walls: dict[int, list[set[int]]]) -> list[set[int]]:
    override = _parse_wall_value(getattr(row, "maze_wall"))
    if override:
        return [set(direction_walls) for direction_walls in override]
    return base_walls[int(row.maze)]


def _wall_features(walls: list[set[int]], state: int) -> list[float]:
    return [1.0 if state in direction_walls else 0.0 for direction_walls in walls]


def _goal_distance_delta(state: int, next_state: int, goal: int) -> float:
    current = abs((state // 7) - (goal // 7)) + abs((state % 7) - (goal % 7))
    nxt = abs((next_state // 7) - (goal // 7)) + abs((next_state % 7) - (goal % 7))
    return float(current - nxt)


def interspersed_trial_split(
    n_trials: int,
    split: str,
    val_every: int = 4,
    val_offset: int = 3,
) -> np.ndarray:
    """Return a boolean mask over ordered trials using an interspersed split."""
    if val_every < 2:
        raise ValueError("val_every must be at least 2")
    if not 0 <= val_offset < val_every:
        raise ValueError("val_offset must be in [0, val_every)")

    trial_positions = np.arange(n_trials)
    val_mask = trial_positions % val_every == val_offset

    normalized_split = split.lower()
    if normalized_split == "train":
        return ~val_mask
    if normalized_split in {"val", "valid", "validation"}:
        return val_mask
    if normalized_split == "all":
        return np.ones(n_trials, dtype=bool)
    raise ValueError(f"Unknown split: {split}")


class SubjectSequenceDataset(Dataset):
    """Within-subject step sequences that preserve trial order across learning."""

    def __init__(
        self,
        csv_path: str | Path,
        maze_wall_path: str | Path,
        subject_id: int,
        split: str = "train",
        val_every: int = 4,
        val_offset: int = 3,
        valid_only: bool = False,
        diagnostics_path: str | Path | None = None,
        max_trials: int | None = None,
        model_variant: str = "A",
        memory_snapshots: dict | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.maze_wall_path = Path(maze_wall_path)
        self.subject_id = int(subject_id)
        self.split = split
        self.val_every = val_every
        self.val_offset = val_offset
        self.model_variant = model_variant.upper()
        self.cognitive_feature_names = feature_names_for_variant(self.model_variant)
        self.cognitive_feature_dim = len(self.cognitive_feature_names)

        df = pd.read_csv(self.csv_path)
        if valid_only:
            if diagnostics_path is None:
                raise ValueError("diagnostics_path is required when valid_only=True")
            df = self._filter_valid_trials(df, Path(diagnostics_path))

        df = df[df["subid"].eq(self.subject_id)].copy()
        if df.empty:
            raise ValueError(f"No rows found for subject_id={self.subject_id}")

        sort_columns = ["day", "block", "trial", "replan", "createdat"]
        df = df.sort_values(sort_columns, na_position="last").reset_index(drop=True)
        if max_trials is not None:
            df = df.head(max_trials)

        keep_mask = interspersed_trial_split(
            len(df),
            split=split,
            val_every=val_every,
            val_offset=val_offset,
        )
        df["sequence_trial_index"] = np.arange(len(df))
        df["is_validation_trial"] = np.arange(len(df)) % val_every == val_offset
        df = df.loc[keep_mask].copy()

        self.base_walls = _load_base_walls(self.maze_wall_path)
        self.samples = self._build_contiguous_samples(df, memory_snapshots=memory_snapshots)

    @staticmethod
    def _filter_valid_trials(df: pd.DataFrame, diagnostics_path: Path) -> pd.DataFrame:
        diagnostics = pd.read_csv(diagnostics_path)
        keep_cols = [
            "true_path_end_ok",
            "hit_consistent",
            "short_path_valid_in_maze",
            "short_path_is_shortest",
        ]
        valid_mask = diagnostics[keep_cols].eq(1).all(axis=1)
        return df.loc[valid_mask.to_numpy()].copy()

    def _build_contiguous_samples(self, df: pd.DataFrame, memory_snapshots: dict | None = None):
        samples = []
        current_rows = []
        previous_index = None
        # Always initialise explicitly so _append_sample sees a clean store.
        self._long_term_memory = SubjectMazeMemoryStore()

        for row in df.itertuples(index=False):
            trial_index = int(row.sequence_trial_index)

            if previous_index is None:
                # First row: warm up from snapshot if available.
                if memory_snapshots is not None and trial_index in memory_snapshots:
                    self._long_term_memory = memory_snapshots[trial_index].copy()
            elif trial_index != previous_index + 1:
                # Chunk boundary: flush current chunk then warm up for next.
                self._append_sample(samples, current_rows)
                current_rows = []
                if memory_snapshots is not None:
                    self._long_term_memory = memory_snapshots.get(
                        trial_index, SubjectMazeMemoryStore()
                    ).copy()

            current_rows.append(row)
            previous_index = trial_index

        self._append_sample(samples, current_rows)
        return samples

    def _append_sample(self, samples: list, rows: list) -> None:
        if not rows:
            return

        states = []
        goals = []
        prev_actions = []
        tasks = []
        replans = []
        trial_starts = []
        prev_hits = []
        prev_rewards = []
        wall_features = []
        cognitive_features = []
        relative_targets = []
        headings = []
        next_positions = []
        collision_targets = []
        distance_delta_targets = []
        loop_targets = []
        newly_seen_wall_targets = []
        targets = []
        trial_indices = []
        step_indices = []

        for row in rows:
            action = parse_json_list(getattr(row, "action"))
            true_path = parse_json_list(getattr(row, "true_path"))
            hits = parse_json_list(getattr(row, "hits"))

            if len(action) == 0 or len(true_path) < len(action):
                continue

            row_states = true_path[: len(action)]
            row_prev_actions = [START_ACTION] + action[:-1]
            row_prev_hits = [0] + [int(bool(hit)) for hit in hits[:-1]]
            row_prev_rewards = [0.0]
            for step_idx in range(1, len(action)):
                previous_hit = bool(hits[step_idx - 1]) if step_idx - 1 < len(hits) else False
                previous_next_state = true_path[step_idx]
                if previous_hit:
                    row_prev_rewards.append(-1.0)
                elif previous_next_state == int(row.goal):
                    row_prev_rewards.append(1.0)
                else:
                    row_prev_rewards.append(0.0)

            walls = _row_walls(row, self.base_walls)
            if not hasattr(self, "_long_term_memory"):
                self._long_term_memory = SubjectMazeMemoryStore()
            task_id = _safe_int(row.task)
            # Task 4 is a hidden maze: no local wall visibility, walls learned only via collisions.
            is_hidden_task = task_id == 4
            memory = self._long_term_memory.start_trial(
                maze_id=_safe_int(row.maze),
                task_id=task_id,
                trial_id=int(row.sequence_trial_index),
            )
            heading = DEFAULT_HEADING
            for step_idx, state in enumerate(row_states):
                state = int(state)
                target_action = int(action[step_idx])
                next_state = (
                    int(true_path[step_idx + 1])
                    if step_idx + 1 < len(true_path)
                    else state
                )
                hit = bool(hits[step_idx]) if step_idx < len(hits) else next_state == state
                local_visual = None if is_hidden_task else local_wall_flags(walls, state)
                would_loop = float(next_state in memory.recent_positions)
                known_wall_before = float(memory.known_wall_mask[state, target_action])
                newly_seen_wall = float(hit and known_wall_before <= 0.0)
                states.append(int(state))
                goals.append(_safe_int(row.goal))
                prev_actions.append(int(row_prev_actions[step_idx]))
                tasks.append(task_id)
                replans.append(_safe_int(row.replan))
                trial_starts.append(1 if step_idx == 0 else 0)
                prev_hits.append(int(row_prev_hits[step_idx]))
                prev_rewards.append(float(row_prev_rewards[step_idx]))
                wall_features.append(_wall_features(walls, int(state)))
                cognitive_features.append(
                    memory.get_features(
                        current_pos=state,
                        goal_pos=_safe_int(row.goal),
                        heading=heading,
                        local_visual=local_visual,
                        variant=self.model_variant,
                        t=step_idx,
                    )
                )
                relative_targets.append(
                    IGNORE_INDEX
                    if heading == DEFAULT_HEADING
                    else compute_relative_action(heading, target_action)
                )
                headings.append(int(heading))
                next_positions.append(int(next_state))
                collision_targets.append(float(hit))
                distance_delta_targets.append(
                    _goal_distance_delta(state, next_state, _safe_int(row.goal))
                )
                loop_targets.append(would_loop)
                newly_seen_wall_targets.append(newly_seen_wall)
                targets.append(target_action)
                trial_indices.append(int(row.sequence_trial_index))
                step_indices.append(step_idx)
                memory.update(
                    current_pos=state,
                    action=target_action,
                    next_pos=next_state,
                    collision=hit,
                    t=step_idx,
                )
                if not is_hidden_task:
                    memory.observe_local_walls(state, walls)
                heading = update_heading(heading, target_action, hit)
            self._long_term_memory.finish_trial(_safe_int(row.maze), memory)

        if not targets:
            return

        samples.append(
            {
                "state": torch.tensor(states, dtype=torch.long),
                "goal": torch.tensor(goals, dtype=torch.long),
                "prev_action": torch.tensor(prev_actions, dtype=torch.long),
                "task": torch.tensor(tasks, dtype=torch.long),
                "replan": torch.tensor(replans, dtype=torch.long),
                "trial_start": torch.tensor(trial_starts, dtype=torch.float32),
                "prev_hit": torch.tensor(prev_hits, dtype=torch.float32),
                "prev_reward": torch.tensor(prev_rewards, dtype=torch.float32),
                "maze_wall": torch.tensor(wall_features, dtype=torch.float32),
                "cognitive_features": torch.tensor(
                    np.asarray(cognitive_features, dtype=np.float32),
                    dtype=torch.float32,
                ),
                "relative_target": torch.tensor(relative_targets, dtype=torch.long),
                "heading": torch.tensor(headings, dtype=torch.long),
                "next_position_target": torch.tensor(next_positions, dtype=torch.long),
                "collision_target": torch.tensor(collision_targets, dtype=torch.float32),
                "distance_delta_target": torch.tensor(distance_delta_targets, dtype=torch.float32),
                "loop_target": torch.tensor(loop_targets, dtype=torch.float32),
                "newly_seen_wall_target": torch.tensor(newly_seen_wall_targets, dtype=torch.float32),
                "target": torch.tensor(targets, dtype=torch.long),
                "trial_index": torch.tensor(trial_indices, dtype=torch.long),
                "step_index": torch.tensor(step_indices, dtype=torch.long),
                "subid": self.subject_id,
                "first_trial_index": int(trial_indices[0]),
                "last_trial_index": int(trial_indices[-1]),
            }
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FullSubjectSequenceDataset(Dataset):
    """One full chronological subject sequence with interspersed train/val masks."""

    def __init__(
        self,
        csv_path: str | Path,
        maze_wall_path: str | Path,
        subject_id: int,
        val_every: int = 4,
        val_offset: int = 3,
        valid_only: bool = False,
        diagnostics_path: str | Path | None = None,
        max_trials: int | None = None,
        model_variant: str = "A",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.maze_wall_path = Path(maze_wall_path)
        self.subject_id = int(subject_id)
        self.val_every = val_every
        self.val_offset = val_offset
        self.model_variant = model_variant.upper()
        self.cognitive_feature_names = feature_names_for_variant(self.model_variant)
        self.cognitive_feature_dim = len(self.cognitive_feature_names)

        df = pd.read_csv(self.csv_path)
        if valid_only:
            if diagnostics_path is None:
                raise ValueError("diagnostics_path is required when valid_only=True")
            df = SubjectSequenceDataset._filter_valid_trials(df, Path(diagnostics_path))

        df = df[df["subid"].eq(self.subject_id)].copy()
        if df.empty:
            raise ValueError(f"No rows found for subject_id={self.subject_id}")

        df = df.sort_values(["day", "block", "trial", "replan", "createdat"], na_position="last")
        df = df.reset_index(drop=True)
        if max_trials is not None:
            df = df.head(max_trials)

        df["sequence_trial_index"] = np.arange(len(df))
        df["is_validation_trial"] = np.arange(len(df)) % val_every == val_offset
        self.trial_table = df.copy()
        self.base_walls = _load_base_walls(self.maze_wall_path)
        self.sample = self._build_full_sample(df)

    def _build_full_sample(self, df: pd.DataFrame):
        states = []
        goals = []
        prev_actions = []
        tasks = []
        replans = []
        trial_starts = []
        prev_hits = []
        prev_rewards = []
        wall_features = []
        cognitive_features = []
        relative_targets = []
        headings = []
        next_positions = []
        collision_targets = []
        distance_delta_targets = []
        loop_targets = []
        newly_seen_wall_targets = []
        targets = []
        trial_indices = []
        step_indices = []
        val_masks = []

        for row in df.itertuples(index=False):
            action = parse_json_list(getattr(row, "action"))
            true_path = parse_json_list(getattr(row, "true_path"))
            hits = parse_json_list(getattr(row, "hits"))
            if len(action) == 0 or len(true_path) < len(action):
                continue

            row_states = true_path[: len(action)]
            row_prev_actions = [START_ACTION] + action[:-1]
            row_prev_hits = [0] + [int(bool(hit)) for hit in hits[:-1]]
            row_prev_rewards = [0.0]
            for step_idx in range(1, len(action)):
                previous_hit = bool(hits[step_idx - 1]) if step_idx - 1 < len(hits) else False
                previous_next_state = true_path[step_idx]
                if previous_hit:
                    row_prev_rewards.append(-1.0)
                elif previous_next_state == int(row.goal):
                    row_prev_rewards.append(1.0)
                else:
                    row_prev_rewards.append(0.0)

            walls = _row_walls(row, self.base_walls)
            if not hasattr(self, "_long_term_memory"):
                self._long_term_memory = SubjectMazeMemoryStore()
            task_id = _safe_int(row.task)
            is_hidden_task = task_id == 4
            memory = self._long_term_memory.start_trial(
                maze_id=_safe_int(row.maze),
                task_id=task_id,
                trial_id=int(row.sequence_trial_index),
            )
            heading = DEFAULT_HEADING
            for step_idx, state in enumerate(row_states):
                state = int(state)
                target_action = int(action[step_idx])
                next_state = (
                    int(true_path[step_idx + 1])
                    if step_idx + 1 < len(true_path)
                    else state
                )
                hit = bool(hits[step_idx]) if step_idx < len(hits) else next_state == state
                local_visual = None if is_hidden_task else local_wall_flags(walls, state)
                would_loop = float(next_state in memory.recent_positions)
                known_wall_before = float(memory.known_wall_mask[state, target_action])
                newly_seen_wall = float(hit and known_wall_before <= 0.0)
                states.append(int(state))
                goals.append(_safe_int(row.goal))
                prev_actions.append(int(row_prev_actions[step_idx]))
                tasks.append(task_id)
                replans.append(_safe_int(row.replan))
                trial_starts.append(1 if step_idx == 0 else 0)
                prev_hits.append(int(row_prev_hits[step_idx]))
                prev_rewards.append(float(row_prev_rewards[step_idx]))
                wall_features.append(_wall_features(walls, int(state)))
                cognitive_features.append(
                    memory.get_features(
                        current_pos=state,
                        goal_pos=_safe_int(row.goal),
                        heading=heading,
                        local_visual=local_visual,
                        variant=self.model_variant,
                        t=step_idx,
                    )
                )
                relative_targets.append(
                    IGNORE_INDEX
                    if heading == DEFAULT_HEADING
                    else compute_relative_action(heading, target_action)
                )
                headings.append(int(heading))
                next_positions.append(int(next_state))
                collision_targets.append(float(hit))
                distance_delta_targets.append(
                    _goal_distance_delta(state, next_state, _safe_int(row.goal))
                )
                loop_targets.append(would_loop)
                newly_seen_wall_targets.append(newly_seen_wall)
                targets.append(target_action)
                trial_indices.append(int(row.sequence_trial_index))
                step_indices.append(step_idx)
                val_masks.append(bool(row.is_validation_trial))
                memory.update(
                    current_pos=state,
                    action=target_action,
                    next_pos=next_state,
                    collision=hit,
                    t=step_idx,
                )
                if not is_hidden_task:
                    memory.observe_local_walls(state, walls)
                heading = update_heading(heading, target_action, hit)
            self._long_term_memory.finish_trial(_safe_int(row.maze), memory)

        val_mask = torch.tensor(val_masks, dtype=torch.bool)
        train_mask = ~val_mask
        return {
            "state": torch.tensor(states, dtype=torch.long),
            "goal": torch.tensor(goals, dtype=torch.long),
            "prev_action": torch.tensor(prev_actions, dtype=torch.long),
            "task": torch.tensor(tasks, dtype=torch.long),
            "replan": torch.tensor(replans, dtype=torch.long),
            "trial_start": torch.tensor(trial_starts, dtype=torch.float32),
            "prev_hit": torch.tensor(prev_hits, dtype=torch.float32),
            "prev_reward": torch.tensor(prev_rewards, dtype=torch.float32),
            "maze_wall": torch.tensor(wall_features, dtype=torch.float32),
            "cognitive_features": torch.tensor(
                np.asarray(cognitive_features, dtype=np.float32),
                dtype=torch.float32,
            ),
            "relative_target": torch.tensor(relative_targets, dtype=torch.long),
            "heading": torch.tensor(headings, dtype=torch.long),
            "next_position_target": torch.tensor(next_positions, dtype=torch.long),
            "collision_target": torch.tensor(collision_targets, dtype=torch.float32),
            "distance_delta_target": torch.tensor(distance_delta_targets, dtype=torch.float32),
            "loop_target": torch.tensor(loop_targets, dtype=torch.float32),
            "newly_seen_wall_target": torch.tensor(newly_seen_wall_targets, dtype=torch.float32),
            "target": torch.tensor(targets, dtype=torch.long),
            "trial_index": torch.tensor(trial_indices, dtype=torch.long),
            "step_index": torch.tensor(step_indices, dtype=torch.long),
            "mask": torch.ones(len(targets), dtype=torch.bool),
            "train_mask": train_mask,
            "val_mask": val_mask,
            "subid": self.subject_id,
            "first_trial_index": int(trial_indices[0]),
            "last_trial_index": int(trial_indices[-1]),
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.sample


def collate_subject_sequences(batch):
    states = pad_sequence([x["state"] for x in batch], batch_first=True, padding_value=0)
    goals = pad_sequence([x["goal"] for x in batch], batch_first=True, padding_value=0)
    prev_actions = pad_sequence(
        [x["prev_action"] for x in batch],
        batch_first=True,
        padding_value=PAD_ACTION,
    )
    tasks = pad_sequence([x["task"] for x in batch], batch_first=True, padding_value=0)
    replans = pad_sequence([x["replan"] for x in batch], batch_first=True, padding_value=0)
    trial_starts = pad_sequence(
        [x["trial_start"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    prev_hits = pad_sequence(
        [x["prev_hit"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    prev_rewards = pad_sequence(
        [x["prev_reward"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    maze_walls = pad_sequence(
        [x["maze_wall"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    cognitive_features = pad_sequence(
        [x["cognitive_features"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    relative_targets = pad_sequence(
        [x["relative_target"] for x in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    headings = pad_sequence(
        [x["heading"] for x in batch],
        batch_first=True,
        padding_value=0,
    )
    next_position_targets = pad_sequence(
        [x["next_position_target"] for x in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    collision_targets = pad_sequence(
        [x["collision_target"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    distance_delta_targets = pad_sequence(
        [x["distance_delta_target"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    loop_targets = pad_sequence(
        [x["loop_target"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    newly_seen_wall_targets = pad_sequence(
        [x["newly_seen_wall_target"] for x in batch],
        batch_first=True,
        padding_value=0.0,
    )
    targets = pad_sequence(
        [x["target"] for x in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    trial_indices = pad_sequence(
        [x["trial_index"] for x in batch],
        batch_first=True,
        padding_value=-1,
    )
    step_indices = pad_sequence(
        [x["step_index"] for x in batch],
        batch_first=True,
        padding_value=-1,
    )
    lengths = torch.tensor([len(x["target"]) for x in batch], dtype=torch.long)
    mask = targets.ne(IGNORE_INDEX)

    return {
        "state": states,
        "goal": goals,
        "prev_action": prev_actions,
        "task": tasks,
        "replan": replans,
        "trial_start": trial_starts,
        "prev_hit": prev_hits,
        "prev_reward": prev_rewards,
        "maze_wall": maze_walls,
        "cognitive_features": cognitive_features,
        "relative_target": relative_targets,
        "heading": headings,
        "next_position_target": next_position_targets,
        "collision_target": collision_targets,
        "distance_delta_target": distance_delta_targets,
        "loop_target": loop_targets,
        "newly_seen_wall_target": newly_seen_wall_targets,
        "target": targets,
        "trial_index": trial_indices,
        "step_index": step_indices,
        "lengths": lengths,
        "mask": mask,
        "subid": torch.tensor([x["subid"] for x in batch], dtype=torch.long),
        "first_trial_index": torch.tensor(
            [x["first_trial_index"] for x in batch],
            dtype=torch.long,
        ),
        "last_trial_index": torch.tensor(
            [x["last_trial_index"] for x in batch],
            dtype=torch.long,
        ),
    }
