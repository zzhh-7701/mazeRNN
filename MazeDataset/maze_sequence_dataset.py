from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD_ACTION = 4
START_ACTION = 5
IGNORE_INDEX = -100


def parse_json_list(value):
    if pd.isna(value) or value == "":
        return []
    return json.loads(value)


def subject_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(df["subid"].dropna().unique()))
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_subjects = set(subjects[:n_val])
    train_df = df[~df["subid"].isin(val_subjects)].copy()
    val_df = df[df["subid"].isin(val_subjects)].copy()
    return train_df, val_df


class MazeSequenceDataset(Dataset):
    """Trial-level maze behavior sequences for next-action prediction."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        max_trials: int | None = None,
        valid_only: bool = False,
        diagnostics_path: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)

        if valid_only:
            if diagnostics_path is None:
                raise ValueError("diagnostics_path is required when valid_only=True")
            df = self._filter_valid_trials(df, Path(diagnostics_path))

        train_df, val_df = subject_split(df, val_ratio=val_ratio, seed=seed)
        if split == "train":
            df = train_df
        elif split in {"val", "valid", "validation"}:
            df = val_df
        elif split == "all":
            pass
        else:
            raise ValueError(f"Unknown split: {split}")

        if max_trials is not None:
            df = df.head(max_trials)

        self.samples = self._build_samples(df)

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

    @staticmethod
    def _build_samples(df: pd.DataFrame):
        samples = []
        for row in df.itertuples(index=False):
            action = parse_json_list(getattr(row, "action"))
            true_path = parse_json_list(getattr(row, "true_path"))

            if len(action) == 0 or len(true_path) < len(action):
                continue

            states = true_path[: len(action)]
            prev_actions = [START_ACTION] + action[:-1]

            samples.append(
                {
                    "state": torch.tensor(states, dtype=torch.long),
                    "goal": torch.full((len(action),), int(row.goal), dtype=torch.long),
                    "prev_action": torch.tensor(prev_actions, dtype=torch.long),
                    "task": torch.full((len(action),), int(row.task), dtype=torch.long),
                    "replan": torch.full((len(action),), int(row.replan), dtype=torch.long),
                    "target": torch.tensor(action, dtype=torch.long),
                    "subid": int(row.subid),
                    "trial": int(row.trial),
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_maze_sequences(batch):
    states = pad_sequence([x["state"] for x in batch], batch_first=True, padding_value=0)
    goals = pad_sequence([x["goal"] for x in batch], batch_first=True, padding_value=0)
    prev_actions = pad_sequence(
        [x["prev_action"] for x in batch],
        batch_first=True,
        padding_value=PAD_ACTION,
    )
    tasks = pad_sequence([x["task"] for x in batch], batch_first=True, padding_value=0)
    replans = pad_sequence([x["replan"] for x in batch], batch_first=True, padding_value=0)
    targets = pad_sequence(
        [x["target"] for x in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    lengths = torch.tensor([len(x["target"]) for x in batch], dtype=torch.long)
    mask = targets.ne(IGNORE_INDEX)

    return {
        "state": states,
        "goal": goals,
        "prev_action": prev_actions,
        "task": tasks,
        "replan": replans,
        "target": targets,
        "lengths": lengths,
        "mask": mask,
    }
