from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from MazeDataset import SubjectSequenceDataset, collate_subject_sequences
from MazeDataset.maze_sequence_dataset import IGNORE_INDEX, START_ACTION
from MazeDataset.subject_sequence_dataset import _load_base_walls, _wall_features
from MazeRNNAgent import MinimalMazeActionRNN
from MazeTrainer.train_action_rnn import move_batch_to_device


ACTION_DELTAS = {
    0: -7,
    1: -1,
    2: 7,
    3: 1,
}


def legal_actions(walls: list[set[int]], state: int) -> list[int]:
    return [action for action, blocked in enumerate(walls) if state not in blocked]


def apply_action(state: int, action: int, walls: list[set[int]]) -> tuple[int, bool]:
    if state in walls[action]:
        return state, True
    next_state = state + ACTION_DELTAS[action]
    if not 0 <= next_state < 49:
        return state, True
    return next_state, False


def shortest_recovery_action(
    walls: list[set[int]],
    state: int,
    goal: int,
    cache: dict[tuple[int, int], int],
) -> int:
    """Return a shortest-path next action from any state, or IGNORE_INDEX at goal."""
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


def scheduled_sampling_probability(args, epoch: int) -> float:
    if args.scheduled_sampling_mode == "off":
        return 0.0
    if args.scheduled_sampling_warmup_epochs <= 0:
        return float(args.scheduled_sampling_end)
    progress = min(epoch / args.scheduled_sampling_warmup_epochs, 1.0)
    return float(
        args.scheduled_sampling_start
        + progress * (args.scheduled_sampling_end - args.scheduled_sampling_start)
    )


def forward_minimal_scheduled(
    model: MinimalMazeActionRNN,
    batch: dict,
    walls: list[set[int]],
    scheduled_sampling_prob: float,
    scheduled_sampling_mode: str,
    is_train: bool,
) -> torch.Tensor:
    """Unroll the GRU with optional state-consistent scheduled sampling."""
    batch_size, max_steps = batch["target"].shape
    device = batch["target"].device
    hidden = None
    logits_by_step = []
    previous_prediction = None
    sampled_next_state = None
    sampled_prev_hit = None
    sampled_prev_reward = None
    sampled_maze_wall = None
    use_state_consistent_sampling = scheduled_sampling_mode == "state_consistent"

    for step_idx in range(max_steps):
        step_batch = {
            key: value[:, step_idx : step_idx + 1]
            for key, value in batch.items()
            if torch.is_tensor(value) and value.ndim >= 2
        }
        prev_action = batch["prev_action"][:, step_idx : step_idx + 1]
        state = batch["state"][:, step_idx : step_idx + 1]
        prev_hit = batch["prev_hit"][:, step_idx : step_idx + 1]
        prev_reward = batch["prev_reward"][:, step_idx : step_idx + 1]
        maze_wall = batch["maze_wall"][:, step_idx : step_idx + 1]
        trial_start = batch["trial_start"][:, step_idx : step_idx + 1]

        if (
            is_train
            and scheduled_sampling_prob > 0.0
            and previous_prediction is not None
            and scheduled_sampling_mode != "off"
        ):
            can_sample = batch["mask"][:, step_idx : step_idx + 1].logical_and(
                trial_start.le(0.0)
            )
            sample_mask = torch.rand_like(prev_action.float()).lt(
                scheduled_sampling_prob
            )
            use_model_prev = can_sample.logical_and(sample_mask)
            prev_action = torch.where(use_model_prev, previous_prediction, prev_action)
            if use_state_consistent_sampling and sampled_next_state is not None:
                state = torch.where(use_model_prev, sampled_next_state, state)
                prev_hit = torch.where(use_model_prev, sampled_prev_hit, prev_hit)
                prev_reward = torch.where(
                    use_model_prev,
                    sampled_prev_reward,
                    prev_reward,
                )
                maze_wall = torch.where(
                    use_model_prev.unsqueeze(-1),
                    sampled_maze_wall,
                    maze_wall,
                )

        step_batch["prev_action"] = prev_action
        step_batch["state"] = state
        step_batch["prev_hit"] = prev_hit
        step_batch["prev_reward"] = prev_reward
        step_batch["maze_wall"] = maze_wall
        x = model.build_input(step_batch)
        output, hidden = model.rnn(x, hidden)
        step_logits = model.action_head(output)
        logits_by_step.append(step_logits)
        previous_prediction = step_logits.argmax(dim=-1).detach()
        if use_state_consistent_sampling:
            (
                sampled_next_state,
                sampled_prev_hit,
                sampled_prev_reward,
                sampled_maze_wall,
            ) = next_step_features_from_actions(
                state.squeeze(1).detach(),
                batch["goal"][:, step_idx].detach(),
                previous_prediction.squeeze(1),
                walls,
                device,
            )

    return torch.cat(logits_by_step, dim=1)


def next_step_features_from_actions(
    states: torch.Tensor,
    goals: torch.Tensor,
    actions: torch.Tensor,
    walls: list[set[int]],
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_states = []
    prev_hits = []
    prev_rewards = []
    maze_walls = []

    for state_value, goal_value, action_value in zip(
        states.detach().cpu().tolist(),
        goals.detach().cpu().tolist(),
        actions.detach().cpu().tolist(),
    ):
        next_state, hit = apply_action(int(state_value), int(action_value), walls)
        prev_hit = 1.0 if hit else 0.0
        if hit:
            prev_reward = -1.0
        elif int(next_state) == int(goal_value):
            prev_reward = 1.0
        else:
            prev_reward = 0.0
        next_states.append([int(next_state)])
        prev_hits.append([prev_hit])
        prev_rewards.append([prev_reward])
        maze_walls.append([_wall_features(walls, int(next_state))])

    return (
        torch.tensor(next_states, dtype=torch.long, device=device),
        torch.tensor(prev_hits, dtype=torch.float32, device=device),
        torch.tensor(prev_rewards, dtype=torch.float32, device=device),
        torch.tensor(maze_walls, dtype=torch.float32, device=device),
    )


def make_recovery_targets(
    batch: dict,
    walls: list[set[int]],
    cache: dict[tuple[int, int], int],
) -> torch.Tensor:
    targets = torch.full_like(batch["target"], IGNORE_INDEX)
    states = batch["state"].detach().cpu()
    goals = batch["goal"].detach().cpu()
    mask = batch["mask"].detach().cpu()

    for batch_idx in range(states.shape[0]):
        for step_idx in range(states.shape[1]):
            if not bool(mask[batch_idx, step_idx]):
                continue
            targets[batch_idx, step_idx] = shortest_recovery_action(
                walls,
                int(states[batch_idx, step_idx]),
                int(goals[batch_idx, step_idx]),
                cache,
            )
    return targets


def make_corrupted_recovery_batch(
    batch: dict,
    walls: list[set[int]],
    cache: dict[tuple[int, int], int],
    corruption_prob: float,
    device,
) -> tuple[dict | None, torch.Tensor | None]:
    if corruption_prob <= 0.0:
        return None, None

    states = batch["state"].detach().cpu()
    goals = batch["goal"].detach().cpu()
    true_actions = batch["target"].detach().cpu()
    mask = batch["mask"].detach().cpu()

    rows = []
    targets = []
    for batch_idx in range(states.shape[0]):
        for step_idx in range(states.shape[1]):
            if not bool(mask[batch_idx, step_idx]):
                continue
            if torch.rand((), device=device).item() >= corruption_prob:
                continue

            state = int(states[batch_idx, step_idx])
            goal = int(goals[batch_idx, step_idx])
            legal = legal_actions(walls, state)
            wrong_actions = [
                action for action in legal if action != int(true_actions[batch_idx, step_idx])
            ]
            if not wrong_actions:
                continue
            corrupt_action = wrong_actions[
                torch.randint(len(wrong_actions), (), device=device).item()
            ]
            corrupt_state, hit = apply_action(state, corrupt_action, walls)
            recovery_target = shortest_recovery_action(
                walls,
                corrupt_state,
                goal,
                cache,
            )
            if recovery_target == IGNORE_INDEX:
                continue

            rows.append(
                {
                    "state": corrupt_state,
                    "goal": goal,
                    "prev_action": corrupt_action,
                    "prev_reward": -1.0 if hit else (1.0 if corrupt_state == goal else 0.0),
                    "maze_wall": _wall_features(walls, corrupt_state),
                    "trial_start": 0.0,
                    "prev_hit": 1.0 if hit else 0.0,
                }
            )
            targets.append(recovery_target)

    if not rows:
        return None, None

    recovery_batch = {
        "state": torch.tensor([[row["state"]] for row in rows], dtype=torch.long, device=device),
        "goal": torch.tensor([[row["goal"]] for row in rows], dtype=torch.long, device=device),
        "prev_action": torch.tensor(
            [[row["prev_action"]] for row in rows],
            dtype=torch.long,
            device=device,
        ),
        "prev_reward": torch.tensor(
            [[row["prev_reward"]] for row in rows],
            dtype=torch.float32,
            device=device,
        ),
        "maze_wall": torch.tensor(
            [[row["maze_wall"]] for row in rows],
            dtype=torch.float32,
            device=device,
        ),
        "trial_start": torch.tensor(
            [[row["trial_start"]] for row in rows],
            dtype=torch.float32,
            device=device,
        ),
        "prev_hit": torch.tensor(
            [[row["prev_hit"]] for row in rows],
            dtype=torch.float32,
            device=device,
        ),
    }
    recovery_targets = torch.tensor(targets, dtype=torch.long, device=device)
    return recovery_batch, recovery_targets


def compute_loss_and_accuracy(
    model: MinimalMazeActionRNN,
    batch: dict,
    walls: list[set[int]],
    recovery_cache: dict[tuple[int, int], int],
    scheduled_sampling_prob: float,
    scheduled_sampling_mode: str,
    recovery_weight: float,
    recovery_corruption_prob: float,
    is_train: bool,
    device,
) -> tuple[torch.Tensor, float, int, dict]:
    logits = forward_minimal_scheduled(
        model,
        batch,
        walls=walls,
        scheduled_sampling_prob=scheduled_sampling_prob if is_train else 0.0,
        scheduled_sampling_mode=scheduled_sampling_mode,
        is_train=is_train,
    )
    targets = batch["target"]
    behavior_loss = F.cross_entropy(
        logits.reshape(-1, 4),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )

    loss = behavior_loss
    recovery_loss_value = 0.0
    corrupt_loss_value = 0.0
    n_corrupt = 0

    if recovery_weight > 0.0:
        recovery_targets = make_recovery_targets(batch, walls, recovery_cache).to(device)
        recovery_loss = F.cross_entropy(
            logits.reshape(-1, 4),
            recovery_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        if torch.isfinite(recovery_loss):
            loss = loss + recovery_weight * recovery_loss
            recovery_loss_value = float(recovery_loss.detach().item())

    if is_train and recovery_weight > 0.0 and recovery_corruption_prob > 0.0:
        recovery_batch, recovery_targets = make_corrupted_recovery_batch(
            batch,
            walls,
            recovery_cache,
            corruption_prob=recovery_corruption_prob,
            device=device,
        )
        if recovery_batch is not None and recovery_targets is not None:
            corrupt_logits = model(recovery_batch).squeeze(1)
            corrupt_loss = F.cross_entropy(corrupt_logits, recovery_targets)
            loss = loss + recovery_weight * corrupt_loss
            corrupt_loss_value = float(corrupt_loss.detach().item())
            n_corrupt = int(recovery_targets.numel())

    mask = batch["mask"]
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets).logical_and(mask).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total else 0.0
    stats = {
        "behavior_loss": float(behavior_loss.detach().item()),
        "recovery_loss": recovery_loss_value,
        "corrupt_recovery_loss": corrupt_loss_value,
        "n_corrupt_recovery": n_corrupt,
    }
    return loss, accuracy, total, stats


def run_epoch(
    model,
    dataloader,
    walls: list[set[int]],
    recovery_cache: dict[tuple[int, int], int],
    device,
    scheduled_sampling_prob: float = 0.0,
    scheduled_sampling_mode: str = "state_consistent",
    recovery_weight: float = 0.0,
    recovery_corruption_prob: float = 0.0,
    optimizer=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_behavior_loss = 0.0
    total_recovery_loss = 0.0
    total_corrupt_loss = 0.0
    total_correct_weighted = 0.0
    total_steps = 0
    total_corrupt = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            loss, accuracy, n_steps, stats = compute_loss_and_accuracy(
                model,
                batch,
                walls=walls,
                recovery_cache=recovery_cache,
                scheduled_sampling_prob=scheduled_sampling_prob,
                scheduled_sampling_mode=scheduled_sampling_mode,
                recovery_weight=recovery_weight,
                recovery_corruption_prob=recovery_corruption_prob,
                is_train=is_train,
                device=device,
            )
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * n_steps
        total_behavior_loss += stats["behavior_loss"] * n_steps
        total_recovery_loss += stats["recovery_loss"] * n_steps
        total_corrupt_loss += stats["corrupt_recovery_loss"] * n_steps
        total_correct_weighted += accuracy * n_steps
        total_steps += n_steps
        total_corrupt += stats["n_corrupt_recovery"]

    mean = lambda value: value / total_steps if total_steps else 0.0
    return {
        "loss": mean(total_loss),
        "behavior_loss": mean(total_behavior_loss),
        "recovery_loss": mean(total_recovery_loss),
        "corrupt_recovery_loss": mean(total_corrupt_loss),
        "accuracy": total_correct_weighted / total_steps if total_steps else 0.0,
        "steps": total_steps,
        "corrupt_recovery_steps": total_corrupt,
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
    first_trial = min(sample["first_trial_index"] for sample in dataset.samples)
    last_trial = max(sample["last_trial_index"] for sample in dataset.samples)
    return {
        "split": split,
        "subject_id": dataset.subject_id,
        "n_sequences": len(dataset),
        "n_steps": n_steps,
        "first_trial_index": first_trial,
        "last_trial_index": last_trial,
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
            "Train the strict minimal subject GRU with scheduled sampling and "
            "shortest-path recovery regularization."
        )
    )
    parser.add_argument("--input", default="maze_healthy_batch123/trial_level.csv")
    parser.add_argument("--maze-wall", default="maze_healthy_batch123/maze_wall.csv")
    parser.add_argument(
        "--diagnostics",
        default="maze_healthy_batch123/trial_level_decode_diagnostics.csv",
    )
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/rnn_subject_minimal_scheduled_recovery_h512",
    )
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=4)
    parser.add_argument("--val-offset", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--recovery-maze", type=int, default=1)
    parser.add_argument(
        "--scheduled-sampling-mode",
        default="state_consistent",
        choices=["state_consistent", "prev_action_only", "off"],
        help=(
            "state_consistent updates state/reward/wall features after sampled "
            "model actions; prev_action_only keeps the original ablation behavior."
        ),
    )
    parser.add_argument("--scheduled-sampling-start", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-end", type=float, default=0.15)
    parser.add_argument("--scheduled-sampling-warmup-epochs", type=int, default=10)
    parser.add_argument("--recovery-weight", type=float, default=0.02)
    parser.add_argument("--recovery-corruption-prob", type=float, default=0.05)
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

    base_walls = _load_base_walls(args.maze_wall)
    if args.recovery_maze not in base_walls:
        raise ValueError(f"recovery_maze={args.recovery_maze} not found in maze wall file")
    recovery_walls = base_walls[args.recovery_maze]
    recovery_cache: dict[tuple[int, int], int] = {}

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

    best_val_accuracy = -1.0
    best_path = output_dir / "best_model.pt"

    print(f"Device: {device}")
    for row in grouping_rows:
        print(
            f"{row['split']}: {row['n_sequences']} sequences, "
            f"{row['n_steps']} steps, trial index "
            f"{row['first_trial_index']}..{row['last_trial_index']}"
        )
    print(
        "Scheduled sampling: "
        f"mode={args.scheduled_sampling_mode}, "
        f"{args.scheduled_sampling_start:.2f} -> {args.scheduled_sampling_end:.2f} "
        f"over {args.scheduled_sampling_warmup_epochs} epochs"
    )
    print(
        "Recovery training: "
        f"weight={args.recovery_weight}, "
        f"corruption_prob={args.recovery_corruption_prob}"
    )

    for epoch in range(1, args.epochs + 1):
        ss_prob = scheduled_sampling_probability(args, epoch)
        train_stats = run_epoch(
            model,
            train_loader,
            walls=recovery_walls,
            recovery_cache=recovery_cache,
            device=device,
            scheduled_sampling_prob=ss_prob,
            scheduled_sampling_mode=args.scheduled_sampling_mode,
            recovery_weight=args.recovery_weight,
            recovery_corruption_prob=args.recovery_corruption_prob,
            optimizer=optimizer,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            walls=recovery_walls,
            recovery_cache=recovery_cache,
            device=device,
            scheduled_sampling_prob=0.0,
            scheduled_sampling_mode=args.scheduled_sampling_mode,
            recovery_weight=args.recovery_weight,
            recovery_corruption_prob=0.0,
            optimizer=None,
        )

        row = {
            "epoch": epoch,
            "scheduled_sampling_mode": args.scheduled_sampling_mode,
            "scheduled_sampling_prob": round(ss_prob, 6),
            "train_loss": round(train_stats["loss"], 6),
            "train_behavior_loss": round(train_stats["behavior_loss"], 6),
            "train_recovery_loss": round(train_stats["recovery_loss"], 6),
            "train_corrupt_recovery_loss": round(
                train_stats["corrupt_recovery_loss"],
                6,
            ),
            "train_accuracy": round(train_stats["accuracy"], 6),
            "train_steps": train_stats["steps"],
            "train_corrupt_recovery_steps": train_stats["corrupt_recovery_steps"],
            "val_loss": round(val_stats["loss"], 6),
            "val_behavior_loss": round(val_stats["behavior_loss"], 6),
            "val_recovery_loss": round(val_stats["recovery_loss"], 6),
            "val_accuracy": round(val_stats["accuracy"], 6),
            "val_steps": val_stats["steps"],
        }
        save_log_row(log_path, row)

        if val_stats["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_stats["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args) | {"model_kind": "minimal"},
                    "model_kind": "minimal",
                    "training_recipe": "minimal_scheduled_sampling_recovery",
                    "input_features": [
                        "current_state",
                        "prev_action",
                        "goal",
                        "prev_reward",
                        "maze_wall",
                        "trial_start_flag",
                        "wall_hit",
                    ],
                    "best_val_accuracy": best_val_accuracy,
                    "grouping": grouping_rows,
                },
                best_path,
            )

        print(
            f"Epoch {epoch:03d} | ss {ss_prob:.3f} | "
            f"train loss {train_stats['loss']:.4f}, acc {train_stats['accuracy']:.4f} | "
            f"val loss {val_stats['loss']:.4f}, acc {val_stats['accuracy']:.4f}"
        )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved best model to {best_path}")
    print(f"Saved training log to {log_path}")
    print(f"Saved grouping summary to {output_dir / 'grouping_summary.csv'}")


if __name__ == "__main__":
    main()
