from __future__ import annotations

import torch
from torch import nn


class MazeActionRNN(nn.Module):
    """Predict the participant's next action at each step of a maze trial."""

    def __init__(
        self,
        n_states: int = 49,
        n_actions_with_special: int = 6,
        n_tasks: int = 6,
        n_replan: int = 2,
        state_dim: int = 16,
        action_dim: int = 8,
        task_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_embedding = nn.Embedding(n_states, state_dim)
        self.goal_embedding = nn.Embedding(n_states, state_dim)
        self.prev_action_embedding = nn.Embedding(n_actions_with_special, action_dim)
        self.task_embedding = nn.Embedding(n_tasks, task_dim)
        self.replan_embedding = nn.Embedding(n_replan, task_dim)

        input_dim = state_dim * 2 + action_dim + task_dim * 2
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.action_head = nn.Linear(hidden_dim, 4)

    def forward(self, batch, return_hidden: bool = False):
        x = torch.cat(
            [
                self.state_embedding(batch["state"]),
                self.goal_embedding(batch["goal"]),
                self.prev_action_embedding(batch["prev_action"]),
                self.task_embedding(batch["task"]),
                self.replan_embedding(batch["replan"]),
            ],
            dim=-1,
        )
        output, _ = self.rnn(x)
        logits = self.action_head(output)
        if return_hidden:
            return {
                "logits": logits,
                "hidden": output,
            }
        return logits


class MinimalMazeActionRNN(nn.Module):
    """Predict actions from the strict minimal within-subject feature set."""

    def __init__(
        self,
        n_states: int = 49,
        n_actions_with_special: int = 6,
        state_dim: int = 16,
        action_dim: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        cognitive_feature_dim: int = 0,
        include_maze_wall: bool = True,
        use_relative_action_head: bool = False,
        use_auxiliary_heads: bool = False,
    ) -> None:
        super().__init__()
        self.cognitive_feature_dim = int(cognitive_feature_dim)
        self.include_maze_wall = bool(include_maze_wall)
        self.use_relative_action_head = bool(use_relative_action_head)
        self.use_auxiliary_heads = bool(use_auxiliary_heads)
        self.state_embedding = nn.Embedding(n_states, state_dim)
        self.goal_embedding = nn.Embedding(n_states, state_dim)
        self.prev_action_embedding = nn.Embedding(n_actions_with_special, action_dim)

        continuous_dim = 3 + (4 if self.include_maze_wall else 0)
        input_dim = state_dim * 2 + action_dim + continuous_dim + self.cognitive_feature_dim
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.action_head = nn.Linear(hidden_dim, 4)
        self.relative_action_head = (
            nn.Linear(hidden_dim, 4) if self.use_relative_action_head else None
        )
        self.next_position_head = nn.Linear(hidden_dim, n_states) if self.use_auxiliary_heads else None
        self.collision_head = nn.Linear(hidden_dim, 1) if self.use_auxiliary_heads else None
        self.distance_delta_head = nn.Linear(hidden_dim, 1) if self.use_auxiliary_heads else None
        self.loop_head = nn.Linear(hidden_dim, 1) if self.use_auxiliary_heads else None
        self.newly_seen_wall_head = nn.Linear(hidden_dim, 1) if self.use_auxiliary_heads else None

    def build_input(self, batch):
        parts = [
            self.state_embedding(batch["state"]),
            self.goal_embedding(batch["goal"]),
            self.prev_action_embedding(batch["prev_action"]),
            batch["prev_reward"].unsqueeze(-1).float(),
        ]
        if self.include_maze_wall:
            parts.append(batch["maze_wall"].float())
        parts.extend(
            [
                batch.get(
                    "cognitive_features",
                    torch.empty(
                        (*batch["state"].shape, 0),
                        dtype=torch.float32,
                        device=batch["state"].device,
                    ),
                ).float(),
                batch["trial_start"].unsqueeze(-1).float(),
                batch["prev_hit"].unsqueeze(-1).float(),
            ]
        )
        return torch.cat(parts, dim=-1)

    def forward(self, batch, return_hidden: bool = False):
        x = self.build_input(batch)
        output, _ = self.rnn(x)
        logits = self.action_head(output)
        if return_hidden:
            result = {
                "logits": logits,
                "hidden": output,
            }
            if self.relative_action_head is not None:
                result["relative_logits"] = self.relative_action_head(output)
            return result
        return logits

    def forward_with_aux(self, batch):
        x = self.build_input(batch)
        output, _ = self.rnn(x)
        result = {
            "logits": self.action_head(output),
            "hidden": output,
        }
        if self.relative_action_head is not None:
            result["relative_logits"] = self.relative_action_head(output)
        if self.use_auxiliary_heads:
            result["next_position_logits"] = self.next_position_head(output)
            result["collision_logits"] = self.collision_head(output).squeeze(-1)
            result["distance_delta"] = self.distance_delta_head(output).squeeze(-1)
            result["loop_logits"] = self.loop_head(output).squeeze(-1)
            result["newly_seen_wall_logits"] = self.newly_seen_wall_head(output).squeeze(-1)
        return result
