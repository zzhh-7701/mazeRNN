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

    def forward(self, batch):
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
        return logits
