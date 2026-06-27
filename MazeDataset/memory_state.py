from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


ACTION_DELTAS = {0: -7, 1: -1, 2: 7, 3: 1}
ACTION_NAMES = ("left", "up", "right", "down")
ACTION_ENCODING = {idx: name for idx, name in enumerate(ACTION_NAMES)}
WALL_COLUMN_ORDER = ("walls_l", "walls_u", "walls_r", "walls_d")
RELATIVE_ACTION_NAMES = ("forward", "left", "right", "back")
DEFAULT_HEADING = -1
GRID_SIZE = 7
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4


def _coord(state: int) -> tuple[int, int]:
    state = int(state)
    return state // GRID_SIZE, state % GRID_SIZE


def _manhattan(a: int, b: int) -> int:
    ac, ar = _coord(a)
    bc, br = _coord(b)
    return abs(ac - bc) + abs(ar - br)


def is_blocked(walls: list[set[int]], state: int, action: int) -> bool:
    next_state = int(state) + ACTION_DELTAS[int(action)]
    return int(state) in walls[int(action)] or not 0 <= next_state < N_STATES


def apply_action_to_state(
    state: int,
    action: int,
    walls: list[set[int]],
) -> tuple[int, bool]:
    if is_blocked(walls, state, action):
        return int(state), True
    return int(state) + ACTION_DELTAS[int(action)], False


def opposite_action(action: int) -> int:
    return (int(action) + 2) % N_ACTIONS


def compute_relative_action(heading: int, action: int) -> int:
    """Return 0/1/2/3 for forward/left/right/back."""
    heading = int(heading)
    action = int(action)
    if heading < 0:
        raise ValueError("Relative action is undefined when heading is unknown")
    if action == heading:
        return 0
    if action == (heading - 1) % N_ACTIONS:
        return 1
    if action == (heading + 1) % N_ACTIONS:
        return 2
    return 3


def update_heading(old_heading: int, action: int, collision: bool) -> int:
    if collision:
        return int(old_heading)
    return int(action)


def local_wall_flags(walls: list[set[int]], state: int) -> np.ndarray:
    return np.array(
        [float(is_blocked(walls, state, action)) for action in range(N_ACTIONS)],
        dtype=np.float32,
    )


def local_topology_from_walls(wall_flags: np.ndarray) -> dict[str, float]:
    open_actions = [action for action, blocked in enumerate(wall_flags) if not blocked]
    degree = len(open_actions)
    is_corridor = (
        degree == 2
        and opposite_action(open_actions[0]) == open_actions[1]
    )
    is_corner = degree == 2 and not is_corridor
    return {
        "degree": degree / float(N_ACTIONS),
        "is_dead_end": float(degree == 1),
        "is_corridor": float(is_corridor),
        "is_corner": float(is_corner),
        "is_junction": float(degree >= 3),
    }


@dataclass
class MemoryFeatureSpec:
    variant: str = "A"

    @property
    def names(self) -> list[str]:
        variant = self.variant.upper()
        names: list[str] = []
        if variant in {"B", "C", "D", "E"}:
            names.extend(f"local_wall_{name}" for name in ACTION_NAMES)
            names.extend(
                [
                    "local_degree",
                    "local_is_dead_end",
                    "local_is_corridor",
                    "local_is_corner",
                    "local_is_junction",
                ]
            )
        if variant in {"C", "D", "E"}:
            names.extend(f"known_wall_{name}" for name in ACTION_NAMES)
            names.extend(f"known_edge_{name}" for name in ACTION_NAMES)
            names.extend(f"unknown_edge_{name}" for name in ACTION_NAMES)
            names.extend(f"visited_edge_{name}" for name in ACTION_NAMES)
            names.extend(f"attempted_edge_{name}" for name in ACTION_NAMES)
            names.extend(f"collision_memory_{name}" for name in ACTION_NAMES)
            names.extend(
                [
                    "visited_node_count",
                    "last_visit_recency",
                    "recent_loop_flag",
                    "backtrack_flag",
                    "conflict_signal",
                    "replan_trigger",
                ]
            )
        if variant in {"D", "E"}:
            names.extend(f"heading_{name}" for name in ACTION_NAMES)
            names.append("heading_unknown")
            names.extend(
                [
                    "wall_in_front",
                    "wall_on_left",
                    "wall_on_right",
                    "wall_behind",
                ]
            )
        if variant in {"B", "C", "D", "E"}:
            names.extend(
                [
                    "goal_delta_col",
                    "goal_delta_row",
                    "goal_manhattan_distance",
                ]
            )
        return names

    @property
    def dim(self) -> int:
        return len(self.names)


class MazeMemoryState:
    """Per-trial subject memory separated from true environment transition."""

    def __init__(self, n_states: int = N_STATES, n_actions: int = N_ACTIONS) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.reset()

    def reset(self, task_id: int | None = None, trial_id: int | None = None) -> None:
        self.task_id = task_id
        self.trial_id = trial_id
        self.known_node_mask = np.zeros(self.n_states, dtype=np.float32)
        self.known_edge_mask = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.known_wall_mask = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.visited_node_count = np.zeros(self.n_states, dtype=np.float32)
        self.visited_edge_count = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.last_visit_time = np.full(self.n_states, -1, dtype=np.int32)
        self.attempted_edge_count = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.collision_memory = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.recent_positions: deque[int] = deque(maxlen=6)
        self.previous_state: int | None = None
        self.recent_loop_flag = 0.0
        self.backtrack_flag = 0.0
        self.conflict_signal = 0.0
        self.replan_trigger = 0.0

    def observe_local_walls(self, state: int, walls: list[set[int]]) -> np.ndarray:
        state = int(state)
        wall_flags = local_wall_flags(walls, state)
        self.known_node_mask[state] = 1.0
        self.known_wall_mask[state] = np.maximum(self.known_wall_mask[state], wall_flags)
        self.known_edge_mask[state] = np.maximum(
            self.known_edge_mask[state],
            1.0 - wall_flags,
        )
        return wall_flags

    def get_features(
        self,
        current_pos: int,
        goal_pos: int,
        heading: int,
        walls: list[set[int]] | None = None,
        local_visual: np.ndarray | None = None,
        variant: str = "A",
        t: int | None = None,
    ) -> np.ndarray:
        variant = variant.upper()
        spec = MemoryFeatureSpec(variant)
        if spec.dim == 0:
            return np.empty((0,), dtype=np.float32)

        current_pos = int(current_pos)
        goal_pos = int(goal_pos)
        heading = int(heading)
        if local_visual is not None:
            wall_flags = np.asarray(local_visual, dtype=np.float32)
        elif walls is not None:
            wall_flags = local_wall_flags(walls, current_pos)
        else:
            wall_flags = self.known_wall_mask[current_pos].copy()

        features: list[float] = []
        if variant in {"B", "C", "D", "E"}:
            topology = local_topology_from_walls(wall_flags)
            features.extend(wall_flags.tolist())
            features.extend(
                [
                    topology["degree"],
                    topology["is_dead_end"],
                    topology["is_corridor"],
                    topology["is_corner"],
                    topology["is_junction"],
                ]
            )
        if variant in {"C", "D", "E"}:
            known_wall = self.known_wall_mask[current_pos]
            known_edge = self.known_edge_mask[current_pos]
            unknown_edge = 1.0 - np.clip(known_wall + known_edge, 0.0, 1.0)
            recency = 0.0
            if t is not None and self.last_visit_time[current_pos] >= 0:
                recency = 1.0 / (1.0 + max(0, int(t) - int(self.last_visit_time[current_pos])))
            features.extend(known_wall.tolist())
            features.extend(known_edge.tolist())
            features.extend(unknown_edge.tolist())
            features.extend(np.log1p(self.visited_edge_count[current_pos]).tolist())
            features.extend(np.log1p(self.attempted_edge_count[current_pos]).tolist())
            features.extend(self.collision_memory[current_pos].tolist())
            features.extend(
                [
                    float(np.log1p(self.visited_node_count[current_pos])),
                    float(recency),
                    float(self.recent_loop_flag),
                    float(self.backtrack_flag),
                    float(self.conflict_signal),
                    float(self.replan_trigger),
                ]
            )
        if variant in {"D", "E"}:
            heading_one_hot = [0.0] * self.n_actions
            heading_known = 0 <= heading < self.n_actions
            if heading_known:
                heading_one_hot[heading] = 1.0
            features.extend(heading_one_hot)
            features.append(float(not heading_known))
            if not heading_known:
                features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend(
                    [
                        float(self.known_wall_mask[current_pos, heading]),
                        float(self.known_wall_mask[current_pos, (heading - 1) % self.n_actions]),
                        float(self.known_wall_mask[current_pos, (heading + 1) % self.n_actions]),
                        float(self.known_wall_mask[current_pos, opposite_action(heading)]),
                    ]
                )
        if variant in {"B", "C", "D", "E"}:
            cc, cr = _coord(current_pos)
            gc, gr = _coord(goal_pos)
            features.extend(
                [
                    (gc - cc) / float(GRID_SIZE - 1),
                    (gr - cr) / float(GRID_SIZE - 1),
                    _manhattan(current_pos, goal_pos) / float((GRID_SIZE - 1) * 2),
                ]
            )

        return np.asarray(features, dtype=np.float32)

    def update(
        self,
        current_pos: int,
        action: int,
        next_pos: int,
        collision: bool,
        t: int | None = None,
    ) -> None:
        current_pos = int(current_pos)
        action = int(action)
        next_pos = int(next_pos)
        step = int(t) if t is not None else 0

        self.known_node_mask[current_pos] = 1.0
        self.visited_node_count[current_pos] += 1.0
        self.last_visit_time[current_pos] = step
        self.visited_edge_count[current_pos, action] += 1.0
        self.attempted_edge_count[current_pos, action] += 1.0

        self.conflict_signal = 0.0
        self.replan_trigger = 0.0
        if collision:
            if self.known_edge_mask[current_pos, action] > 0:
                self.conflict_signal = 1.0
                self.replan_trigger = 1.0
            self.known_wall_mask[current_pos, action] = 1.0
            self.collision_memory[current_pos, action] = 1.0
        else:
            self.known_edge_mask[current_pos, action] = 1.0
            self.known_node_mask[next_pos] = 1.0

        self.backtrack_flag = float(
            self.previous_state is not None and next_pos == self.previous_state
        )
        self.recent_loop_flag = float(next_pos in self.recent_positions)
        self.previous_state = current_pos
        self.recent_positions.append(next_pos)

    def merge_from(self, other: "MazeMemoryState") -> None:
        self.known_node_mask = np.maximum(self.known_node_mask, other.known_node_mask)
        self.known_edge_mask = np.maximum(self.known_edge_mask, other.known_edge_mask)
        self.known_wall_mask = np.maximum(self.known_wall_mask, other.known_wall_mask)
        self.collision_memory = np.maximum(self.collision_memory, other.collision_memory)


class SubjectMazeMemoryStore:

    """Long-term subject memory keyed by maze id, carried across trials."""

    def __init__(self) -> None:
        self._by_maze: dict[int, MazeMemoryState] = {}

    def start_trial(
        self,
        maze_id: int,
        task_id: int | None = None,
        trial_id: int | None = None,
    ) -> MazeMemoryState:
        memory = MazeMemoryState()
        memory.reset(task_id=task_id, trial_id=trial_id)
        long_term = self._by_maze.get(int(maze_id))
        if long_term is not None:
            memory.merge_from(long_term)
        return memory

    def finish_trial(self, maze_id: int, memory: MazeMemoryState) -> None:
        maze_id = int(maze_id)
        if maze_id not in self._by_maze:
            self._by_maze[maze_id] = MazeMemoryState()
        self._by_maze[maze_id].merge_from(memory)

    def copy(self) -> "SubjectMazeMemoryStore":
        """Deep-copy only the long-term fields needed across trials."""
        new_store = SubjectMazeMemoryStore()
        for maze_id, mem in self._by_maze.items():
            new_mem = MazeMemoryState()
            new_mem.known_node_mask = mem.known_node_mask.copy()
            new_mem.known_edge_mask = mem.known_edge_mask.copy()
            new_mem.known_wall_mask = mem.known_wall_mask.copy()
            new_mem.collision_memory = mem.collision_memory.copy()
            new_store._by_maze[maze_id] = new_mem
        return new_store


def feature_names_for_variant(variant: str) -> list[str]:
    return MemoryFeatureSpec(variant).names
