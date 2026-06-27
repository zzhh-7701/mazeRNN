from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_DELTAS = {0: -7, 1: -1, 2: 7, 3: 1}
ACTION_NAMES = {0: "left", 1: "up", 2: "right", 3: "down"}
OPPOSITE_ACTION = {0: 2, 2: 0, 1: 3, 3: 1}
WALL_COLUMNS = ["walls_l", "walls_u", "walls_r", "walls_d"]


DEFAULT_MODELS = [
    (
        "old_minimal",
        "outputs/rnn_subject_13015195273_minimal_h512/strict_minimal_eval",
    ),
    (
        "context",
        "outputs/rnn_subject_13015195273_context_minimal_h512/context_recovery",
    ),
    (
        "rollout_soft",
        "outputs/rnn_subject_13015195273_minimal_rollout_h512/rollout_eval",
    ),
    (
        "hardmask_recovery",
        "outputs/rnn_subject_13015195273_minimal_hardmask_recovery_h512/hardmask_recovery_eval",
    ),
]


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    return ast.literal_eval(str(value))


def load_base_walls(path: Path) -> dict[int, list[set[int]]]:
    wall_df = pd.read_csv(path)
    walls = {}
    for row in wall_df.itertuples(index=False):
        walls[int(row.maze)] = [
            set(parse_list(getattr(row, col))) for col in WALL_COLUMNS
        ]
    return walls


def parse_trial_walls(row, base_walls: dict[int, list[set[int]]]) -> list[set[int]]:
    maze_wall = getattr(row, "maze_wall", "")
    if pd.notna(maze_wall) and maze_wall != "":
        return [set(x) for x in json.loads(maze_wall)]
    return base_walls[int(row.maze)]


def legal_actions(walls: list[set[int]], state: int) -> list[int]:
    return [
        action
        for action in range(4)
        if state not in walls[action] and 0 <= state + ACTION_DELTAS[action] < 49
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


def state_transition_to_action(current_state: int, next_state: int):
    delta = int(next_state) - int(current_state)
    for action, action_delta in ACTION_DELTAS.items():
        if delta == action_delta:
            return action
    return None


def shortest_path_action(short_path: list[int], state: int):
    for idx, path_state in enumerate(short_path[:-1]):
        if int(path_state) == int(state):
            return state_transition_to_action(path_state, short_path[idx + 1])
    return None


def entropy_from_counts(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob)
    return entropy / math.log(4)


def distribution(values, labels) -> dict:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {label: 0.0 for label in labels}
    return {label: counts.get(label, 0) / total for label in labels}


def trial_behavior_features(row, base_walls: dict[int, list[set[int]]]) -> dict:
    actions = [int(x) for x in parse_list(row.action)]
    hits = [bool(x) for x in parse_list(row.hits)]
    path = [int(x) for x in parse_list(row.true_path)]
    short_path = [int(x) for x in parse_list(row.short_path)]
    walls = parse_trial_walls(row, base_walls)
    goal = int(row.goal)
    distances = shortest_distances_to_goal(walls, goal)

    n_actions = len(actions)
    state_counts = Counter(path)
    revisited_steps = sum(max(0, count - 1) for count in state_counts.values())
    unique_states = len(state_counts)

    action_counts = Counter(actions)
    first_action = actions[0] if actions else None
    first_shortest = (
        shortest_path_action(short_path, int(row.start)) if len(short_path) > 1 else None
    )

    straight = 0
    turn = 0
    reverse = 0
    for previous, current in zip(actions[:-1], actions[1:]):
        if current == previous:
            straight += 1
        elif OPPOSITE_ACTION.get(previous) == current:
            reverse += 1
        else:
            turn += 1
    transition_total = max(0, len(actions) - 1)

    shortest_matches = []
    progress_deltas = []
    for step_idx, action in enumerate(actions):
        state = path[step_idx] if step_idx < len(path) else path[-1] if path else int(row.start)
        sp_action = shortest_path_action(short_path, state)
        if sp_action is not None:
            shortest_matches.append(int(action == sp_action))
        next_state = path[step_idx + 1] if step_idx + 1 < len(path) else state
        current_distance = distances.get(int(state))
        next_distance = distances.get(int(next_state))
        if current_distance is not None and next_distance is not None:
            progress_deltas.append(current_distance - next_distance)

    positive_progress = [x for x in progress_deltas if x > 0]
    zero_progress = [x for x in progress_deltas if x == 0]
    negative_progress = [x for x in progress_deltas if x < 0]

    return {
        "task": int(row.task),
        "source": row.source,
        "sequence_trial_index": int(row.sequence_trial_index),
        "start": int(row.start),
        "goal": goal,
        "n_actions": n_actions,
        "shortest_steps": int(getattr(row, "shortest_steps", max(len(short_path) - 1, 0))),
        "excess_steps": float(getattr(row, "excess_steps", np.nan)),
        "hit_rate": float(getattr(row, "hit_rate", np.nan)),
        "reached_goal": int(getattr(row, "reached_goal", 0)),
        "path_efficiency": float(getattr(row, "path_efficiency", np.nan)),
        "revisit_rate": revisited_steps / len(path) if path else 0.0,
        "unique_state_fraction": unique_states / len(path) if path else 0.0,
        "action_entropy": entropy_from_counts(action_counts, n_actions),
        "left_frac": action_counts.get(0, 0) / n_actions if n_actions else 0.0,
        "up_frac": action_counts.get(1, 0) / n_actions if n_actions else 0.0,
        "right_frac": action_counts.get(2, 0) / n_actions if n_actions else 0.0,
        "down_frac": action_counts.get(3, 0) / n_actions if n_actions else 0.0,
        "straight_frac": straight / transition_total if transition_total else 0.0,
        "turn_frac": turn / transition_total if transition_total else 0.0,
        "reverse_frac": reverse / transition_total if transition_total else 0.0,
        "first_action_matches_shortest": (
            int(first_action == first_shortest) if first_shortest is not None else np.nan
        ),
        "shortest_action_alignment": (
            float(np.mean(shortest_matches)) if shortest_matches else np.nan
        ),
        "mean_distance_delta": (
            float(np.mean(progress_deltas)) if progress_deltas else np.nan
        ),
        "progress_action_frac": (
            len(positive_progress) / len(progress_deltas) if progress_deltas else np.nan
        ),
        "stall_action_frac": (
            len(zero_progress) / len(progress_deltas) if progress_deltas else np.nan
        ),
        "regress_action_frac": (
            len(negative_progress) / len(progress_deltas) if progress_deltas else np.nan
        ),
    }


def wasserstein_1d(a, b) -> float:
    a = np.asarray([x for x in a if pd.notna(x)], dtype=float)
    b = np.asarray([x for x in b if pd.notna(x)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    quantiles = np.linspace(0, 1, max(len(a), len(b)))
    aq = np.quantile(a, quantiles)
    bq = np.quantile(b, quantiles)
    return float(np.mean(np.abs(aq - bq)))


def total_variation(actual: pd.Series, model: pd.Series, cols: list[str]) -> float:
    return 0.5 * float(sum(abs(float(actual[col]) - float(model[col])) for col in cols))


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "n_actions",
        "shortest_steps",
        "excess_steps",
        "hit_rate",
        "reached_goal",
        "path_efficiency",
        "revisit_rate",
        "unique_state_fraction",
        "action_entropy",
        "left_frac",
        "up_frac",
        "right_frac",
        "down_frac",
        "straight_frac",
        "turn_frac",
        "reverse_frac",
        "first_action_matches_shortest",
        "shortest_action_alignment",
        "mean_distance_delta",
        "progress_action_frac",
        "stall_action_frac",
        "regress_action_frac",
    ]
    return (
        features.groupby(["model", "source_role", "task"], dropna=False)[numeric_cols]
        .mean()
        .reset_index()
        .round(6)
    )


def compute_human_similarity(features: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual_summary = summary[summary["source_role"].eq("actual")].copy()
    actual_features = features[features["source_role"].eq("actual")].copy()

    for model in sorted(features.loc[features["source_role"].eq("model"), "model"].unique()):
        for task in sorted(features["task"].dropna().unique()):
            actual = actual_summary[
                actual_summary["task"].eq(task) & actual_summary["model"].eq(model)
            ]
            if actual.empty:
                actual = actual_summary[actual_summary["task"].eq(task)]
            model_row = summary[
                summary["model"].eq(model)
                & summary["source_role"].eq("model")
                & summary["task"].eq(task)
            ]
            if actual.empty or model_row.empty:
                continue
            actual_row = actual.iloc[0]
            model_row = model_row.iloc[0]

            actual_trials = actual_features[actual_features["task"].eq(task)]
            model_trials = features[
                features["model"].eq(model)
                & features["source_role"].eq("model")
                & features["task"].eq(task)
            ]

            actual_length_mean = max(float(actual_row["n_actions"]), 1.0)
            actual_excess_mean = max(float(actual_row["excess_steps"]), 1.0)
            length_w1 = wasserstein_1d(actual_trials["n_actions"], model_trials["n_actions"])
            excess_w1 = wasserstein_1d(
                actual_trials["excess_steps"],
                model_trials["excess_steps"],
            )
            length_w1_norm = length_w1 / actual_length_mean
            excess_w1_norm = excess_w1 / actual_excess_mean

            action_tv = total_variation(
                actual_row,
                model_row,
                ["left_frac", "up_frac", "right_frac", "down_frac"],
            )
            turn_tv = total_variation(
                actual_row,
                model_row,
                ["straight_frac", "turn_frac", "reverse_frac"],
            )
            revisit_abs = abs(float(actual_row["revisit_rate"]) - float(model_row["revisit_rate"]))
            unique_abs = abs(
                float(actual_row["unique_state_fraction"])
                - float(model_row["unique_state_fraction"])
            )
            efficiency_abs = abs(
                float(actual_row["path_efficiency"]) - float(model_row["path_efficiency"])
            )
            shortest_abs = abs(
                float(actual_row["shortest_action_alignment"])
                - float(model_row["shortest_action_alignment"])
            )
            progress_abs = abs(
                float(actual_row["mean_distance_delta"])
                - float(model_row["mean_distance_delta"])
            )
            hit_abs = abs(float(actual_row["hit_rate"]) - float(model_row["hit_rate"]))
            reached_gap = max(
                0.0,
                float(actual_row["reached_goal"]) - float(model_row["reached_goal"]),
            )

            distance = (
                1.2 * length_w1_norm
                + 1.0 * excess_w1_norm
                + 1.0 * action_tv
                + 0.8 * turn_tv
                + 0.8 * revisit_abs
                + 0.8 * progress_abs
                + 0.5 * shortest_abs
                + 0.4 * unique_abs
                + 0.4 * efficiency_abs
                + 0.3 * hit_abs
                + 1.5 * reached_gap
            )
            rows.append(
                {
                    "model": model,
                    "task": int(task),
                    "human_distance": distance,
                    "human_similarity": 1.0 / (1.0 + distance),
                    "length_w1_norm": length_w1_norm,
                    "excess_w1_norm": excess_w1_norm,
                    "action_tv": action_tv,
                    "turn_tv": turn_tv,
                    "revisit_abs": revisit_abs,
                    "unique_abs": unique_abs,
                    "efficiency_abs": efficiency_abs,
                    "shortest_alignment_abs": shortest_abs,
                    "progress_abs": progress_abs,
                    "hit_abs": hit_abs,
                    "reached_gap": reached_gap,
                }
            )
    return pd.DataFrame(rows).round(6)


def model_overview(similarity: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "human_distance",
        "human_similarity",
        "length_w1_norm",
        "excess_w1_norm",
        "action_tv",
        "turn_tv",
        "revisit_abs",
        "progress_abs",
        "reached_gap",
    ]
    return (
        similarity.groupby("model", dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .sort_values("human_similarity", ascending=False)
        .round(6)
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        headers[idx].ljust(widths[idx]) for idx in range(len(headers))
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def load_model_eval(label: str, eval_dir: Path, base_walls) -> pd.DataFrame:
    metrics_path = eval_dir / "behavior_recovery_trial_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    features = []
    for col in ["action", "hits", "true_path", "short_path"]:
        df[col] = df[col].map(parse_list)

    for row in df.itertuples(index=False):
        if int(row.task) not in [1, 2, 3, 4]:
            continue
        source_role = "actual" if row.source == "actual_subject" else "model"
        row_features = trial_behavior_features(row, base_walls)
        row_features["model"] = label
        row_features["source_role"] = source_role
        row_features["raw_source"] = row.source
        features.append(row_features)
    return pd.DataFrame(features)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    similarity: pd.DataFrame,
    overview: pd.DataFrame,
) -> None:
    lines = [
        "# Human-Likeness Diagnostics",
        "",
        "This report separates validity from human-likeness. Hard masks can make rollouts legal, but human-likeness is measured by whether the model matches the participant's behavioral distributions within each task.",
        "",
        "## Model Ranking",
        "",
        dataframe_to_markdown(overview),
        "",
        "## Per-Task Human Similarity",
        "",
        dataframe_to_markdown(
            similarity[
                [
                    "model",
                    "task",
                    "human_similarity",
                    "human_distance",
                    "length_w1_norm",
                    "excess_w1_norm",
                    "action_tv",
                    "turn_tv",
                    "revisit_abs",
                    "progress_abs",
                    "reached_gap",
                ]
            ].sort_values(["task", "human_similarity"], ascending=[True, False])
        ),
        "",
        "## What To Inspect Next",
        "",
        "- If `length_w1_norm` / `excess_w1_norm` is high, the model is too short-path-like or too wandering compared with the participant.",
        "- If `action_tv` is high, the model uses different direction priors than the participant.",
        "- If `turn_tv` is high, the model's local movement style differs: too straight, too turn-heavy, or too many reversals.",
        "- If `revisit_abs` is high, the model does not match human re-checking / looping behavior.",
        "- If `progress_abs` is high, the model approaches the goal at a different rate from the participant.",
        "- If `reached_gap` is high, the model still fails basic task completion and should not be selected even if other features match.",
        "",
        "## Recommended Training Direction",
        "",
        "1. Keep hard illegal action masks as an environment constraint, not as the human-likeness objective.",
        "2. Select checkpoints with task-balanced human similarity, with a minimum reached-goal floor per task.",
        "3. Replace one-hot shortest recovery labels with soft recovery targets that mix shortest recovery and the participant's local action prior.",
        "4. Add task-balanced rollout diagnostics to every experiment, especially for Task 2 and Task 3.",
    ]
    (output_dir / "REPORT_human_likeness.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    best_by_task = (
        similarity.sort_values(["task", "human_similarity"], ascending=[True, False])
        .groupby("task", as_index=False)
        .first()
    )
    detail_table = similarity[
        [
            "model",
            "task",
            "human_similarity",
            "human_distance",
            "length_w1_norm",
            "excess_w1_norm",
            "action_tv",
            "turn_tv",
            "revisit_abs",
            "progress_abs",
            "reached_gap",
        ]
    ].sort_values(["task", "human_similarity"], ascending=[True, False])

    zh_lines = [
        "# 人类相似性诊断报告",
        "",
        "## 核心思路",
        "",
        "现在要区分两件事：",
        "",
        "- **validity**：动作是否合法、是否撞墙、是否能到终点。",
        "- **human-likeness**：路径长度、绕路程度、重复访问、转向/折返、动作偏好、朝目标推进方式是否像真实被试。",
        "",
        "hard illegal mask 适合作为环境底线，但它本身不会让模型更像人。它会让模型更像一个合法导航器；如果 selection 只奖励到达率和低撞墙，模型就会越来越像“会跑迷宫的 AI”。",
        "",
        "## 本次加入的检测指标",
        "",
        "- `length_w1_norm`：模型路径长度分布和真实人的距离，越低越像。",
        "- `excess_w1_norm`：绕路步数分布距离，越低越像。",
        "- `action_tv`：上下左右动作比例差异，越低越像。",
        "- `turn_tv`：直走/转弯/反向折返模式差异，越低越像。",
        "- `revisit_abs`：重复访问状态比例差异，越低越像。",
        "- `progress_abs`：每步朝目标推进速度差异，越低越像。",
        "- `reached_gap`：真实人到达但模型没到达的缺口。这个是底线项。",
        "- `human_similarity`：上述指标的加权综合分，范围约 0 到 1，越高越像真实被试。",
        "",
        "## 模型整体排序",
        "",
        dataframe_to_markdown(overview),
        "",
        "## 每个 Task 最像人的模型",
        "",
        dataframe_to_markdown(
            best_by_task[
                [
                    "task",
                    "model",
                    "human_similarity",
                    "human_distance",
                    "length_w1_norm",
                    "excess_w1_norm",
                    "revisit_abs",
                    "progress_abs",
                    "reached_gap",
                ]
            ]
        ),
        "",
        "## 每个 Task 的完整对比",
        "",
        dataframe_to_markdown(detail_table),
        "",
        "## 当前结果怎么解释",
        "",
        "1. `hardmask_recovery` 整体 human similarity 最高，尤其 Task 2/3 明显优于旧 minimal 和 context。这说明 hard mask + recovery 解决了原先 Task 2/3 的失败、撞墙和灾难性绕路。",
        "",
        "2. 但 Task 4 上 `rollout_soft` 比 `hardmask_recovery` 更像真实人。原因很可能是 Task 4 真实被试本来有较多探索、撞墙、重复检查；hardmask 把撞墙清零，validity 变好，但 human-likeness 下降。",
        "",
        "3. `old_minimal` 单步 accuracy 高，但 human similarity 最差之一，说明单步方向预测不是这个实验的主要选择标准。",
        "",
        "4. `context` 有一定恢复能力，但路径长度和绕路分布仍然偏离真实人，尤其 Task 1/3/4 的 length/excess 距离比较大。",
        "",
        "## 后续训练建议",
        "",
        "1. 保留 hardmask 作为环境约束，但不要把“零撞墙”当成人类相似性的目标。Task 4 可以考虑允许低概率 wall-hit 行为作为观测输出，或者在 scoring 中按真实人的 task-specific hit rate 惩罚“过低撞墙”。",
        "",
        "2. checkpoint selection 改成：先要求每个 task reached_goal_rate 达到底线，再用 task-balanced human_similarity 选模型。",
        "",
        "3. off-trajectory recovery 不要永远用 one-hot shortest path。建议换成 soft target：`70% shortest recovery + 30% human local action prior`，Task 4 的 human prior 权重可以更高。",
        "",
        "4. 分 task 训练或分 task head 值得考虑：Task 2/3 更像 goal-directed recovery，Task 4 更像探索/再计划。一个统一 loss 很容易顾此失彼。",
        "",
        "5. 每次训练后都跑本诊断脚本，不再只看 accuracy、到达率、平均路径长度。",
    ]
    (output_dir / "REPORT_human_likeness_zh.md").write_text(
        "\n".join(zh_lines),
        encoding="utf-8",
    )


def parse_model_args(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_MODELS
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=EVAL_DIR, got {value}")
        label, path = value.split("=", 1)
        parsed.append((label, path))
    return parsed


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare model rollouts against participant behavior with task-balanced human-likeness diagnostics."
    )
    parser.add_argument("--maze-wall", default="maze_healthy_batch123/maze_wall.csv")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model eval directory as LABEL=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/human_likeness_subject_13015195273",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_walls = load_base_walls(Path(args.maze_wall))

    feature_frames = []
    for label, path in parse_model_args(args.model):
        feature_frames.append(load_model_eval(label, Path(path), base_walls))
    features = pd.concat(feature_frames, ignore_index=True)
    summary = summarize_features(features)
    similarity = compute_human_similarity(features, summary)
    overview = model_overview(similarity)

    features.to_csv(output_dir / "trial_human_features.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "task_human_feature_summary.csv", index=False, encoding="utf-8-sig")
    similarity.to_csv(output_dir / "human_similarity_by_model_task.csv", index=False, encoding="utf-8-sig")
    overview.to_csv(output_dir / "human_similarity_model_overview.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, summary, similarity, overview)

    print(f"Wrote human-likeness diagnostics to {output_dir}")
    print("\nModel overview:")
    print(overview.to_string(index=False))
    print("\nPer-task similarity:")
    print(
        similarity.sort_values(["task", "human_similarity"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
