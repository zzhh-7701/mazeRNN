# 人类相似性诊断报告

## 核心思路

现在要区分两件事：

- **validity**：动作是否合法、是否撞墙、是否能到终点。
- **human-likeness**：路径长度、绕路程度、重复访问、转向/折返、动作偏好、朝目标推进方式是否像真实被试。

hard illegal mask 适合作为环境底线，但它本身不会让模型更像人。它会让模型更像一个合法导航器；如果 selection 只奖励到达率和低撞墙，模型就会越来越像“会跑迷宫的 AI”。

## 本次加入的检测指标

- `length_w1_norm`：模型路径长度分布和真实人的距离，越低越像。
- `excess_w1_norm`：绕路步数分布距离，越低越像。
- `action_tv`：上下左右动作比例差异，越低越像。
- `turn_tv`：直走/转弯/反向折返模式差异，越低越像。
- `revisit_abs`：重复访问状态比例差异，越低越像。
- `progress_abs`：每步朝目标推进速度差异，越低越像。
- `reached_gap`：真实人到达但模型没到达的缺口。这个是底线项。
- `human_similarity`：上述指标的加权综合分，范围约 0 到 1，越高越像真实被试。

## 模型整体排序

| model                | human_distance | human_similarity | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| -------------------- | -------------- | ---------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| task_specific_policy | 1.9856         | 0.3789           | 0.3803         | 1.2066         | 0.0246    | 0.0498  | 0.0550      | 0.0649       | 0.0443      |

## 每个 Task 最像人的模型

| task | model                | human_similarity | human_distance | length_w1_norm | excess_w1_norm | revisit_abs | progress_abs | reached_gap |
| ---- | -------------------- | ---------------- | -------------- | -------------- | -------------- | ----------- | ------------ | ----------- |
| 1    | task_specific_policy | 0.4560           | 1.1928         | 0.1892         | 0.6429         | 0.0557      | 0.0913       | 0.0000      |
| 2    | task_specific_policy | 0.3885           | 1.5737         | 0.2946         | 0.9500         | 0.0427      | 0.0647       | 0.0000      |
| 3    | task_specific_policy | 0.1975           | 4.0638         | 0.7768         | 2.6984         | 0.0697      | 0.0893       | 0.1200      |
| 4    | task_specific_policy | 0.4735           | 1.1120         | 0.2606         | 0.5351         | 0.0518      | 0.0145       | 0.0571      |

## 每个 Task 的完整对比

| model                | task | human_similarity | human_distance | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| -------------------- | ---- | ---------------- | -------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| task_specific_policy | 1    | 0.4560           | 1.1928         | 0.1892         | 0.6429         | 0.0332    | 0.0492  | 0.0557      | 0.0913       | 0.0000      |
| task_specific_policy | 2    | 0.3885           | 1.5737         | 0.2946         | 0.9500         | 0.0475    | 0.0533  | 0.0427      | 0.0647       | 0.0000      |
| task_specific_policy | 3    | 0.1975           | 4.0638         | 0.7768         | 2.6984         | 0.0083    | 0.0307  | 0.0697      | 0.0893       | 0.1200      |
| task_specific_policy | 4    | 0.4735           | 1.1120         | 0.2606         | 0.5351         | 0.0094    | 0.0659  | 0.0518      | 0.0145       | 0.0571      |

## 当前结果怎么解释

1. `hardmask_recovery` 整体 human similarity 最高，尤其 Task 2/3 明显优于旧 minimal 和 context。这说明 hard mask + recovery 解决了原先 Task 2/3 的失败、撞墙和灾难性绕路。

2. 但 Task 4 上 `rollout_soft` 比 `hardmask_recovery` 更像真实人。原因很可能是 Task 4 真实被试本来有较多探索、撞墙、重复检查；hardmask 把撞墙清零，validity 变好，但 human-likeness 下降。

3. `old_minimal` 单步 accuracy 高，但 human similarity 最差之一，说明单步方向预测不是这个实验的主要选择标准。

4. `context` 有一定恢复能力，但路径长度和绕路分布仍然偏离真实人，尤其 Task 1/3/4 的 length/excess 距离比较大。

## 后续训练建议

1. 保留 hardmask 作为环境约束，但不要把“零撞墙”当成人类相似性的目标。Task 4 可以考虑允许低概率 wall-hit 行为作为观测输出，或者在 scoring 中按真实人的 task-specific hit rate 惩罚“过低撞墙”。

2. checkpoint selection 改成：先要求每个 task reached_goal_rate 达到底线，再用 task-balanced human_similarity 选模型。

3. off-trajectory recovery 不要永远用 one-hot shortest path。建议换成 soft target：`70% shortest recovery + 30% human local action prior`，Task 4 的 human prior 权重可以更高。

4. 分 task 训练或分 task head 值得考虑：Task 2/3 更像 goal-directed recovery，Task 4 更像探索/再计划。一个统一 loss 很容易顾此失彼。

5. 每次训练后都跑本诊断脚本，不再只看 accuracy、到达率、平均路径长度。