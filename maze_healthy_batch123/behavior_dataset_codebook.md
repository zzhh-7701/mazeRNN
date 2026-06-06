# 迷宫寻宝行为数据集说明

本文档说明 trial-level 行为数据集的文件位置、字段格式和编码方式。实验任务背景请同时参考 `experiment_paradigm.md`。

## 数据文件

行为数据由 MATLAB 脚本生成：

`extract_trial_level.m`

主数据文件：

- `trial_level.mat`
- `trial_level.csv`

诊断文件：

- `trial_level_decode_diagnostics.csv`
- `trial_level_decode_summary.csv`

`trial_level.mat` 中保存 MATLAB table `trial_data`。长变量保存为原生 cell。  
`trial_level.csv` 长变量以 JSON 字符串保存，便于跨软件读取。

## 行定义

每一行对应一条 trial-level 记录。

普通试次对应一次从 `start` 到 `goal` 的寻路。Task5 replan试次在原始保存中被拆成两行：

- `replan = 0`：plan 阶段，从起点移动到目标
- `replan = 1`：replan 阶段，从目标返回原起点。

## 批次和迷宫

共有 7 个实验批次。当前行为表中的 `batch` 按被试最早一条试次的创建日期 `createdAt` 距离哪个批次首日最近来判定。

批次首日和对应迷宫编号：

| batch | 首日 | 迷宫编号 |
|---:|---|---|
| 1 | 2024-08-10 | 1 |
| 2 | 2024-08-15 | 1 |
| 3 | 2024-09-06 | 2 |
| 4 | 2024-09-13 | 3 |
| 5 | 2024-10-06 | 3 |
| 6 | 2024-12-14 | 3 |
| 7 | 2025-08-25 & 26 | 2 |

迷宫所使用的 Mazeset 版本：

- batch 1-6 使用 `D:\jxwang\data\maze_healthy\maze_structure\v1\maze{maze}`。
- batch 7 使用 `D:\jxwang\data\maze_healthy\maze_structure\v2\maze{maze}`。

## Task 和 Block 编码

`task` 是语义任务编号，取值 1-5。

| task | 含义 |
|---:|---|
| 1 | Fixed start and goal with local visibility |
| 2 | Fixed goal, changing start with local visibility |
| 3 | Changing start and goal with local visibility |
| 4 | Changing start and goal with hidden maze |
| 5 | Plan / replan task with hidden maze|

`block` 是跨天统一编号：

| block | day | task | 含义 |
|---:|---:|---:|---|
| 1 | 1 | 1 或 4 | 第一天第 1 个 block |
| 2 | 1 | 2 或 4 | 第一天第 2 个 block |
| 3 | 1 | 3 或 4 | 第一天第 3 个 block |
| 4 | 1 | 4 | 第一天第 4 个 block |
| 5 | 2 | 4 | 第二天 Task4 block 2 |
| 6 | 2 | 4 | 第二天 Task4 block 3 |
| 7 | 2 | 4 | 第二天 Task4 block 4 |
| 8 | 2 | 5 | 第二天 Task5 plan/replan |

对于 batch 1-6，第一天的 `raw_phase = 0,1,2,3` 分别映射为 `task = 1,2,3,4` 和 `block = 1,2,3,4`。

对于 batch 7，第一天的 4 个 block 均为隐藏迷宫任务，因此 `raw_phase = 0,1,2,3` 均映射为 `task = 4`，但 `block` 仍为 1-4。

第二天中，`raw_phase = 0,1,2` 映射为 Task4 的 `block = 5,6,7`；`raw_phase = 4` 映射为 Task5 的 `block = 8`。

## 主表字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `subid` | numeric | 被试 ID |
| `batch` | numeric | 实验批次，1-7。按被试最早一条试次记录的时间判定 |
| `maze` | numeric | 迷宫编号，1-3 |
| `day` | numeric | 实验天数，1 或 2 |
| `createdat` | numeric | MATLAB datenum 格式的试次创建时间 |
| `task` | numeric | 任务编号，1-5 |
| `block` | numeric | 跨天的 block 编号，1-8。异常或代码 bug 编码为 NaN |
| `raw_phase` | numeric | 原始数据中 `TaskPhase` 编码，通常从 0 开始（用于校验） |
| `raw_trial` | numeric | 原始数据中 `TaskTrial` 编码，通常从 0 开始（用于校验），遇到新的 block 或被试当前 block 重新开始会重置。|
| `trial` | numeric | 重构后的试次编号，从 1 开始；遇到新的 block 或被试当前 block 重新开始会重置。 |
| `rep` | numeric | 当前行相对同一被试同一天上一行的 start/goal 是否发生变化；变化为 1，不变为 0（可用于判断 mini-block）|
| `trial_set` | numeric | 当前行使用的 MazeSet 配置编号（用于校验） |
| `replan` | numeric | 是否为 Task5 replan 阶段；0 为 plan，1 为 replan |
| `start` | numeric | 当前行路径起点 state。replan 行中为 goal-to-start 路径的起点，即原试次的 goal |
| `goal` | numeric | 当前行路径终点 state。replan 行中为 goal-to-start 路径的终点，即原试次的 start |
| `short_path` | cell / JSON | 实验代码保存的最短路径 state 序列 |
| `action` | cell / JSON | 被试按键动作序列 |
| `hits` | cell / JSON | 每一步 action 是否撞墙，true/false |
| `rt_raw` | cell / JSON | 未修正反应时序列，单位秒 |
| `rt` | cell / JSON | 修正后反应时序列，单位秒 |
| `true_path` | cell / JSON | 根据 `start`、`action` 和当前迷宫结构模拟出的实际 state 序列。长度通常为 `numel(action)+1` |
| `maze_wall` | cell / JSON | replan 行保存修改后的墙结构；普通行为空 |

## State 编码

迷宫为 7 x 7 网格，state 编号为 0-48。

state 与网格坐标关系：

```matlab
state = 7 * col + row;
```

其中 `col` 和 `row` 均从 0 开始。实验 JS 坐标转换为 state 时使用：

```matlab
col = x / 2;
row = 6 - z / 2;
state = 7 * col + row;
```

因此 state 编号按列优先排列：

- 同一列内，`row` 从 0 到 6。
- 向右移动通常使 state 增加 7。
- 向下移动通常使 state 增加 1。

## Action 编码

`action` 由原始 `PlayerTrialKey` 中的按键解析得到。

| action | 按键 | 方向 | state 变化 |
|---:|---|---|---:|
| 0 | A | left | -7 |
| 1 | W | up | -1 |
| 2 | D | right | +7 |
| 3 | S | down | +1 |

如果某一步撞墙，`true_path` 中该步前后的 state 会相同，`hits` 对应位置为 true。

## RT 编码和修正

`rt_raw` 来自原始 `PlayerTrialRT` 的相邻时间戳差值：

```matlab
rt_raw = diff(PlayerTrialRT) / 1000;
```

单位为秒。

`rt` 是修正后的反应时：

- 普通试次：`numel(rt_raw) = numel(action)`。
- replan：`numel(rt_raw) = numel(action) + 1`，先合并前两个 interval，以抵消 replan 保存机制多出的一个 rt 元素。
- 如果 rt 保存的数量超出 action 数量，这是由于实验代码本身的问题导致，使用 rt 的末尾对齐 `action`，并在 diagnostics 中标记 `rt_was_aligned = 1`。
- 撞墙后 agent 有约 1 秒动画，因此对撞墙后的下一步 rt 减 1 秒。

## Maze Wall 编码

基础迷宫墙结构来自：

`maze_wall.mat` 或 `maze_wall.csv`

`maze_wall` 使用 4 个向量表示可移动方向限制：

```matlab
{walls_l, walls_u, walls_r, walls_d}
```

`maze_wall.csv` 一行对应一个基础迷宫，字段为：

| 字段 | 说明 |
|---|---|
| `maze` | 迷宫编号，1-3。 |
| `walls_l` | 左方向 wall 向量，JSON 数组。 |
| `walls_u` | 上方向 wall 向量，JSON 数组。 |
| `walls_r` | 右方向 wall 向量，JSON 数组。 |
| `walls_d` | 下方向 wall 向量，JSON 数组。 |

普通行的 `maze_wall` 为空，表示该试次使用基础迷宫结构。replan 行保存当前试次发生局部 block/unblock 后的完整墙结构。

replan 的结构变化只对当前 replan 行有效。下一试次不继承该变化。

## CSV 中的长字段

CSV 的表头与 MATLAB table 保持一致。以下字段在 CSV 中是 JSON 字符串：

- `short_path`
- `action`
- `hits`
- `rt_raw`
- `rt`
- `true_path`
- `maze_wall`

例如在 MATLAB 中可使用：

```matlab
x = jsondecode(trial_level.short_path{1});
```

在 Python 中可使用：

```python
import json
x = json.loads(row["short_path"])
```

空字符串表示该字段为空，例如普通试次的 `maze_wall`。

## Diagnostics 文件

主表只保存分析所需字段。Diagnostics 文件是用于诊断实验代码本身问题和校验迷宫结构的。

`trial_level_decode_diagnostics.csv` 与主表同样是一行一个 7norm trial。常用字段包括：

| 字段 | 说明 |
|---|---|
| `n_action` | action 数量 |
| `n_hit` | 撞墙次数 |
| `total_rt_raw` | 未修正 rt 总和 |
| `total_rt` | 修正后 rt 总和 |
| `rt_length_ok` | rt 长度是否与 action 匹配 |
| `rt_was_aligned` | rt 是否进行末尾对齐 |
| `short_path_valid_in_maze` | `short_path` 是否在当前迷宫结构中合法 |
| `short_path_is_shortest` | `short_path` 是否确实为最短路径 |
| `true_path_end_ok` | `true_path` 是否最终到达 `goal` |
| `hit_consistent` | `hits` 是否与 `true_path` 中的停留一致 |
| `trialset_match_ok` | `start/goal` 是否与 trial-set 文件匹配 |
| `batch_by_first_trial` | 按被试最早 7norm 试次时间判定的 batch |
| `batch_by_objectid_folder` | 按 objectId 所在导出文件夹判定的 batch，仅用于诊断 |
| `batch_conflict` | 上述两种 batch 判定是否不一致 |
| `first_trial_createdat` | 被试最早 7norm 试次时间，MATLAB datenum |
| `days_from_first_trial` | 当前试次距被试最早试次的天数 |
| `days_to_nearest_batch_start` | 被试最早试次距最近批次首日的天数 |
| `post_goal_extra_action` | 首次到达 goal 后是否仍有额外 action |
| `n_post_goal_action` | 到达 goal 后额外 action 数 |
| `post_goal_extra_hit` | 到达 goal 后额外 action 中是否包含撞墙 |
| `decode_error` | 解码错误信息；为空表示无错误 |

`trial_level_decode_summary.csv` 按 `batch/day/task/block/replan` 汇总上述异常数量。

## 已知特殊情况

当前提取脚本尽量保留原始试次行，不在主表中自动剔除异常试次或异常被试。后续分析应根据 diagnostics 和研究问题自行制定剔除规则。

已知需要关注的情况包括：

- 少数被试存在 `batch_conflict = 1`，通常是因为，每一批次的实验数据是累加导出的，批次的文件夹和实际最早试次时间不一致导致。主表的 `batch` 已按最早试次时间固定。
- 少数试次存在到达 goal 后仍多记录 action 的情况，可用 `post_goal_extra_action` 标记。
- 极少数试次存在路径或撞墙异常（比如穿墙），可用 `true_path_end_ok`、`hit_consistent`、`trialset_match_ok` 检查。
- 第 7 批中存在一个特殊被试在第一天进入了 Task5 replan；该类行会保留在主表中，`task = 5` 且 `block` 为 NaN。

## 使用建议

建议先使用 diagnostics 文件确定分析剔除标准，再从主表筛选试次或 subject。主表中的 `raw_phase`、`raw_trial`、`trial_set` 和 diagnostics 中的质量检查字段应保留到最终分析前，以便回溯异常来源。
