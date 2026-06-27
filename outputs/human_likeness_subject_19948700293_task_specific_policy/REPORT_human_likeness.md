# Human-Likeness Diagnostics

This report separates validity from human-likeness. Hard masks can make rollouts legal, but human-likeness is measured by whether the model matches the participant's behavioral distributions within each task.

## Model Ranking

| model                | human_distance | human_similarity | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| -------------------- | -------------- | ---------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| task_specific_policy | 1.9856         | 0.3789           | 0.3803         | 1.2066         | 0.0246    | 0.0498  | 0.0550      | 0.0649       | 0.0443      |

## Per-Task Human Similarity

| model                | task | human_similarity | human_distance | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| -------------------- | ---- | ---------------- | -------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| task_specific_policy | 1    | 0.4560           | 1.1928         | 0.1892         | 0.6429         | 0.0332    | 0.0492  | 0.0557      | 0.0913       | 0.0000      |
| task_specific_policy | 2    | 0.3885           | 1.5737         | 0.2946         | 0.9500         | 0.0475    | 0.0533  | 0.0427      | 0.0647       | 0.0000      |
| task_specific_policy | 3    | 0.1975           | 4.0638         | 0.7768         | 2.6984         | 0.0083    | 0.0307  | 0.0697      | 0.0893       | 0.1200      |
| task_specific_policy | 4    | 0.4735           | 1.1120         | 0.2606         | 0.5351         | 0.0094    | 0.0659  | 0.0518      | 0.0145       | 0.0571      |

## What To Inspect Next

- If `length_w1_norm` / `excess_w1_norm` is high, the model is too short-path-like or too wandering compared with the participant.
- If `action_tv` is high, the model uses different direction priors than the participant.
- If `turn_tv` is high, the model's local movement style differs: too straight, too turn-heavy, or too many reversals.
- If `revisit_abs` is high, the model does not match human re-checking / looping behavior.
- If `progress_abs` is high, the model approaches the goal at a different rate from the participant.
- If `reached_gap` is high, the model still fails basic task completion and should not be selected even if other features match.

## Recommended Training Direction

1. Keep hard illegal action masks as an environment constraint, not as the human-likeness objective.
2. Select checkpoints with task-balanced human similarity, with a minimum reached-goal floor per task.
3. Replace one-hot shortest recovery labels with soft recovery targets that mix shortest recovery and the participant's local action prior.
4. Add task-balanced rollout diagnostics to every experiment, especially for Task 2 and Task 3.