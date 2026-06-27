# Human-Likeness Diagnostics

This report separates validity from human-likeness. Hard masks can make rollouts legal, but human-likeness is measured by whether the model matches the participant's behavioral distributions within each task.

## Model Ranking

| model                | human_distance | human_similarity | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| -------------------- | -------------- | ---------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| hardmask_recovery    | 1.3117         | 0.4405           | 0.2462         | 0.6775         | 0.0399    | 0.0485  | 0.0647      | 0.0292       | 0.0350      |
| rollout_soft         | 2.1368         | 0.3732           | 0.4499         | 1.1706         | 0.0393    | 0.0253  | 0.0950      | 0.0970       | 0.0442      |
| human_mask_old       | 2.4691         | 0.3644           | 0.5346         | 1.3796         | 0.0431    | 0.0386  | 0.0875      | 0.1166       | 0.0408      |
| task_specific_policy | 2.3873         | 0.3445           | 0.5032         | 1.2874         | 0.0390    | 0.0439  | 0.0558      | 0.1006       | 0.1167      |

## Per-Task Human Similarity

| model                | task | human_similarity | human_distance | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| -------------------- | ---- | ---------------- | -------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| rollout_soft         | 1    | 0.5761           | 0.7357         | 0.1043         | 0.4617         | 0.0253    | 0.0224  | 0.0378      | 0.0406       | 0.0000      |
| human_mask_old       | 1    | 0.5650           | 0.7700         | 0.1545         | 0.4164         | 0.0427    | 0.0206  | 0.0455      | 0.0177       | 0.0000      |
| hardmask_recovery    | 1    | 0.4952           | 1.0193         | 0.1408         | 0.6228         | 0.0359    | 0.0171  | 0.0623      | 0.0590       | 0.0000      |
| task_specific_policy | 1    | 0.4920           | 1.0326         | 0.1975         | 0.5962         | 0.0470    | 0.0251  | 0.0587      | 0.0271       | 0.0000      |
| hardmask_recovery    | 2    | 0.4862           | 1.0567         | 0.1955         | 0.5747         | 0.0433    | 0.0583  | 0.0198      | 0.0296       | 0.0000      |
| task_specific_policy | 2    | 0.3384           | 1.9551         | 0.4086         | 1.0479         | 0.0241    | 0.0676  | 0.0394      | 0.1316       | 0.0769      |
| human_mask_old       | 2    | 0.3295           | 2.0345         | 0.3956         | 0.9972         | 0.0212    | 0.0492  | 0.1306      | 0.1727       | 0.0000      |
| rollout_soft         | 2    | 0.2060           | 3.8537         | 0.8128         | 2.1356         | 0.0340    | 0.0136  | 0.1870      | 0.1993       | 0.0769      |
| hardmask_recovery    | 3    | 0.4269           | 1.3426         | 0.3229         | 0.8289         | 0.0417    | 0.0244  | 0.0172      | 0.0000       | 0.0000      |
| rollout_soft         | 3    | 0.2717           | 2.6800         | 0.5728         | 1.4732         | 0.0729    | 0.0336  | 0.1247      | 0.1360       | 0.0000      |
| task_specific_policy | 3    | 0.1689           | 4.9216         | 1.0219         | 2.7792         | 0.0525    | 0.0373  | 0.0684      | 0.2298       | 0.2500      |
| human_mask_old       | 3    | 0.1504           | 5.6503         | 1.2405         | 3.4313         | 0.0705    | 0.0426  | 0.1263      | 0.2610       | 0.0833      |
| rollout_soft         | 4    | 0.4390           | 1.2777         | 0.3097         | 0.6121         | 0.0249    | 0.0317  | 0.0304      | 0.0120       | 0.1000      |
| human_mask_old       | 4    | 0.4129           | 1.4219         | 0.3477         | 0.6735         | 0.0381    | 0.0418  | 0.0478      | 0.0149       | 0.0800      |
| task_specific_policy | 4    | 0.3788           | 1.6399         | 0.3848         | 0.7264         | 0.0322    | 0.0457  | 0.0569      | 0.0140       | 0.1400      |
| hardmask_recovery    | 4    | 0.3536           | 1.8282         | 0.3254         | 0.6835         | 0.0387    | 0.0943  | 0.1594      | 0.0281       | 0.1400      |

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