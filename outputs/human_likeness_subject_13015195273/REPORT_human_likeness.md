# Human-Likeness Diagnostics

This report separates validity from human-likeness. Hard masks can make rollouts legal, but human-likeness is measured by whether the model matches the participant's behavioral distributions within each task.

## Model Ranking

| model             | human_distance | human_similarity | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| ----------------- | -------------- | ---------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| hardmask_recovery | 1.3117         | 0.4405           | 0.2462         | 0.6775         | 0.0399    | 0.0485  | 0.0647      | 0.0292       | 0.0350      |
| rollout_soft      | 2.1368         | 0.3732           | 0.4499         | 1.1706         | 0.0393    | 0.0253  | 0.0950      | 0.0970       | 0.0442      |
| context           | 4.5789         | 0.1903           | 0.8702         | 2.6359         | 0.0437    | 0.0511  | 0.1547      | 0.1864       | 0.1826      |
| old_minimal       | 5.1396         | 0.1686           | 1.0554         | 2.8144         | 0.0381    | 0.0673  | 0.1614      | 0.1781       | 0.3103      |

## Per-Task Human Similarity

| model             | task | human_similarity | human_distance | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| ----------------- | ---- | ---------------- | -------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| rollout_soft      | 1    | 0.5761           | 0.7357         | 0.1043         | 0.4617         | 0.0253    | 0.0224  | 0.0378      | 0.0406       | 0.0000      |
| hardmask_recovery | 1    | 0.4952           | 1.0193         | 0.1408         | 0.6228         | 0.0359    | 0.0171  | 0.0623      | 0.0590       | 0.0000      |
| old_minimal       | 1    | 0.2043           | 3.8947         | 0.6548         | 2.5975         | 0.0235    | 0.0821  | 0.0496      | 0.0399       | 0.1667      |
| context           | 1    | 0.1239           | 7.0718         | 1.0137         | 4.7449         | 0.0434    | 0.0897  | 0.1550      | 0.2047       | 0.2500      |
| hardmask_recovery | 2    | 0.4862           | 1.0567         | 0.1955         | 0.5747         | 0.0433    | 0.0583  | 0.0198      | 0.0296       | 0.0000      |
| context           | 2    | 0.2350           | 3.2560         | 0.6515         | 1.7199         | 0.0423    | 0.0355  | 0.1559      | 0.2067       | 0.0769      |
| rollout_soft      | 2    | 0.2060           | 3.8537         | 0.8128         | 2.1356         | 0.0340    | 0.0136  | 0.1870      | 0.1993       | 0.0769      |
| old_minimal       | 2    | 0.1247           | 7.0187         | 1.4504         | 3.8441         | 0.0361    | 0.0295  | 0.2514      | 0.3184       | 0.3846      |
| hardmask_recovery | 3    | 0.4269           | 1.3426         | 0.3229         | 0.8289         | 0.0417    | 0.0244  | 0.0172      | 0.0000       | 0.0000      |
| rollout_soft      | 3    | 0.2717           | 2.6800         | 0.5728         | 1.4732         | 0.0729    | 0.0336  | 0.1247      | 0.1360       | 0.0000      |
| context           | 3    | 0.1881           | 4.3165         | 0.8564         | 2.3233         | 0.0633    | 0.0500  | 0.2479      | 0.2261       | 0.0833      |
| old_minimal       | 3    | 0.1604           | 5.2345         | 0.9848         | 2.7297         | 0.0782    | 0.0741  | 0.2669      | 0.2412       | 0.2500      |
| rollout_soft      | 4    | 0.4390           | 1.2777         | 0.3097         | 0.6121         | 0.0249    | 0.0317  | 0.0304      | 0.0120       | 0.1000      |
| hardmask_recovery | 4    | 0.3536           | 1.8282         | 0.3254         | 0.6835         | 0.0387    | 0.0943  | 0.1594      | 0.0281       | 0.1400      |
| context           | 4    | 0.2141           | 3.6712         | 0.9591         | 1.7555         | 0.0259    | 0.0293  | 0.0601      | 0.1080       | 0.3200      |
| old_minimal       | 4    | 0.1848           | 4.4106         | 1.1315         | 2.0864         | 0.0145    | 0.0836  | 0.0779      | 0.1128       | 0.4400      |

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