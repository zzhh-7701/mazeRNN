# Human-Likeness Diagnostics

This report separates validity from human-likeness. Hard masks can make rollouts legal, but human-likeness is measured by whether the model matches the participant's behavioral distributions within each task.

## Model Ranking

| model                    | human_distance | human_similarity | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| ------------------------ | -------------- | ---------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| hardmask_recovery        | 1.3041         | 0.4418           | 0.2439         | 0.6726         | 0.0399    | 0.0485  | 0.0647      | 0.0292       | 0.0350      |
| rollout_soft             | 2.1271         | 0.3736           | 0.4472         | 1.1643         | 0.0393    | 0.0253  | 0.0950      | 0.0970       | 0.0442      |
| human_mask_humanselect   | 2.4620         | 0.3656           | 0.5330         | 1.3744         | 0.0431    | 0.0386  | 0.0875      | 0.1166       | 0.0408      |
| human_mask_rolloutselect | 2.3042         | 0.3530           | 0.4868         | 1.2720         | 0.0402    | 0.0137  | 0.0894      | 0.0891       | 0.0859      |
| context                  | 4.5522         | 0.1912           | 0.8650         | 2.6155         | 0.0437    | 0.0511  | 0.1547      | 0.1864       | 0.1826      |
| old_minimal              | 5.1153         | 0.1694           | 1.0504         | 2.7961         | 0.0381    | 0.0673  | 0.1614      | 0.1781       | 0.3103      |

## Per-Task Human Similarity

| model                    | task | human_similarity | human_distance | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| ------------------------ | ---- | ---------------- | -------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| rollout_soft             | 1    | 0.5737           | 0.7430         | 0.1057         | 0.4672         | 0.0253    | 0.0224  | 0.0378      | 0.0406       | 0.0000      |
| human_mask_humanselect   | 1    | 0.5675           | 0.7620         | 0.1519         | 0.4116         | 0.0427    | 0.0206  | 0.0455      | 0.0177       | 0.0000      |
| human_mask_rolloutselect | 1    | 0.5477           | 0.8259         | 0.1147         | 0.5121         | 0.0327    | 0.0104  | 0.0471      | 0.0511       | 0.0000      |
| hardmask_recovery        | 1    | 0.4936           | 1.0258         | 0.1424         | 0.6273         | 0.0359    | 0.0171  | 0.0623      | 0.0590       | 0.0000      |
| old_minimal              | 1    | 0.2068           | 3.8362         | 0.6453         | 2.5505         | 0.0235    | 0.0821  | 0.0496      | 0.0399       | 0.1667      |
| context                  | 1    | 0.1245           | 7.0345         | 1.0097         | 4.7124         | 0.0434    | 0.0897  | 0.1550      | 0.2047       | 0.2500      |
| hardmask_recovery        | 2    | 0.4890           | 1.0449         | 0.1929         | 0.5660         | 0.0433    | 0.0583  | 0.0198      | 0.0296       | 0.0000      |
| human_mask_humanselect   | 2    | 0.3308           | 2.0234         | 0.3941         | 0.9879         | 0.0212    | 0.0492  | 0.1306      | 0.1727       | 0.0000      |
| human_mask_rolloutselect | 2    | 0.2434           | 3.1083         | 0.6472         | 1.6720         | 0.0288    | 0.0147  | 0.1741      | 0.1997       | 0.0769      |
| context                  | 2    | 0.2371           | 3.2172         | 0.6414         | 1.6931         | 0.0423    | 0.0355  | 0.1559      | 0.2067       | 0.0769      |
| rollout_soft             | 2    | 0.2072           | 3.8264         | 0.8066         | 2.1159         | 0.0340    | 0.0136  | 0.1870      | 0.1993       | 0.0769      |
| old_minimal              | 2    | 0.1250           | 7.0002         | 1.4455         | 3.8314         | 0.0361    | 0.0295  | 0.2514      | 0.3184       | 0.3846      |
| hardmask_recovery        | 3    | 0.4298           | 1.3269         | 0.3181         | 0.8190         | 0.0417    | 0.0244  | 0.0172      | 0.0000       | 0.0000      |
| rollout_soft             | 3    | 0.2723           | 2.6725         | 0.5706         | 1.4685         | 0.0729    | 0.0336  | 0.1247      | 0.1360       | 0.0000      |
| human_mask_rolloutselect | 3    | 0.2050           | 3.8770         | 0.8542         | 2.2599         | 0.0726    | 0.0068  | 0.0829      | 0.0905       | 0.1667      |
| context                  | 3    | 0.1889           | 4.2941         | 0.8525         | 2.3056         | 0.0633    | 0.0500  | 0.2479      | 0.2261       | 0.0833      |
| old_minimal              | 3    | 0.1607           | 5.2213         | 0.9818         | 2.7202         | 0.0782    | 0.0741  | 0.2669      | 0.2412       | 0.2500      |
| human_mask_humanselect   | 3    | 0.1505           | 5.6460         | 1.2400         | 3.4277         | 0.0705    | 0.0426  | 0.1263      | 0.2610       | 0.0833      |
| rollout_soft             | 4    | 0.4412           | 1.2665         | 0.3058         | 0.6056         | 0.0249    | 0.0317  | 0.0304      | 0.0120       | 0.1000      |
| human_mask_rolloutselect | 4    | 0.4157           | 1.4055         | 0.3310         | 0.6438         | 0.0268    | 0.0228  | 0.0536      | 0.0153       | 0.1000      |
| human_mask_humanselect   | 4    | 0.4138           | 1.4165         | 0.3459         | 0.6703         | 0.0381    | 0.0418  | 0.0478      | 0.0149       | 0.0800      |
| hardmask_recovery        | 4    | 0.3548           | 1.8186         | 0.3221         | 0.6779         | 0.0387    | 0.0943  | 0.1594      | 0.0281       | 0.1400      |
| context                  | 4    | 0.2145           | 3.6631         | 0.9562         | 1.7509         | 0.0259    | 0.0293  | 0.0601      | 0.1080       | 0.3200      |
| old_minimal              | 4    | 0.1851           | 4.4036         | 1.1290         | 2.0824         | 0.0145    | 0.0836  | 0.0779      | 0.1128       | 0.4400      |

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