# Human-Likeness Diagnostics

This report separates validity from human-likeness. Hard masks can make rollouts legal, but human-likeness is measured by whether the model matches the participant's behavioral distributions within each task.

## Model Ranking

| model                    | human_distance | human_similarity | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| ------------------------ | -------------- | ---------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| hardmask_recovery        | 1.3072         | 0.4412           | 0.2449         | 0.6745         | 0.0399    | 0.0485  | 0.0647      | 0.0292       | 0.0350      |
| rollout_soft             | 2.1307         | 0.3735           | 0.4483         | 1.1665         | 0.0393    | 0.0253  | 0.0950      | 0.0970       | 0.0442      |
| human_mask_soft_recovery | 2.4657         | 0.3650           | 0.5337         | 1.3772         | 0.0431    | 0.0386  | 0.0875      | 0.1166       | 0.0408      |
| context                  | 4.5629         | 0.1908           | 0.8669         | 2.6238         | 0.0437    | 0.0511  | 0.1547      | 0.1864       | 0.1826      |
| old_minimal              | 5.1247         | 0.1691           | 1.0523         | 2.8032         | 0.0381    | 0.0673  | 0.1614      | 0.1781       | 0.3103      |

## Per-Task Human Similarity

| model                    | task | human_similarity | human_distance | length_w1_norm | excess_w1_norm | action_tv | turn_tv | revisit_abs | progress_abs | reached_gap |
| ------------------------ | ---- | ---------------- | -------------- | -------------- | -------------- | --------- | ------- | ----------- | ------------ | ----------- |
| rollout_soft             | 1    | 0.5746           | 0.7404         | 0.1054         | 0.4650         | 0.0253    | 0.0224  | 0.0378      | 0.0406       | 0.0000      |
| human_mask_soft_recovery | 1    | 0.5660           | 0.7668         | 0.1529         | 0.4151         | 0.0427    | 0.0206  | 0.0455      | 0.0177       | 0.0000      |
| hardmask_recovery        | 1    | 0.4943           | 1.0230         | 0.1419         | 0.6251         | 0.0359    | 0.0171  | 0.0623      | 0.0590       | 0.0000      |
| old_minimal              | 1    | 0.2058           | 3.8594         | 0.6491         | 2.5691         | 0.0235    | 0.0821  | 0.0496      | 0.0399       | 0.1667      |
| context                  | 1    | 0.1243           | 7.0481         | 1.0109         | 4.7246         | 0.0434    | 0.0897  | 0.1550      | 0.2047       | 0.2500      |
| hardmask_recovery        | 2    | 0.4878           | 1.0501         | 0.1940         | 0.5698         | 0.0433    | 0.0583  | 0.0198      | 0.0296       | 0.0000      |
| human_mask_soft_recovery | 2    | 0.3302           | 2.0284         | 0.3948         | 0.9921         | 0.0212    | 0.0492  | 0.1306      | 0.1727       | 0.0000      |
| context                  | 2    | 0.2363           | 3.2328         | 0.6454         | 1.7040         | 0.0423    | 0.0355  | 0.1559      | 0.2067       | 0.0769      |
| rollout_soft             | 2    | 0.2068           | 3.8363         | 0.8088         | 2.1230         | 0.0340    | 0.0136  | 0.1870      | 0.1993       | 0.0769      |
| old_minimal              | 2    | 0.1249           | 7.0078         | 1.4475         | 3.8367         | 0.0361    | 0.0295  | 0.2514      | 0.3184       | 0.3846      |
| hardmask_recovery        | 3    | 0.4286           | 1.3333         | 0.3202         | 0.8229         | 0.0417    | 0.0244  | 0.0172      | 0.0000       | 0.0000      |
| rollout_soft             | 3    | 0.2721           | 2.6751         | 0.5716         | 1.4698         | 0.0729    | 0.0336  | 0.1247      | 0.1360       | 0.0000      |
| context                  | 3    | 0.1885           | 4.3043         | 0.8542         | 2.3137         | 0.0633    | 0.0500  | 0.2479      | 0.2261       | 0.0833      |
| old_minimal              | 3    | 0.1606           | 5.2252         | 0.9826         | 2.7231         | 0.0782    | 0.0741  | 0.2669      | 0.2412       | 0.2500      |
| human_mask_soft_recovery | 3    | 0.1504           | 5.6490         | 1.2405         | 3.4301         | 0.0705    | 0.0426  | 0.1263      | 0.2610       | 0.0833      |
| rollout_soft             | 4    | 0.4403           | 1.2710         | 0.3074         | 0.6082         | 0.0249    | 0.0317  | 0.0304      | 0.0120       | 0.1000      |
| human_mask_soft_recovery | 4    | 0.4134           | 1.4187         | 0.3466         | 0.6716         | 0.0381    | 0.0418  | 0.0478      | 0.0149       | 0.0800      |
| hardmask_recovery        | 4    | 0.3543           | 1.8225         | 0.3234         | 0.6802         | 0.0387    | 0.0943  | 0.1594      | 0.0281       | 0.1400      |
| context                  | 4    | 0.2143           | 3.6663         | 0.9574         | 1.7527         | 0.0259    | 0.0293  | 0.0601      | 0.1080       | 0.3200      |
| old_minimal              | 4    | 0.1850           | 4.4063         | 1.1300         | 2.0839         | 0.0145    | 0.0836  | 0.0779      | 0.1128       | 0.4400      |

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