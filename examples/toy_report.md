# Audit of toy_results.csv

## Summary

- Systems: 12
- Tasks: 40
- Missing cells: 1
- KR-20 reliability of the total score: not defined
- Tasks solved by every system: 2
- Tasks solved by no system: 1
- Tasks with negative discrimination: 1
- Tasks with discrimination between 0 and 0.1: 0

### Solve-rate distribution

| band | n_tasks |
| --- | --- |
| exactly 0 | 1 |
| 0.0 to 0.1 | 2 |
| 0.1 to 0.2 | 2 |
| 0.2 to 0.3 | 3 |
| 0.3 to 0.4 | 6 |
| 0.4 to 0.5 | 9 |
| 0.5 to 0.6 | 5 |
| 0.6 to 0.7 | 3 |
| 0.7 to 0.8 | 4 |
| 0.8 to 0.9 | 3 |
| 0.9 to 1.0 (below 1) | 0 |
| exactly 1 | 2 |

## Leaderboard (12 systems ranked)

| rank | system | solved | accuracy | ci | rank interval | bootstrap rank | tier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | system_k | 33/40 | 0.8250 | 0.681 to 0.913 | 1 to 6 | 1 to 4 | 1 |
| 2 | system_l | 33/40 | 0.8250 | 0.681 to 0.913 | 1 to 5 | 1 to 4 | 1 |
| 3 | system_j | 32/40 | 0.8000 | 0.652 to 0.895 | 1 to 7 | 1 to 4 | 1 |
| 4 | system_i | 30/40 | 0.7500 | 0.598 to 0.858 | 1 to 8 | 2 to 4 | 1 |
| 5 | system_h | 20/40 | 0.5000 | 0.352 to 0.648 | 3 to 12 | 5 to 8 | 2 |
| 6 | system_f | 19/40 | 0.4750 | 0.329 to 0.625 | 1 to 12 | 5 to 8 | 2 |
| 7 | system_g | 19/40 | 0.4750 | 0.329 to 0.625 | 2 to 12 | 5 to 8 | 2 |
| 8 | system_e | 15/40 | 0.3750 | 0.242 to 0.530 | 4 to 12 | 5 to 10 | 2 |
| 9 | system_d | 11/39 | 0.2821 | 0.165 to 0.438 | 5 to 12 | 8 to 11 | 2 |
| 10 | system_c | 9/40 | 0.2250 | 0.123 to 0.375 | 5 to 12 | 8 to 12 | 2 |
| 11 | system_a | 7/40 | 0.1750 | 0.087 to 0.319 | 5 to 12 | 10 to 12 | 2 |
| 12 | system_b | 7/40 | 0.1750 | 0.087 to 0.319 | 5 to 12 | 10 to 12 | 2 |

Rank intervals and tiers use exact McNemar tests on all 66 pairs with Holm correction at alpha = 0.05. 44 of those pairs are not distinguishable. The ranked systems fall into 2 tier(s).

### Adjacent ranks

11 of 11 adjacent pairs are statistically indistinguishable (Holm-corrected over the adjacent pairs).

| rank | system_upper | system_lower | diff | ci_low | ci_high | n_only_upper | n_only_lower | p_value | p_holm | distinguishable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | system_k | system_l | +0.0000 | -0.1569 | +0.1569 | 5 | 5 | 1 | 1 | False |
| 2 | system_l | system_j | +0.0250 | -0.1237 | +0.1737 | 5 | 4 | 1 | 1 | False |
| 3 | system_j | system_i | +0.0500 | -0.0895 | +0.1895 | 5 | 3 | 0.7266 | 1 | False |
| 4 | system_i | system_h | +0.2500 | +0.0971 | +0.4029 | 11 | 1 | 0.006348 | 0.06982 | False |
| 5 | system_h | system_f | +0.0250 | -0.1394 | +0.1894 | 6 | 5 | 1 | 1 | False |
| 6 | system_f | system_g | +0.0000 | -0.1857 | +0.1857 | 7 | 7 | 1 | 1 | False |
| 7 | system_g | system_e | +0.1000 | -0.0960 | +0.2960 | 10 | 6 | 0.4545 | 1 | False |
| 8 | system_e | system_d | +0.0769 | -0.0902 | +0.2440 | 7 | 4 | 0.5488 | 1 | False |
| 9 | system_d | system_c | +0.0513 | -0.1243 | +0.2269 | 7 | 5 | 0.7744 | 1 | False |
| 10 | system_c | system_a | +0.0500 | -0.0895 | +0.1895 | 5 | 3 | 0.7266 | 1 | False |
| 11 | system_a | system_b | +0.0000 | -0.0702 | +0.0702 | 1 | 1 | 1 | 1 | False |

### Smallest difference this benchmark can resolve

For two systems on 40 shared tasks, at the disagreement levels seen among the ranked systems:

| disagreement level | share of tasks where the two differ | smallest significant difference | detectable with 80% power |
| --- | --- | --- | --- |
| p25 | 0.3000 | 0.1697 | 0.2355 |
| median | 0.4000 | 0.1960 | 0.2720 |
| p75 | 0.5351 | 0.2267 | 0.3146 |

## Suspect tasks

Tasks that weaker systems solve more often than stronger ones (most negative first, up to 50 of 1):

| task | n_solved | n | solve_rate | discrimination | disc_ci_low | disc_ci_high | irt_discrimination |
| --- | --- | --- | --- | --- | --- | --- | --- |
| task_07 | 6 | 12 | 0.500 | -0.857 | -0.959 | -0.558 | -1.085 |

## Item response model (2PL)

- Converged: True after 24 iterations
- Tasks with a negative fitted slope: 1
- Effective number of informative tasks over all systems: 30.2
- Effective number of informative tasks over the ranked systems: 30.2
- Tasks holding 90% of the information over all systems: 28
- Tasks holding 90% of the information over the ranked systems: 28

### Test information

| ability | information |
| --- | --- |
| -3.0 | 1.94 |
| -2.5 | 2.90 |
| -2.0 | 4.21 |
| -1.5 | 5.83 |
| -1.0 | 7.61 |
| -0.5 | 9.31 |
| +0.0 | 10.51 |
| +0.5 | 10.53 |
| +1.0 | 9.09 |
| +1.5 | 6.97 |
| +2.0 | 5.02 |
| +2.5 | 3.51 |
| +3.0 | 2.42 |

## Systems with unusual response patterns

None flagged.
