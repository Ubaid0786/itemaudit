# SWE-bench Verified: generic itemaudit report

## Summary

- Systems: 173
- Tasks: 500
- Missing cells: 0
- KR-20 reliability of the total score: 0.9938
- Tasks solved by every system: 0
- Tasks solved by no system: 26
- Tasks with negative discrimination: 7
- Tasks with discrimination between 0 and 0.1: 25

### Solve-rate distribution

| band | n_tasks |
| --- | --- |
| exactly 0 | 26 |
| 0.0 to 0.1 | 59 |
| 0.1 to 0.2 | 27 |
| 0.2 to 0.3 | 24 |
| 0.3 to 0.4 | 21 |
| 0.4 to 0.5 | 46 |
| 0.5 to 0.6 | 47 |
| 0.6 to 0.7 | 48 |
| 0.7 to 0.8 | 65 |
| 0.8 to 0.9 | 76 |
| 0.9 to 1.0 (below 1) | 61 |
| exactly 1 | 0 |

## Leaderboard (20 systems ranked)

| rank | system | solved | accuracy | ci | rank interval | bootstrap rank | tier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20251205_sonar-foundation-agent_claude-opus-4-5 | 396/500 | 0.7920 | 0.754 to 0.825 | 1 to 20 | 1 to 6 | 1 |
| 2 | 20251215_livesweagent_claude-opus-4-5 | 396/500 | 0.7920 | 0.754 to 0.825 | 1 to 20 | 1 to 7 | 1 |
| 3 | 20250928_trae_doubao_seed_code | 394/500 | 0.7880 | 0.750 to 0.822 | 1 to 20 | 1 to 9 | 1 |
| 4 | 20251127_openhands_claude-opus-4-5 | 388/500 | 0.7760 | 0.737 to 0.810 | 1 to 20 | 2 to 14 | 1 |
| 5 | 20251120_livesweagent_gemini-3-pro-preview | 387/500 | 0.7740 | 0.735 to 0.808 | 1 to 20 | 2 to 15 | 1 |
| 6 | 20250804_epam-ai-run-claude-4-sonnet | 384/500 | 0.7680 | 0.729 to 0.803 | 1 to 20 | 2 to 17 | 1 |
| 7 | 20250902_atlassian-rovo-dev | 384/500 | 0.7680 | 0.729 to 0.803 | 1 to 20 | 2 to 16 | 1 |
| 8 | 20260217_mini-v2.0.0_claude-4-5-opus-high | 384/500 | 0.7680 | 0.729 to 0.803 | 1 to 20 | 3 to 17 | 1 |
| 9 | 20250819_ACoder | 382/500 | 0.7640 | 0.725 to 0.799 | 1 to 20 | 3 to 18 | 1 |
| 10 | 20260217_mini-v2.0.0_gemini-3-flash-high | 379/500 | 0.7580 | 0.719 to 0.793 | 1 to 20 | 4 to 20 | 1 |
| 11 | 20260217_mini-v2.0.0_minimax-2-5-high | 379/500 | 0.7580 | 0.719 to 0.793 | 1 to 20 | 4 to 20 | 1 |
| 12 | 20250901_warp | 378/500 | 0.7560 | 0.716 to 0.792 | 1 to 20 | 4 to 20 | 1 |
| 13 | 20260217_mini-v2.0.0_claude-4-6-opus | 378/500 | 0.7560 | 0.716 to 0.792 | 1 to 20 | 5 to 20 | 1 |
| 14 | 20250612_trae | 376/500 | 0.7520 | 0.712 to 0.788 | 1 to 20 | 6 to 20 | 1 |
| 15 | 20250731_harness_ai | 374/500 | 0.7480 | 0.708 to 0.784 | 1 to 20 | 8 to 20 | 1 |
| 16 | 20251103_sonar-foundation-agent_claude-sonnet-4-5 | 374/500 | 0.7480 | 0.708 to 0.784 | 1 to 20 | 7 to 20 | 1 |
| 17 | 20250720_Lingxi-v1.5_claude-4-sonnet-20250514 | 373/500 | 0.7460 | 0.706 to 0.782 | 1 to 20 | 8 to 20 | 1 |
| 18 | 20250915_JoyCode | 373/500 | 0.7460 | 0.706 to 0.782 | 1 to 20 | 6 to 20 | 1 |
| 19 | 20250603_Refact_Agent_claude-4-sonnet | 372/500 | 0.7440 | 0.704 to 0.780 | 1 to 20 | 8 to 20 | 1 |
| 20 | 20251015_Prometheus_v1.2.1_gpt5 | 372/500 | 0.7440 | 0.704 to 0.780 | 1 to 20 | 7 to 20 | 1 |

Rank intervals and tiers use exact McNemar tests on all 190 pairs with Holm correction at alpha = 0.05. 190 of those pairs are not distinguishable. The ranked systems fall into 1 tier(s).

### Adjacent ranks

19 of 19 adjacent pairs are statistically indistinguishable (Holm-corrected over the adjacent pairs).

| rank | system_upper | system_lower | diff | ci_low | ci_high | n_only_upper | n_only_lower | p_value | p_holm | distinguishable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20251205_sonar-foundation-agent_claude-opus-4-5 | 20251215_livesweagent_claude-opus-4-5 | +0.0000 | -0.0235 | +0.0235 | 18 | 18 | 1 | 1 | False |
| 2 | 20251215_livesweagent_claude-opus-4-5 | 20250928_trae_doubao_seed_code | +0.0040 | -0.0279 | +0.0359 | 34 | 32 | 0.9022 | 1 | False |
| 3 | 20250928_trae_doubao_seed_code | 20251127_openhands_claude-opus-4-5 | +0.0120 | -0.0203 | +0.0443 | 37 | 31 | 0.5446 | 1 | False |
| 4 | 20251127_openhands_claude-opus-4-5 | 20251120_livesweagent_gemini-3-pro-preview | +0.0020 | -0.0260 | +0.0300 | 26 | 25 | 1 | 1 | False |
| 5 | 20251120_livesweagent_gemini-3-pro-preview | 20250804_epam-ai-run-claude-4-sonnet | +0.0060 | -0.0241 | +0.0361 | 31 | 28 | 0.7948 | 1 | False |
| 6 | 20250804_epam-ai-run-claude-4-sonnet | 20250902_atlassian-rovo-dev | +0.0000 | -0.0294 | +0.0294 | 28 | 28 | 1 | 1 | False |
| 7 | 20250902_atlassian-rovo-dev | 20260217_mini-v2.0.0_claude-4-5-opus-high | +0.0000 | -0.0309 | +0.0309 | 31 | 31 | 1 | 1 | False |
| 8 | 20260217_mini-v2.0.0_claude-4-5-opus-high | 20250819_ACoder | +0.0040 | -0.0274 | +0.0354 | 33 | 31 | 0.9007 | 1 | False |
| 9 | 20250819_ACoder | 20260217_mini-v2.0.0_gemini-3-flash-high | +0.0060 | -0.0251 | +0.0371 | 33 | 30 | 0.8013 | 1 | False |
| 10 | 20260217_mini-v2.0.0_gemini-3-flash-high | 20260217_mini-v2.0.0_minimax-2-5-high | +0.0000 | -0.0294 | +0.0294 | 28 | 28 | 1 | 1 | False |
| 11 | 20260217_mini-v2.0.0_minimax-2-5-high | 20250901_warp | +0.0020 | -0.0320 | +0.0360 | 38 | 37 | 1 | 1 | False |
| 12 | 20250901_warp | 20260217_mini-v2.0.0_claude-4-6-opus | +0.0000 | -0.0319 | +0.0319 | 33 | 33 | 1 | 1 | False |
| 13 | 20260217_mini-v2.0.0_claude-4-6-opus | 20250612_trae | +0.0040 | -0.0284 | +0.0364 | 35 | 33 | 0.9036 | 1 | False |
| 14 | 20250612_trae | 20250731_harness_ai | +0.0040 | -0.0195 | +0.0275 | 19 | 17 | 0.8679 | 1 | False |
| 15 | 20250731_harness_ai | 20251103_sonar-foundation-agent_claude-sonnet-4-5 | +0.0000 | -0.0283 | +0.0283 | 26 | 26 | 1 | 1 | False |
| 16 | 20251103_sonar-foundation-agent_claude-sonnet-4-5 | 20250720_Lingxi-v1.5_claude-4-sonnet-20250514 | +0.0020 | -0.0276 | +0.0316 | 29 | 28 | 1 | 1 | False |
| 17 | 20250720_Lingxi-v1.5_claude-4-sonnet-20250514 | 20250915_JoyCode | +0.0000 | -0.0294 | +0.0294 | 28 | 28 | 1 | 1 | False |
| 18 | 20250915_JoyCode | 20250603_Refact_Agent_claude-4-sonnet | +0.0020 | -0.0281 | +0.0321 | 30 | 29 | 1 | 1 | False |
| 19 | 20250603_Refact_Agent_claude-4-sonnet | 20251015_Prometheus_v1.2.1_gpt5 | +0.0000 | -0.0338 | +0.0338 | 37 | 37 | 1 | 1 | False |

### Smallest difference this benchmark can resolve

For two systems on 500 shared tasks, at the disagreement levels seen among the ranked systems:

| disagreement level | share of tasks where the two differ | smallest significant difference | detectable with 80% power |
| --- | --- | --- | --- |
| p25 | 0.1120 | 0.0293 | 0.0418 |
| median | 0.1240 | 0.0309 | 0.0440 |
| p75 | 0.1355 | 0.0323 | 0.0460 |

## Suspect tasks

Tasks that weaker systems solve more often than stronger ones (most negative first, up to 50 of 7):

| task | n_solved | n | solve_rate | discrimination | disc_ci_low | disc_ci_high | irt_discrimination |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sympy__sympy-18763 | 56 | 173 | 0.324 | -0.245 | -0.380 | -0.099 | -0.292 |
| astropy__astropy-7606 | 8 | 173 | 0.046 | -0.180 | -0.321 | -0.032 | -0.307 |
| django__django-11790 | 37 | 173 | 0.214 | -0.065 | -0.212 | 0.085 | -0.034 |
| sphinx-doc__sphinx-10614 | 2 | 173 | 0.012 | -0.061 | -0.208 | 0.089 | -0.091 |
| matplotlib__matplotlib-23476 | 2 | 173 | 0.012 | -0.044 | -0.192 | 0.106 | -0.031 |
| matplotlib__matplotlib-23299 | 7 | 173 | 0.040 | -0.031 | -0.179 | 0.119 | -0.065 |
| django__django-13794 | 7 | 173 | 0.040 | -0.030 | -0.179 | 0.120 | -0.030 |

## Item response model (2PL)

- Converged: True after 181 iterations
- Tasks with a negative fitted slope: 7
- Effective number of informative tasks over all systems: 288.0
- Effective number of informative tasks over the ranked systems: 103.7
- Tasks holding 90% of the information over all systems: 286
- Tasks holding 90% of the information over the ranked systems: 169

### Test information

| ability | information |
| --- | --- |
| -3.0 | 50.56 |
| -2.5 | 65.98 |
| -2.0 | 82.38 |
| -1.5 | 98.22 |
| -1.0 | 110.98 |
| -0.5 | 118.83 |
| +0.0 | 117.82 |
| +0.5 | 102.86 |
| +1.0 | 80.20 |
| +1.5 | 61.94 |
| +2.0 | 48.67 |
| +2.5 | 36.63 |
| +3.0 | 26.48 |

## Systems with unusual response patterns

These systems solve hard tasks while failing easy ones more than the fitted model expects of a system with their score. That is a reason to look, not a finding about the system.

| system | accuracy | guttman_norm | guttman_expected | guttman_z | n_hard_solved | n_easy_failed |
| --- | --- | --- | --- | --- | --- | --- |
| 20260901_mini-v2.4.2_gemini-3-5-flash | 0.7180 | 0.1657 | 0.0528 | 11.90 | 17 | 2 |
| 20250110_learn_by_interact_claude3.5 | 0.6020 | 0.1530 | 0.0753 | 8.21 | 5 | 6 |
| 20251209_mini-v1.17.2_devstral-2512 | 0.5380 | 0.1558 | 0.0762 | 7.03 | 1 | 8 |
| 20250807_mini-v1.7.0_kimi-k2-instruct | 0.4380 | 0.1636 | 0.0888 | 6.17 | 0 | 12 |
| 20250627_agentless_MCTS-Refine-7B | 0.2320 | 0.2069 | 0.1203 | 5.51 | 6 | 23 |
| 20251211_mini-v1.17.2_gpt-5.2-2025-12-11 | 0.6900 | 0.1124 | 0.0564 | 5.24 | 6 | 6 |
| 20251209_mini-v1.17.2_devstral-small-2512 | 0.5640 | 0.1216 | 0.0762 | 4.16 | 2 | 5 |
| 20251201_mini-v1.17.1_deepseek-v3.2-reasoner | 0.6000 | 0.1116 | 0.0696 | 4.08 | 2 | 2 |
| 20260217_mini-v2.0.0_glm-5-high | 0.7280 | 0.0874 | 0.0509 | 3.78 | 8 | 2 |
