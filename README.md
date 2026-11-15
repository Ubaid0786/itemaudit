# itemaudit

A statistical and psychometric audit of an agent benchmark, computed from its
results table alone: which systems solved which tasks. It does not run any
task. It answers two questions. Which differences on the leaderboard are
larger than the noise? Which tasks look broken?

I wrote it because I build benchmark tasks and graders, and I wanted a way to
check a task set after the fact, using only the data every leaderboard already
has. I then ran it on the public SWE-bench Verified leaderboard. The results
are in this README and in `results/`.

## The problem

A leaderboard ranks systems by the share of tasks they solve. With a few
hundred tasks, a gap of one or two points between neighbours is often within
what you would see from a different draw of tasks, yet it is read as progress.
The right comparison is paired, because both systems ran the same tasks, and
when many pairs are compared the tests need a multiple-comparison correction.
A list of scores shows neither.

The second problem is the tasks. A task that every system solves, or none
does, cannot separate anyone. A task that weaker systems solve more often than
stronger ones points the wrong way: it usually means the test is wrong, the
task is ambiguous, the environment is unstable or the grading changed. Test
theory has had standard tools for finding such items for decades. They need
only the matrix of who solved what.

## What the tool does

Input: a CSV with one row per system and one column per task. Cells are
`1`/`0` (or `true`/`false`, `pass`/`fail`); an empty cell means the system was
not run on that task.

| function | what it returns |
| --- | --- |
| `load_results(path)` | the results matrix |
| `compare(a, b)` | paired difference in accuracy, analytic paired standard error and interval, paired bootstrap interval, exact McNemar p-value |
| `leaderboard(matrix)` | ranking, Holm-corrected tests of adjacent ranks, rank intervals from Holm-corrected tests of all pairs, bootstrap rank intervals, tiers |
| `min_detectable_difference(n, discordance)` | the smallest true difference a paired test on `n` tasks detects at a given power |
| `item_stats(matrix)` | per task: solve rate, corrected item-rest (point-biserial) discrimination with an interval, upper-lower index, flags for no-information and negative or near-zero discrimination |
| `fit_2pl(matrix)` | two-parameter logistic IRT fit (numpy/scipy, marginal likelihood by EM, normal priors on slope and intercept), test information, effective number of informative tasks |
| `system_fit(matrix)` | per system: Guttman errors (hard tasks solved while easier ones are failed), compared with what the fitted model expects |
| `dif(matrix, groups)` | Mantel-Haenszel test for tasks whose result depends on which of two groups a system belongs to, at equal overall ability |
| `report(matrix, out_dir)` | all of the above as Markdown and JSON |

Command line:

```
itemaudit report results.csv --out out/
itemaudit compare results.csv SYSTEM_A SYSTEM_B
itemaudit fetch swebench-verified
```

`examples/toy_results.csv` is simulated data (12 systems, 40 tasks, made by
`examples/make_toy.py`) with one reverse-keyed task planted in it:

```
$ itemaudit report examples/toy_results.csv --out examples --name toy_report --top 12
12 systems x 40 tasks
adjacent ranks indistinguishable: 11 of 11 (top 12)
tasks solved by all: 2, by none: 1, negative discrimination: 1
```

The planted task is the one it reports (`examples/toy_report.md`):

```
| task | n_solved | n | solve_rate | discrimination | disc_ci_low | disc_ci_high | irt_discrimination |
| task_07 | 6 | 12 | 0.500 | -0.857 | -0.959 | -0.558 | -1.085 |
```

On real data, rank 1 against rank 20 of SWE-bench Verified
(`results/example_compare.txt`):

```
$ itemaudit compare data/swebench_verified_matrix.csv 20251205_sonar-foundation-agent_claude-opus-4-5 20251015_Prometheus_v1.2.1_gpt5
20251205_sonar-foundation-agent_claude-opus-4-5: 79.20%  20251015_Prometheus_v1.2.1_gpt5: 74.40%  on 500 shared tasks
difference (a - b): +4.80 points, paired SE 1.61 (unpaired SE would be 2.67)
95% CI analytic [+1.64, +7.96], bootstrap [+1.60, +8.00]
solved by a only: 45, by b only: 21, both: 351, neither: 83
McNemar exact p = 0.004272
```

From Python:

```python
from itemaudit import load_results, compare, leaderboard, item_stats, fit_2pl, report

m = load_results("examples/toy_results.csv")
print(compare(m.row("system_i"), m.row("system_h"), names=("system_i", "system_h")).summary())
board = leaderboard(m)            # board.adjacent, board.rows, board.n_tiers
items = item_stats(m)             # DataFrame, one row per task
fit = fit_2pl(m)                  # fit.discrimination, fit.difficulty, fit.ability
report(m, "out")                  # writes out/report.md and out/report.json
```

## Findings on SWE-bench Verified

Everything in this section is copied from `results/swebench_verified_audit.md`
and `results/swebench_verified_audit.json`, which
`experiments/swebench_verified_audit.py` wrote on my machine (Python 3.14.8,
numpy 2.5.4, scipy 1.18.1, pandas 3.1.0). The full tables are in those files.

### Data

The source is the `evaluation/verified` folder of
[swe-bench/experiments](https://github.com/swe-bench/experiments) at commit
`40f164d5b8f1d249bf95a6df8b74b577fd8e519d` (3 September 2026). The task list is
`princeton-nlp/SWE-bench_Verified` on Hugging Face, revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.

- The folder has 182 submission directories. I used 173: 135 store their
  results in `results/results.json` and 38 in `per_instance_details.json`.
- I left out 9. Seven have no per-task results file at all. Two have a
  per-task file that contradicts the score in their own metadata: one file
  marks every task unresolved (0.0%) while the metadata says 69.6%, the other
  gives 10.2% against a reported 52.8%. I cannot tell which side is right, so
  both are out. The list is in the results file.
- Of the 135 submissions that say how many tasks they produced no patch for,
  32 have ten or more such tasks; the largest count is 121. The leaderboard
  scores those tasks as failures and I do the same, but a task that was never
  attempted says nothing about the task.
- Resolved rates run from 0.4% to 79.2%, median 56.6%.

### Which differences are real

Among the top 20 submissions (79.2% down to 74.4%):

- **19 of 19 adjacent pairs are statistically indistinguishable** at the 5%
  level (exact McNemar tests, Holm-corrected over the 19 pairs). The largest
  adjacent gap is 1.2 points.
- With Holm correction over all 190 pairs, none of the 190 is distinguishable.
  The smallest adjusted p-value is 0.3119. The top 20 form a single tier and
  every one of them has a rank interval of 1 to 20.
- Holm is conservative, so here is the uncorrected picture too: 29 of the 190
  pairs have p < 0.05. Rank 1 against rank 20 is a difference of 4.8 points,
  95% interval 1.6 to 8.0, p = 0.004272 on its own. It does not survive the
  correction for having looked at 190 pairs.
- Over the whole leaderboard of 173, all 172 adjacent pairs are
  indistinguishable and the submissions fall into 10 tiers. Going down from
  rank 1, the first submission that differs from it at an uncorrected
  p < 0.05 is rank 10; with Holm correction over the 172 comparisons it is
  rank 16.
- 15 of the top 20 name a Claude model in their model tag or directory name,
  4 are mini-SWE-agent runs, and none carries the maintainers'
  `checked: true` mark.

How small a difference can 500 tasks resolve? That depends on how often two
systems disagree on a task. Between pairs of top-20 submissions they disagree
on 5.6% to 17.8% of tasks, median 12.4%.

| disagreement between the pair | smallest difference that reaches p < 0.05 | detectable with 80% power |
| --- | --- | --- |
| 5.6% (lowest pair) | 2.07 points | 2.96 points |
| 11.2% (25th percentile) | 2.93 points | 4.18 points |
| 12.4% (median) | 3.09 points | 4.40 points |
| 13.6% (75th percentile) | 3.23 points | 4.60 points |
| 17.8% (highest pair) | 3.70 points | 5.27 points |

An unpaired comparison at the same accuracy would need 5.31 points to be
significant and 7.59 for 80% power. The entire top 20 spans 4.8 points.

### What the tasks look like

- No task is solved by all 173 submissions, which only reflects the weak
  2023 baselines in the data. 26 tasks are solved by none. 67 are solved by
  at most 5% of submissions and 10 by at least 95%. Solve-rate quartiles are
  0.266, 0.598 and 0.809.
- For the top 20 the picture is different: 254 tasks are solved by all 20,
  36 by none of them, and 210 by some. Only those 210 can change the order.
- The human difficulty label that ships with the dataset tracks the solve
  rate: mean 0.684 for "<15 min fix" (194 tasks), 0.483 for "15 min - 1 hour"
  (261), 0.195 for "1-4 hours" (42), 0.160 for ">4 hours" (3). Two tasks
  labelled "<15 min fix" are solved by nobody (see below).

### Tasks with negative or near-zero discrimination

I computed discrimination among "capable" submissions, those resolving at
least 50% (109 submissions: 13 from 2024, 84 from 2025, 12 from 2026),
because that is the population people compare.

- 30 tasks have negative discrimination: stronger submissions solve them less
  often. For 5 of them the 95% interval is entirely below zero.
- 60 more have discrimination between 0 and 0.1.
- 27 are solved by all capable submissions and 26 by none.

The counts move with the threshold (19 negative at 40%, 57 at 60%), so the
exact number means little. The five clearest cases:

| task | solved by capable | discrimination | 95% interval | solve rate, 2024 / 2025 / 2026 submissions | human label |
| --- | --- | --- | --- | --- | --- |
| sympy__sympy-23950 | 65/109 | -0.307 | -0.47 to -0.13 | 0.85 / 0.61 / 0.25 | 15 min - 1 hour |
| sympy__sympy-18763 | 22/109 | -0.259 | -0.43 to -0.07 | 0.46 / 0.19 / 0.00 | <15 min fix |
| sphinx-doc__sphinx-9711 | 97/109 | -0.259 | -0.43 to -0.07 | 0.92 / 0.88 / 0.92 | <15 min fix |
| sphinx-doc__sphinx-8595 | 88/109 | -0.248 | -0.42 to -0.06 | 0.92 / 0.77 / 0.92 | <15 min fix |
| pytest-dev__pytest-7205 | 69/109 | -0.193 | -0.37 to -0.00 | 0.85 / 0.60 / 0.67 | <15 min fix |

All 30 are listed in the results file. For several, the solve rate falls with
the year of submission, which is what produces the negative correlation with
overall accuracy.

Are these flags noise? Partly. In 200 random splits of the capable
submissions into halves, 38.3% of the tasks that are negative in one half are
also negative in the other, against 8.5% of all tasks. Splitting into two
disjoint groups instead (33 mini-SWE-agent runs, 76 others), 36.7% of the
tasks that are negative among the others are also negative among the
mini-SWE-agent runs, against 13.8% of all tasks. So a negative flag carries
signal, since it replicates well above the base rate, but most individual
flags do not replicate and should be treated as candidates, not verdicts.

### Tasks whose result depends on who ran them

This was not in my original plan. Looking at the flagged tasks, I saw some
whose solve rate differed sharply between the mini-SWE-agent runs and all
other submissions of similar strength, so I added a Mantel-Haenszel test
(`dif`). Among capable submissions, matched on resolved rate and
Holm-corrected over the 500 tasks, five tasks are flagged:

| task | solved by mini-SWE-agent runs | solved by other capable submissions | Holm p |
| --- | --- | --- | --- |
| pylint-dev__pylint-7277 | 3.0% | 81.6% | 2.3e-10 |
| pylint-dev__pylint-6528 | 3.0% | 60.5% | 2.8e-05 |
| django__django-11239 | 90.9% | 53.9% | 0.014 |
| matplotlib__matplotlib-20488 | 78.8% | 39.5% | 0.026 |
| pydata__xarray-4695 | 100.0% | 65.8% | 0.037 |

A task that 3.0% of one group passes and 81.6% of the other group passes, at
matched overall accuracy, is not measuring the system. The two groups differ in
scaffold, in who ran the evaluation and in when, and I cannot separate those
from the data.

### Cross-check against public lists

I looked for published lists of problematic SWE-bench Verified tasks and
found three I could open and verify (sources in
`experiments/known_issues.yaml`).

1. **OpenAI's 23 tasks that "could not run on our infrastructure"** (footnote
   to the GPT-4.1 announcement, April 2025).
   - Of the 5 group-dependent tasks above, 3 are on this list
     (pylint-7277, pylint-6528, matplotlib-20488). Chance would give 0.23;
     hypergeometric p = 0.0008. This is the one clear agreement.
   - Of my 30 negative-discrimination tasks, 3 are on the list. Chance would
     give 1.38; p = 0.15. That is not a significant overlap.
   - Of the 90 negative or near-zero tasks, 5 are on the list (4.14 by
     chance, p = 0.4). Of the 26 never-solved tasks, 1 is (1.20 by chance).
2. **The three flawed-test examples in OpenAI's February 2026 audit.** That
   audit reviewed 138 tasks and reports material problems in 59.4% of them,
   but publishes only examples, not the list. Of the three, one
   (pylint-dev__pylint-4551) is solved by none of the 173 submissions, one
   (sympy__sympy-18199) by exactly 1, and one (django__django-14725) by 16.
   None of the three has negative discrimination. Tasks that nobody solves
   have no discrimination at all, so that statistic cannot see this kind of
   defect; the "solved by none" list can.
3. **UTBoost's tasks with insufficient tests** (26 of its 36 are in
   Verified). 1 is among my 30 negative tasks (1.56 by chance) and 6 among
   the 90 negative or near-zero ones (4.68 by chance). No enrichment. That is
   expected: a test that is too weak makes a task easier, it does not make
   strong systems fail it.

The honest summary: discrimination statistics alone agree only weakly with
what others have published. The group-dependence test agrees strongly with
the one list that is about evaluation environments.

### My reading of some flagged tasks

I read the problem statement, reference patch and tests of nine tasks in the
public dataset. I did not run them and I did not read any submission's patch,
so the explanations are hypotheses. The full notes are in
`experiments/task_notes.yaml` and at the end of the results file. In short:

- `pylint-dev__pylint-7277` and `pylint-dev__pylint-6528`: the failing tests
  are in the same test class and build expectations from the working
  directory and `sys.path`. They look environment-sensitive.
- `matplotlib__matplotlib-20488`: the issue text itself says the test fails
  in CI and cannot be reproduced locally because of a numpy change.
- `psf__requests-2317`: seven of the eight tests that must go from failing to
  passing are older tests that make live requests to httpbin.org.
- `sympy__sympy-18763`: the tests require parentheses in a case the issue
  does not ask about; a fix that adds them only where needed would fail.
- `pytest-dev__pytest-7205`: nine of the ten required tests check a quoting
  change for strings that follows from the reference patch's choice of
  `saferepr`, which appears only in the maintainers' hints. The issue is
  about bytes.
- `django__django-10999`: labelled "<15 min fix", solved by nobody. The tests
  encode behaviour that differs from the fix the issue proposes; I ran the
  proposed regular expression and it does not satisfy them.
- `matplotlib__matplotlib-25479`: labelled "<15 min fix", solved by nobody.
  One required test inverts an existing equality assertion that the issue
  does not mention.
- `pylint-dev__pylint-4551`: OpenAI's published example; nobody solves it.

### Effective number of informative tasks

A 2PL model fitted to all 173 submissions converges in 181 iterations. Seven
tasks get a negative slope. Fitted ability and resolved rate have a Spearman
correlation of 0.9975.

- Over all submissions the effective number of informative tasks is 288.0 of
  500, and 286 tasks hold 90% of the information.
- At the abilities of the top 20 it is 103.7, 169 tasks hold 90% of the
  information and 36 tasks hold half of it.
- Test information peaks at ability -0.30 (119.9). The top 20 sit between
  +1.87 and +3.15, where it is 42.8 at their median, a ratio of 2.80. The
  benchmark carries far more information about mid-table systems than about
  the leaders.

### Does the order change if non-discriminating tasks are removed?

I flagged tasks as non-discriminating (negative, below 0.1, or solved by all
or none) using the 89 capable submissions outside the top 20, so the flags do
not depend on the submissions being re-ranked. That removes 174 tasks and
leaves 326.

- 16 of the top 20 change rank; the largest move is 7 places. Rank 1 stays
  where it is. Kendall's tau-b between the two orders is 0.765.
- Removing 174 random tasks instead gives a mean tau-b of 0.732 (5th to 95th
  percentile 0.597 to 0.848 over 1000 draws). The reordering from removing
  the flagged tasks is no larger than the reordering from removing random
  ones.
- On the 326 remaining tasks, 19 of 19 adjacent pairs are still
  indistinguishable.

Flagging with all 109 capable submissions removes 143 tasks and gives the
same picture (tau-b 0.796; random removal 0.765, range 0.648 to 0.868).

I expected removing bad tasks to tidy the ranking. It does not. The order
among the top 20 moves about as much under any change of task set, because
the differences between them are within noise to begin with.

### Submissions with unusual response patterns

Nine submissions solve hard tasks while failing easier ones more than the
fitted model expects (more than three standard deviations). Seven of the nine
are mini-SWE-agent runs, which fits the group-dependent tasks above. The most
extreme one has no patch for 59 tasks, which are scored as failures. These
are statistical anomalies with ordinary explanations available; they say
nothing about how any submission was produced. The model also fits
imperfectly: the z-scores have a standard deviation of 2.18 where a
well-fitting model would give about 1.

### What surprised me or went against what I expected

- I expected the negative-discrimination flags to line up with published
  lists of bad tasks. The overlap is small and not significant.
- The most useful signal was group dependence, which I added after seeing the
  data.
- Removing flagged tasks does not stabilise the ranking.
- Two submissions' per-task files contradict their own reported scores
  (0.0% against 69.6%, and 10.2% against 52.8%).

## How to reproduce

Requirements: Python 3.10 or newer, and the GitHub CLI (`gh`) signed in, or a
`GITHUB_TOKEN` in the environment. 358 small files are downloaded through
the GitHub API (about a minute and a half on my connection), plus the task
list from the Hugging Face datasets server. The repository is not cloned.

```
python -m venv .venv
.venv/bin/pip install -e ".[test]"        # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest

.venv/bin/python experiments/build_swebench_matrix.py
.venv/bin/python experiments/swebench_verified_audit.py
```

`build_swebench_matrix.py` is pinned to the commit above (override with
`--ref`), caches the downloaded files under `data/raw/`, checks each against
its Git blob hash, and writes `data/swebench_verified_matrix.csv`,
`data/swebench_verified_submissions.csv` and
`data/swebench_verified_build.json`. Rerunning uses the cache.
`swebench_verified_audit.py` uses fixed seeds; running it twice here gave
byte-identical output files.

Files in `results/`:

| file | content |
| --- | --- |
| `swebench_verified_audit.md`, `.json` | the audit: every number quoted above |
| `swebench_verified_item_stats.csv` | one row per task: solve rates, discrimination, IRT parameters, group-dependence test, list membership |
| `swebench_verified_top20.csv` | the ranked top 20 with intervals |
| `swebench_verified_report.md`, `.json` | the generic `itemaudit report` output on the same matrix |
| `example_compare.txt` | the `itemaudit compare` output shown above |

### Data licence

The `swe-bench/experiments` repository has no licence file, and GitHub
reports no licence for it. Its README asks that people citing leaderboard
numbers cite SWE-bench. The dataset card of `princeton-nlp/SWE-bench_Verified`
on Hugging Face has no licence field either. (The SWE-bench code repository
is MIT, but that covers the code.) Since redistribution of the data is not
clearly allowed, this repository does not contain the downloaded files, the
results matrix or the per-submission table; `.gitignore` excludes them. It
contains the build script, the provenance record
(`data/swebench_verified_build.json`) and the derived statistics in
`results/`. The code here is MIT.

## Limitations and confounds

- **Submissions are not independent.** 15 of the top 20 name a Claude model,
  and many submissions share a base model or a scaffold. The paired tests
  treat tasks as the random unit and are valid for each pair. The intervals
  on task discrimination and the group-dependence test treat submissions as
  independent draws, which they are not, so those are too optimistic.
- **One run per submission.** A leaderboard entry is one run. The tests here
  account for which tasks happen to be in the benchmark, not for run-to-run
  variation of the same system, which would add to the noise.
- **Contamination.** The tasks are public and old. Newer models may have
  seen them and their fixes; OpenAI and several papers report evidence of
  this. A task can discriminate well and still be measuring memory.
- **Time, scaffold and harness are tangled.** Later submissions use newer
  models, different scaffolds and possibly a different version of the
  evaluation harness. When a task's solve rate falls over time I cannot say
  from the results alone which of these changed.
- **Formats differ across years.** Older submissions list resolved ids,
  newer ones give a per-instance record, some have both, some neither. A
  task with no patch is scored as a failure.
- **Unverified entries.** None of the top 20 carries the maintainers'
  `checked` mark. I take the results files at face value.
- **Thresholds are choices.** "Capable" at 50%, "near zero" below 0.1, top
  20. I report the sensitivity to the first; the others I did not vary.
- **The top-20 cut falls inside a tie.** Ranks 19, 20 and 21 all resolve 372
  of 500 tasks, so which two of the three make the cut is arbitrary. I
  recomputed the pairwise tests with the left-out submission swapped in: 30 of
  190 pairs have uncorrected p < 0.05 (not 29), the smallest Holm-adjusted p
  is 0.1018 (not 0.3119), 256 tasks are solved by all 20 (not 254) and 37 by
  none (not 36). No pair becomes distinguishable after correction, so the
  conclusion holds, but the individual figures move with the tie-break.
- **Discrimination depends on the population.** A task is flagged relative to
  the submissions in the matrix. With a different mix it could look fine.
- **Holm is conservative.** I report uncorrected counts next to the
  corrected ones.
- **The IRT model is simple.** One dimension, a normal ability distribution,
  no standard errors on item parameters. The Guttman z-scores show it does
  not fit these data closely.
- **The task notes are my reading**, of nine tasks, without running anything.
- **Rank intervals** from pairwise tests are a simple construction, not the
  optimal confidence sets from the ranking-inference literature.

## Related work

I verified each item below during this work: papers through the arXiv API
(title, authors, abstract; I read abstracts, not full texts, unless noted),
OpenAI's posts through Internet Archive snapshots, repositories through the
GitHub API.

Statistics for evaluations:

- Evan Miller, "Adding Error Bars to Evals: A Statistical Approach to
  Language Model Evaluations", arXiv:2411.00640 (2024). Paired differences,
  standard errors and power analysis for evals. The paired standard error
  and the detectable-difference calculation here follow that approach.
- Chetverikov, Mogstad, Morgen, Romano, Shaikh and Wilhelm, "csranks: An R
  Package for Estimation and Inference Involving Ranks", arXiv:2401.15205
  (2024). Proper confidence sets for ranks; my rank intervals are cruder.

Item response theory on model benchmarks:

- Vania et al., "Comparing Test Sets with Item Response Theory",
  arXiv:2106.00840 (2021).
- Polo et al., "tinyBenchmarks: evaluating LLMs with fewer examples",
  arXiv:2402.14992 (2024).
- Kipnis et al., "metabench: A Sparse Benchmark of Reasoning and Knowledge in
  Large Language Models", arXiv:2407.12844 (2024).
- Zhou et al., "Lost in Benchmarks? Rethinking Large Language Model
  Benchmarking with Item Response Theory", arXiv:2505.15055 (2025).
- Hofmann et al., "Fluid Language Model Benchmarking", arXiv:2509.11106
  (2025).
- Lalor and Rodriguez, "py-irt: A Scalable Item Response Theory Library for
  Python", arXiv:2203.01282 (2022).
- Ge et al., "Agent psychometrics: Task-level performance prediction in
  agentic coding benchmarks", arXiv:2604.00594 (2026). The closest paper I
  found: IRT on agentic coding leaderboards including SWE-bench Verified,
  with ability split into model and scaffold. From the HTML version (I read
  the first part), it uses a one-parameter model, so no discrimination, and
  its aim is predicting task outcomes, not auditing tasks or testing rank
  differences.
- Ndzomga, "Efficient Benchmarking of AI Agents", arXiv:2603.23749 (2026).
  Keeps tasks with mid-range historical pass rates to rank agents cheaply.
- Zheng and Yang, "Are Near-Tied LLM Rankings Robust to Family-DIF-Guided
  Benchmark Recomposition?", arXiv:2609.00482 (2026). Differential item
  functioning across model families and a matched-random baseline for
  near-tie reversals, on five benchmarks; close in spirit to the last two
  analyses here.

Quality of SWE-bench Verified:

- Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub
  Issues?", arXiv:2310.06770 (2023).
- OpenAI, "Why SWE-bench Verified no longer measures frontier coding
  capabilities" (23 February 2026). Manual audit of 138 tasks, 59.4% with
  material issues in tests or description, plus contamination evidence.
- OpenAI, "Introducing GPT-4.1 in the API" (April 2025), footnote listing 23
  Verified tasks that could not run on their infrastructure.
- Aleithan et al., "SWE-Bench+: Enhanced Coding Benchmark for LLMs",
  arXiv:2410.06992 (2024). Solution leakage and weak tests.
- Yu, Zhu, He and Kang, "UTBoost: Rigorous Evaluation of Coding Agents on
  SWE-Bench", arXiv:2506.09289 (2025). Augmented tests; 36 instances with
  insufficient tests.
- Wang, Pradel and Liu, "Are 'Solved Issues' in SWE-bench Really Solved
  Correctly? An Empirical Study", arXiv:2503.15223 (2025).
- Liang, Garg and Moghaddam, "The SWE-Bench Illusion: When State-of-the-Art
  LLMs Remember Instead of Reason", arXiv:2506.12286 (2025).
- Prathifkumar, Mathews and Nagappan, "Does SWE-Bench-Verified Test Agent
  Ability or Model Memory?", arXiv:2512.10218 (2025).

Existing tools (found with the GitHub search API):

- `nd-ball/py-irt`: Bayesian IRT models in Python.
- `felipemaiapolo/tinyBenchmarks`: the tinyBenchmarks code.
- `zhux0445/llm-eval-psychometrics`: IRT and classical diagnostics for LLM
  benchmarks and judges, on PyPI.
- `vipul21435/irtbench`: Rasch and 2PL fits for models-by-items results,
  including backwards items and subset selection.
- `Peter-A-P/model-selection-tenth-cost`: IRT-based adaptive testing for
  choosing between models.

What is and is not new here. None of the methods is new: paired tests,
Holm, point-biserial discrimination, Guttman errors, 2PL and Mantel-Haenszel
are textbook, and other packages already fit IRT models to benchmark results.
What this repository adds is the combination in one small tool that works
from a results CSV, including the leaderboard-level tests and the
group-dependence test next to the item statistics, and the audit of the
SWE-bench Verified leaderboard at a pinned commit with a cross-check against
published task lists. I did not find that audit elsewhere, which is not proof
that nobody has done it.

## Who would use or pay for this

I have no users and no revenue. These are the people I think the tool is
for, and I have not tested that belief.

- **Benchmark maintainers.** Before publishing a leaderboard: which ranks
  can be told apart, which tasks carry no information, which tasks behave
  differently depending on who ran them. On SWE-bench Verified the last
  check pointed at tasks that a lab had separately found it could not run.
- **Vendors of evaluation data.** A vendor selling tasks to a lab needs to
  show that the tasks separate strong systems from weak ones and that none
  is broken. Item statistics from a pilot run on a few models are a concrete
  deliverable for that, and a cheap one compared with human review of every
  task. They do not replace review: the worst tasks here were ones nobody
  solves, which only reading the task explains.
- **Teams choosing between models.** "Model A scores 2 points higher on a
  500-task benchmark" is, on the evidence above, usually not a reason to
  switch. `itemaudit compare` gives the interval for a specific pair.

Whether anyone would pay is an open question. The statistics are standard
and free to reimplement. If there is a product, I think it is the audit as a
service on private results, or the checks built into a benchmark's
submission pipeline, not the library.

## Licence

MIT, see `LICENSE`. Copyright 2026 Ubaid Mushtaq.
