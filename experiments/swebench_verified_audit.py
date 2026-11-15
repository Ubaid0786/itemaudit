"""Audit the SWE-bench Verified leaderboard with itemaudit.

Reads the matrix built by build_swebench_matrix.py and writes to results/:

    swebench_verified_audit.json        every number, machine readable
    swebench_verified_audit.md          the same as a readable report
    swebench_verified_item_stats.csv    one row per task
    swebench_verified_top20.csv         the ranked top 20
    swebench_verified_report.{md,json}  the generic `itemaudit report` output

Everything is deterministic (fixed seeds), so rerunning on the same matrix
reproduces the files byte for byte.

    python experiments/swebench_verified_audit.py
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy import stats

from itemaudit import (
    ResultsMatrix,
    compare,
    dif,
    fit_2pl,
    holm,
    item_stats,
    kr20,
    leaderboard,
    load_results,
    min_detectable_difference,
    min_significant_difference,
    report,
    system_fit,
)
from itemaudit.items import FLAG_ALL, FLAG_NEAR_ZERO, FLAG_NEGATIVE, FLAG_NONE
from itemaudit.report import markdown_table, solve_rate_histogram, to_jsonable

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

TOP = 20
ALPHA = 0.05
# "Capable" systems resolve at least half the tasks. Discrimination among
# them asks whether a task separates the systems people actually compare.
CAPABLE = 0.5
CAPABLE_SENSITIVITY = (0.4, 0.5, 0.6)
NEAR_ZERO = 0.1
SEED = 0
N_SPLITS = 200
N_RANDOM_REMOVALS = 1000
MINI = "mini-SWE-agent"


def flag_sets(items: pd.DataFrame) -> dict[str, set[str]]:
    flag = items["flag"]
    return {
        "negative": set(items.index[flag == FLAG_NEGATIVE]),
        "near_zero": set(items.index[flag == FLAG_NEAR_ZERO]),
        "solved_by_all": set(items.index[flag == FLAG_ALL]),
        "solved_by_none": set(items.index[flag == FLAG_NONE]),
    }


def flag_counts(items: pd.DataFrame) -> dict[str, int]:
    sets = flag_sets(items)
    counts = {name: len(tasks) for name, tasks in sets.items()}
    counts["negative_ci_excludes_zero"] = int(
        ((items["flag"] == FLAG_NEGATIVE) & (items["disc_ci_high"] < 0)).sum()
    )
    counts["unflagged"] = int((items["flag"] == "").sum())
    return counts


def ordering_change(matrix: ResultsMatrix, top_systems: list[str], removed: set[str]) -> dict[str, Any]:
    """Re-rank ``top_systems`` after dropping ``removed`` tasks and compare the two orders."""
    before = matrix.select_systems(top_systems)
    after = before.drop_tasks(removed)
    acc_before, acc_after = before.accuracy(), after.accuracy()
    rank_before = stats.rankdata(-acc_before, method="min").astype(int)
    rank_after = stats.rankdata(-acc_after, method="min").astype(int)
    shift = rank_after - rank_before
    board = leaderboard(after, alpha=ALPHA, seed=SEED)
    return {
        "n_tasks_removed": len(removed),
        "n_tasks_remaining": after.n_tasks,
        "kendall_tau_b": float(stats.kendalltau(acc_before, acc_after).statistic),
        "spearman": float(stats.spearmanr(acc_before, acc_after).statistic),
        "n_systems_whose_rank_changed": int((shift != 0).sum()),
        "max_rank_shift": int(np.abs(shift).max()),
        "top_system_before": top_systems[int(np.argmin(rank_before))],
        "systems_ranked_first_after": [s for s, r in zip(top_systems, rank_after) if r == 1],
        "n_adjacent_indistinguishable_after": board.n_adjacent_indistinguishable,
        "n_pairs_indistinguishable_after": board.n_pairs_indistinguishable,
        "n_pairs": board.n_pairs,
        "table": [
            {
                "system": s,
                "resolved_before": float(a),
                "rank_before": int(rb),
                "resolved_after": float(b),
                "rank_after": int(ra),
            }
            for s, a, rb, b, ra in zip(top_systems, acc_before, rank_before, acc_after, rank_after)
        ],
    }


def random_removal_baseline(
    matrix: ResultsMatrix, top_systems: list[str], n_removed: int, rng: np.random.Generator
) -> dict[str, float]:
    """Kendall tau between the original order and the order after removing random tasks."""
    before = matrix.select_systems(top_systems)
    acc_before = before.accuracy()
    taus = np.empty(N_RANDOM_REMOVALS)
    for i in range(N_RANDOM_REMOVALS):
        keep = np.sort(rng.choice(before.n_tasks, before.n_tasks - n_removed, replace=False))
        taus[i] = stats.kendalltau(acc_before, before.values[:, keep].mean(axis=1)).statistic
    return {
        "n_draws": N_RANDOM_REMOVALS,
        "kendall_tau_b_mean": float(taus.mean()),
        "kendall_tau_b_p05": float(np.quantile(taus, 0.05)),
        "kendall_tau_b_p95": float(np.quantile(taus, 0.95)),
    }


def split_half_replication(matrix: ResultsMatrix, rng: np.random.Generator) -> dict[str, float]:
    """How often a task with negative discrimination in one random half is negative in the other."""
    replicated, base, n_flagged = [], [], []
    systems = np.array(matrix.systems)
    for _ in range(N_SPLITS):
        order = rng.permutation(len(systems))
        half = len(systems) // 2
        a = item_stats(matrix.select_systems(list(systems[order[:half]])))["discrimination"]
        b = item_stats(matrix.select_systems(list(systems[order[half:]])))["discrimination"]
        both = a.notna() & b.notna()
        flagged = both & (a < 0)
        n_flagged.append(int(flagged.sum()))
        if flagged.any():
            replicated.append(float((b[flagged] < 0).mean()))
        base.append(float((b[both] < 0).mean()))
    return {
        "n_splits": N_SPLITS,
        "mean_n_negative_in_first_half": float(np.mean(n_flagged)),
        "mean_share_also_negative_in_second_half": float(np.mean(replicated)),
        "mean_share_negative_in_second_half_overall": float(np.mean(base)),
    }


def optional_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def overlap(flagged: set[str], listed: set[str], n_tasks: int) -> dict[str, Any]:
    k = len(flagged & listed)
    return {
        "n_flagged": len(flagged),
        "n_listed": len(listed),
        "overlap": k,
        "overlap_tasks": sorted(flagged & listed),
        "expected_by_chance": len(flagged) * len(listed) / n_tasks,
        # P(overlap >= k) if the flagged set were drawn at random from all tasks.
        "p_hypergeometric": float(stats.hypergeom.sf(k - 1, n_tasks, len(listed), len(flagged))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    matrix = load_results(args.data_dir / "swebench_verified_matrix.csv")
    submissions = pd.read_csv(args.data_dir / "swebench_verified_submissions.csv", index_col=0)
    provenance = json.loads((args.data_dir / "swebench_verified_build.json").read_text(encoding="utf-8"))
    meta = submissions.loc[list(matrix.systems)]
    index_path = args.data_dir / "raw" / "swebench_verified_instances.json"
    task_index = pd.DataFrame(json.loads(index_path.read_text(encoding="utf-8"))["instances"]).set_index(
        "instance_id"
    )
    known = yaml.safe_load((HERE / "known_issues.yaml").read_text(encoding="utf-8"))
    notes = yaml.safe_load((HERE / "task_notes.yaml").read_text(encoding="utf-8"))

    rng = np.random.default_rng(SEED)
    accuracy = pd.Series(matrix.accuracy(), index=matrix.systems)
    n_tasks = matrix.n_tasks
    R: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "settings": {
            "top": TOP,
            "alpha": ALPHA,
            "capable_min_resolved_rate": CAPABLE,
            "near_zero_discrimination_below": NEAR_ZERO,
            "seed": SEED,
        },
        "data": provenance,
    }

    # ---- 1. tasks over all submissions ---------------------------------
    items_all = item_stats(matrix, near_zero=NEAR_ZERO)
    R["tasks_all_submissions"] = {
        "n_submissions": matrix.n_systems,
        "n_tasks": n_tasks,
        "kr20": kr20(matrix),
        "resolved_rate_min": float(accuracy.min()),
        "resolved_rate_median": float(accuracy.median()),
        "resolved_rate_max": float(accuracy.max()),
        "flags": flag_counts(items_all),
        "solve_rate_histogram": solve_rate_histogram(items_all["solve_rate"].to_numpy()),
        "solve_rate_quartiles": [float(q) for q in items_all["solve_rate"].quantile([0.25, 0.5, 0.75])],
        "n_tasks_solved_by_at_most_5pct": int((items_all["solve_rate"] <= 0.05).sum()),
        "n_tasks_solved_by_at_least_95pct": int((items_all["solve_rate"] >= 0.95).sum()),
    }
    difficulty = items_all.join(task_index)
    R["annotated_difficulty"] = [
        {
            "label": label,
            "n_tasks": int(len(group)),
            "mean_solve_rate": float(group["solve_rate"].mean()),
            "n_solved_by_none": int((group["n_solved"] == 0).sum()),
        }
        for label, group in sorted(difficulty.groupby("difficulty"), key=lambda kv: -kv[1]["solve_rate"].mean())
    ]

    # ---- 2. top of the leaderboard --------------------------------------
    board = leaderboard(matrix, top=TOP, alpha=ALPHA, seed=SEED)
    top_systems = list(board.systems)
    top_matrix = matrix.select_systems(top_systems)
    upper = np.triu_indices(TOP, k=1)
    raw_p = board.p_values[upper]
    top_solved = top_matrix.values.sum(axis=0)
    first_vs_last = compare(
        top_matrix.values[0], top_matrix.values[-1], names=(top_systems[0], top_systems[-1]), seed=SEED
    )
    text = (meta.loc[top_systems, "model"].fillna("") + " " + pd.Series(top_systems, index=top_systems)).str.lower()
    R["top"] = {
        "leaderboard": board.to_dict(),
        "n_pairs_raw_p_below_alpha": int((raw_p < ALPHA).sum()),
        "smallest_raw_p": float(raw_p.min()),
        "largest_adjacent_gap": float(max(pair.comparison.diff for pair in board.adjacent)),
        "smallest_holm_p": float(np.nanmin(board.p_holm[upper])),
        "first_vs_last": first_vs_last.to_dict(),
        "n_tasks_solved_by_all_top": int((top_solved == TOP).sum()),
        "n_tasks_solved_by_none_of_top": int((top_solved == 0).sum()),
        "n_tasks_where_top_systems_disagree": int(((top_solved > 0) & (top_solved < TOP)).sum()),
        "n_outside_top_tied_with_last": int(
            (accuracy.drop(index=top_systems) == accuracy[top_systems[-1]]).sum()
        ),
        "n_top_from_mini_swe_agent": int((meta.loc[top_systems, "agent"] == MINI).sum()),
        "n_top_naming_a_claude_model": int(text.str.contains("claude|opus|sonnet").sum()),
        "n_top_checked_by_maintainers": int(
            meta.loc[top_systems, "checked"].astype(str).str.lower().eq("true").sum()
        ),
    }

    full = leaderboard(matrix, alpha=ALPHA, seed=SEED)
    against_first = full.p_values[0, 1:]
    against_first_holm = holm(against_first)
    R["full_leaderboard"] = {
        "n_systems": matrix.n_systems,
        "n_adjacent_pairs": len(full.adjacent),
        "n_adjacent_indistinguishable": full.n_adjacent_indistinguishable,
        "n_tiers": full.n_tiers,
        "first_rank_with_raw_p_below_alpha_vs_rank_1": int(2 + np.argmax(against_first < ALPHA)),
        "first_rank_with_holm_p_below_alpha_vs_rank_1": int(2 + np.argmax(against_first_holm < ALPHA)),
    }

    # ---- 3. what 500 tasks can resolve ----------------------------------
    discordance = board.discordance[upper]
    levels = {
        "top20_pairs_min": float(discordance.min()),
        "top20_pairs_p25": float(np.quantile(discordance, 0.25)),
        "top20_pairs_median": float(np.median(discordance)),
        "top20_pairs_p75": float(np.quantile(discordance, 0.75)),
        "top20_pairs_max": float(discordance.max()),
    }
    reference_accuracy = float(accuracy[top_systems].median())
    z = stats.norm.ppf(1 - ALPHA / 2)
    unpaired_se = float(np.sqrt(2 * reference_accuracy * (1 - reference_accuracy) / n_tasks))
    R["detectable_difference"] = {
        "n_tasks": n_tasks,
        "levels": {
            name: {
                "discordance": level,
                "min_significant_difference": min_significant_difference(n_tasks, level, ALPHA),
                "min_detectable_difference_power_0.8": min_detectable_difference(n_tasks, level, ALPHA, 0.8),
            }
            for name, level in levels.items()
        },
        "unpaired_at_median_top20_accuracy": {
            "accuracy": reference_accuracy,
            "min_significant_difference": float(z * unpaired_se),
            "min_detectable_difference_power_0.8": float((z + stats.norm.ppf(0.8)) * unpaired_se),
        },
        "top20_spread_first_minus_last": float(accuracy[top_systems[0]] - accuracy[top_systems[-1]]),
    }

    # ---- 4. discrimination among capable systems ------------------------
    def capable_matrix(threshold: float) -> ResultsMatrix:
        return matrix.select_systems([s for s in matrix.systems if accuracy[s] >= threshold])

    capable = capable_matrix(CAPABLE)
    items_cap = item_stats(capable, near_zero=NEAR_ZERO)
    sets_cap = flag_sets(items_cap)
    fit = fit_2pl(matrix)
    irt_items = fit.items_frame()
    period = meta.loc[list(capable.systems), "date"].astype(str).str[:4]
    capable_frame = capable.to_frame()
    solve_by_year = capable_frame.groupby(period.to_numpy()).mean().T

    negative = items_cap[items_cap["flag"] == FLAG_NEGATIVE].sort_values("discrimination")
    negative_rows = []
    for task, row in negative.iterrows():
        entry = {
            "task": task,
            "n_solved_capable": int(row["n_solved"]),
            "n_capable": int(row["n"]),
            "solve_rate_capable": float(row["solve_rate"]),
            "discrimination_capable": float(row["discrimination"]),
            "disc_ci_low": float(row["disc_ci_low"]),
            "disc_ci_high": float(row["disc_ci_high"]),
            "solve_rate_all": float(items_all.loc[task, "solve_rate"]),
            "discrimination_all": float(items_all.loc[task, "discrimination"]),
            "irt_discrimination": float(irt_items.loc[task, "irt_discrimination"]),
            "annotated_difficulty": task_index.loc[task, "difficulty"],
        }
        for year in solve_by_year.columns:
            entry[f"solve_rate_capable_{year}"] = float(solve_by_year.loc[task, year])
        negative_rows.append(entry)

    groups = ["mini" if a == MINI else "other" for a in meta.loc[list(capable.systems), "agent"]]
    is_mini = np.array(groups) == "mini"
    mini_cap = capable.select_systems([s for s, m in zip(capable.systems, is_mini) if m])
    other_cap = capable.select_systems([s for s, m in zip(capable.systems, is_mini) if not m])
    disc_mini = item_stats(mini_cap)["discrimination"]
    disc_other = item_stats(other_cap)["discrimination"]
    both = disc_mini.notna() & disc_other.notna()
    negative_other = both & (disc_other < 0)
    R["capable"] = {
        "threshold": CAPABLE,
        "n_systems": capable.n_systems,
        "n_per_year": {str(k): int(v) for k, v in period.value_counts().sort_index().items()},
        "kr20": kr20(capable),
        "flags": flag_counts(items_cap),
        "sensitivity_to_threshold": {
            str(t): {"n_systems": capable_matrix(t).n_systems, **flag_counts(item_stats(capable_matrix(t)))}
            for t in CAPABLE_SENSITIVITY
        },
        "negative_tasks": negative_rows,
        "split_half_replication": split_half_replication(capable, rng),
        "scaffold_split": {
            "n_mini_swe_agent": int(is_mini.sum()),
            "n_other": int((~is_mini).sum()),
            "n_tasks_with_discrimination_in_both": int(both.sum()),
            "n_negative_among_other": int(negative_other.sum()),
            "share_of_those_also_negative_among_mini": float((disc_mini[negative_other] < 0).mean()),
            "share_negative_among_mini_overall": float((disc_mini[both] < 0).mean()),
            "discrimination_correlation": float(disc_mini[both].corr(disc_other[both])),
        },
    }

    # ---- 5. tasks that behave differently in the two groups of submissions
    dif_frame = dif(capable, groups, focal="mini", alpha=ALPHA)
    dif_flagged = dif_frame[dif_frame["flag"] == "dif"].sort_values("p_value")
    R["dif_mini_vs_other"] = {
        "focal_group": f"{MINI} runs",
        "n_focal": int(is_mini.sum()),
        "n_reference": int((~is_mini).sum()),
        "n_flagged": int(len(dif_flagged)),
        "tasks": [
            {
                "task": task,
                "rate_mini": float(row["rate_focal"]),
                "rate_other": float(row["rate_reference"]),
                "odds_ratio": float(row["odds_ratio"]),
                "chi2": float(row["chi2"]),
                "p_value": float(row["p_value"]),
                "p_holm": float(row["p_holm"]),
            }
            for task, row in dif_flagged.iterrows()
        ],
    }
    dif_set = set(dif_flagged.index)

    # ---- 6. cross-check against public lists ----------------------------
    task_set = set(matrix.tasks)
    flagged_sets = {
        "negative_discrimination_capable": sets_cap["negative"],
        "negative_or_near_zero_capable": sets_cap["negative"] | sets_cap["near_zero"],
        "solved_by_no_submission": flag_sets(items_all)["solved_by_none"],
        "group_dependent_dif": dif_set,
        "any_of_the_above": sets_cap["negative"]
        | sets_cap["near_zero"]
        | flag_sets(items_all)["solved_by_none"]
        | dif_set,
    }
    cross: dict[str, Any] = {}
    for name, entry in known.items():
        listed = set(entry["tasks"]) & task_set
        rows = items_cap.loc[sorted(listed)]
        cross[name] = {
            "description": " ".join(entry["description"].split()),
            "source": " ".join(entry["source"].split()),
            "n_listed": len(entry["tasks"]),
            "n_listed_in_verified": len(listed),
            "mean_discrimination_capable_listed": float(rows["discrimination"].mean()),
            "mean_discrimination_capable_other_tasks": float(
                items_cap.drop(index=sorted(listed))["discrimination"].mean()
            ),
            "mean_solve_rate_all_listed": float(items_all.loc[sorted(listed), "solve_rate"].mean()),
            "overlaps": {key: overlap(flagged, listed, n_tasks) for key, flagged in flagged_sets.items()},
            "listed_tasks": [
                {
                    "task": task,
                    "n_solved_all": int(items_all.loc[task, "n_solved"]),
                    "solve_rate_capable": float(items_cap.loc[task, "solve_rate"]),
                    "discrimination_capable": float(items_cap.loc[task, "discrimination"]),
                    "flag_capable": items_cap.loc[task, "flag"],
                    "dif": task in dif_set,
                }
                for task in sorted(listed)
            ],
        }
    R["cross_check"] = cross
    R["flagged_set_sizes"] = {key: len(value) for key, value in flagged_sets.items()}

    # ---- 7. item response model -----------------------------------------
    top_idx = [fit.systems.index(s) for s in top_systems]
    grid = np.linspace(-3.0, 4.0, 15)
    fine = np.linspace(-6.0, 6.0, 241)
    info_fine = fit.test_information(fine)
    top_ability = fit.ability[top_idx]
    R["irt"] = {
        "model": "2PL, marginal maximum likelihood, normal priors on slope (1, 1) and intercept (0, 3)",
        "converged": fit.converged,
        "n_iter": fit.n_iter,
        "log_likelihood": fit.log_likelihood,
        "n_tasks_negative_slope": int((fit.discrimination < 0).sum()),
        "tasks_negative_slope": sorted(np.array(fit.tasks)[fit.discrimination < 0].tolist()),
        "slope_quartiles": [float(q) for q in np.quantile(fit.discrimination, [0.25, 0.5, 0.75])],
        "ability_accuracy_spearman": float(stats.spearmanr(fit.ability, accuracy.to_numpy()).statistic),
        "slope_vs_ctt_discrimination_spearman": float(
            stats.spearmanr(fit.discrimination, items_all["discrimination"], nan_policy="omit").statistic
        ),
        "effective_n_tasks_all_submissions": fit.effective_n_items(),
        "effective_n_tasks_top": fit.effective_n_items(top_ability),
        "n_tasks_for_90pct_information_all_submissions": fit.items_for_information_share(0.9),
        "n_tasks_for_90pct_information_top": fit.items_for_information_share(0.9, top_ability),
        "n_tasks_for_50pct_information_top": fit.items_for_information_share(0.5, top_ability),
        "top_ability_min": float(top_ability.min()),
        "top_ability_median": float(np.median(top_ability)),
        "top_ability_max": float(top_ability.max()),
        "ability_of_peak_information": float(fine[np.argmax(info_fine)]),
        "peak_information": float(info_fine.max()),
        "information_at_median_top_ability": float(fit.test_information(np.median(top_ability))[0]),
        "peak_information_over_information_at_median_top_ability": float(
            info_fine.max() / fit.test_information(np.median(top_ability))[0]
        ),
        "test_information": [
            {"ability": float(t), "information": float(i)} for t, i in zip(grid, fit.test_information(grid))
        ],
    }

    # ---- 8. does the order change without the non-discriminating tasks? --
    def non_discriminating(reference: ResultsMatrix) -> set[str]:
        sets = flag_sets(item_stats(reference, near_zero=NEAR_ZERO))
        return sets["negative"] | sets["near_zero"] | sets["solved_by_all"] | sets["solved_by_none"]

    held_out = capable.select_systems([s for s in capable.systems if s not in set(top_systems)])
    removed_held_out = non_discriminating(held_out)
    removed_all_capable = non_discriminating(capable)
    R["ordering"] = {
        "flags_from_capable_systems_outside_top": {
            "n_reference_systems": held_out.n_systems,
            **ordering_change(matrix, top_systems, removed_held_out),
            "random_removal_baseline": random_removal_baseline(matrix, top_systems, len(removed_held_out), rng),
        },
        "flags_from_all_capable_systems": {
            "n_reference_systems": capable.n_systems,
            **ordering_change(matrix, top_systems, removed_all_capable),
            "random_removal_baseline": random_removal_baseline(
                matrix, top_systems, len(removed_all_capable), rng
            ),
        },
    }

    # ---- 9. unusual response patterns -----------------------------------
    fits = system_fit(matrix, fit=fit, seed=SEED)
    unusual = fits[fits["flag"] != ""].sort_values("guttman_z", ascending=False)
    R["unusual_systems"] = [
        {
            "system": system,
            "resolved_rate": float(row["accuracy"]),
            "guttman_norm": float(row["guttman_norm"]),
            "guttman_expected": float(row["guttman_expected"]),
            "guttman_z": float(row["guttman_z"]),
            "n_hard_solved": int(row["n_hard_solved"]),
            "n_easy_failed": int(row["n_easy_failed"]),
            "mini_swe_agent": bool(meta.loc[system, "agent"] == MINI),
            "n_tasks_without_a_patch": optional_int(meta.loc[system, "n_no_generation"]),
            "n_instances_in_per_instance_file": optional_int(meta.loc[system, "n_instances_listed"]),
        }
        for system, row in unusual.iterrows()
    ]
    z_all = fits["guttman_z"].dropna()
    R["unusual_systems_summary"] = {
        "n_flagged": int(len(unusual)),
        "n_flagged_mini_swe_agent": int((meta.loc[list(unusual.index), "agent"] == MINI).sum()),
        "n_mini_swe_agent_in_matrix": int((meta["agent"] == MINI).sum()),
        "guttman_z_median": float(z_all.median()),
        "guttman_z_sd": float(z_all.std()),
        "share_of_systems_with_z_above_2": float((z_all > 2).mean()),
    }

    # ---- 10. my reading of individual tasks -----------------------------
    R["task_notes"] = [
        {
            "task": note["task"],
            "n_solved_all": int(items_all.loc[note["task"], "n_solved"]),
            "n_all": int(items_all.loc[note["task"], "n"]),
            "n_solved_capable": int(items_cap.loc[note["task"], "n_solved"]),
            "n_capable": int(items_cap.loc[note["task"], "n"]),
            "discrimination_capable": float(items_cap.loc[note["task"], "discrimination"]),
            "rate_mini": float(dif_frame.loc[note["task"], "rate_focal"]),
            "rate_other": float(dif_frame.loc[note["task"], "rate_reference"]),
            "annotated_difficulty": task_index.loc[note["task"], "difficulty"],
            "on_openai_infra_list": note["task"] in set(known["openai_infra_23"]["tasks"]),
            "reading": " ".join(note["reading"].split()),
        }
        for note in notes
    ]

    R = to_jsonable(R)

    # ---- outputs ---------------------------------------------------------
    per_task = (
        items_all.add_suffix("_all")
        .join(items_cap.add_suffix("_capable"))
        .join(irt_items)
        .join(dif_frame[["rate_focal", "rate_reference", "odds_ratio", "p_holm"]].add_prefix("dif_mini_"))
        .join(task_index[["repo", "difficulty"]])
    )
    for name, entry in known.items():
        per_task[f"listed_{name}"] = per_task.index.isin(entry["tasks"])
    per_task.to_csv(args.out / "swebench_verified_item_stats.csv", lineterminator="\n", float_format="%.6g")
    board.to_frame().to_csv(
        args.out / "swebench_verified_top20.csv", index=False, lineterminator="\n", float_format="%.6g"
    )
    (args.out / "swebench_verified_audit.json").write_text(
        json.dumps(R, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "swebench_verified_audit.md").write_text(render(R), encoding="utf-8", newline="\n")
    report(
        matrix,
        args.out,
        name="swebench_verified_report",
        title="SWE-bench Verified: generic itemaudit report",
        top=TOP,
        alpha=ALPHA,
        seed=SEED,
    )
    print(f"wrote results to {args.out}")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def render(R: dict[str, Any]) -> str:
    d, t, top, full = R["data"], R["tasks_all_submissions"], R["top"], R["full_leaderboard"]
    cap, irt, det = R["capable"], R["irt"], R["detectable_difference"]
    board = top["leaderboard"]
    out: list[str] = []
    add = out.append

    add("# SWE-bench Verified leaderboard audit")
    add("")
    add(
        "Generated by `experiments/swebench_verified_audit.py`. Every number below is computed "
        "from the files that `experiments/build_swebench_matrix.py` writes under `data/`; the same "
        "numbers are in `swebench_verified_audit.json`."
    )
    add("")
    add("## Data")
    add("")
    add(f"- Source: {d['source_repository']}, `{d['source_path']}`, commit `{d['source_commit_sha']}` "
        f"({d['source_commit_date']}).")
    add(f"- Task list: `{d['task_list_dataset']}` on Hugging Face, revision `{d['task_list_revision']}`.")
    add(f"- Submission directories: {d['n_submission_directories']}. Used: {d['n_included']} "
        f"({d['n_from_results_json']} from `results/results.json`, {d['n_from_per_instance_details']} "
        f"from `per_instance_details.json`). Left out: {d['n_excluded']} "
        f"({d['n_excluded_no_per_task_results']} with no per-task results file, "
        f"{d['n_excluded_inconsistent_with_reported_score']} whose per-task file contradicts the score in "
        f"their own metadata by more than {d['max_score_gap_pct_allowed']} point).")
    add("")
    add(markdown_table(pd.DataFrame([{"submission": k, "reason": v} for k, v in d["excluded"].items()])))
    add("")
    add(f"- Of the submissions used, {d['n_with_reported_score']} carry a score in their metadata; "
        f"{d['n_included_reported_score_differs_over_0.1pt']} of those differ from the per-task count by "
        "more than 0.1 point (and by less than the exclusion limit).")
    add(f"- {d['n_reporting_no_generation_count']} of the submissions used say how many tasks they "
        f"produced no patch for. {d['n_with_no_generation_on_10_or_more_tasks']} of them have 10 or more "
        f"such tasks (the largest count is {d['max_no_generation']}). The leaderboard scores these as "
        "failures and so does the matrix, but a task that was never attempted says nothing about how "
        "hard the task is.")
    add(f"- Resolved rate across the {t['n_submissions']} submissions: minimum {pct(t['resolved_rate_min'])}, "
        f"median {pct(t['resolved_rate_median'])}, maximum {pct(t['resolved_rate_max'])}.")
    add("")

    add("## Tasks")
    add("")
    f = t["flags"]
    add(f"- Solved by every submission: {f['solved_by_all']}. Solved by none: {f['solved_by_none']}.")
    add(f"- Solved by at most 5% of submissions: {t['n_tasks_solved_by_at_most_5pct']}. "
        f"By at least 95%: {t['n_tasks_solved_by_at_least_95pct']}.")
    q = t["solve_rate_quartiles"]
    add(f"- Solve-rate quartiles: {q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}.")
    add(f"- KR-20 reliability of the total score: {t['kr20']:.4f}.")
    add("")
    add(markdown_table(pd.DataFrame(t["solve_rate_histogram"])))
    add("")
    add("Solve rate against the human difficulty label that ships with the dataset:")
    add("")
    add(markdown_table(pd.DataFrame(R["annotated_difficulty"]), {"mean_solve_rate": ".3f"}))
    add("")
    add(f"Among the top {board['n_systems']} submissions: {top['n_tasks_solved_by_all_top']} tasks are solved "
        f"by all of them, {top['n_tasks_solved_by_none_of_top']} by none, and "
        f"{top['n_tasks_where_top_systems_disagree']} by some but not all. Only the last group can change "
        "their order.")
    add("")

    add(f"## Top {board['n_systems']}")
    add("")
    rows = pd.DataFrame(board["rows"])
    table = pd.DataFrame(
        {
            "rank": rows["rank"],
            "submission": rows["system"],
            "resolved": rows["n_solved"],
            "rate": rows["accuracy"],
            "95% CI": [f"{lo:.3f} to {hi:.3f}" for lo, hi in zip(rows["ci_low"], rows["ci_high"])],
            "rank interval": rows["rank_low"].astype(str) + " to " + rows["rank_high"].astype(str),
            "bootstrap rank": rows["boot_rank_low"].astype(str) + " to " + rows["boot_rank_high"].astype(str),
            "tier": rows["tier"],
        }
    )
    add(markdown_table(table, {"rate": ".3f"}))
    add("")
    add(f"The rank interval comes from Holm-corrected tests against the other {board['n_systems'] - 1}; "
        "the bootstrap rank is a 95% percentile interval over task resamples, without a multiplicity "
        f"correction. {top['n_outside_top_tied_with_last']} submission(s) outside the top "
        f"{board['n_systems']} have the same resolved count as rank {board['n_systems']}; ties at the "
        "cut-off are broken by directory name.")
    add("")
    add(f"- Adjacent pairs that are statistically indistinguishable at {board['alpha']} (exact McNemar, "
        f"Holm over the {board['n_adjacent_pairs']} adjacent pairs): "
        f"**{board['n_adjacent_indistinguishable']} of {board['n_adjacent_pairs']}**.")
    add(f"- Largest gap between adjacent ranks: {pct(top['largest_adjacent_gap'])}.")
    add(f"- All {board['n_pairs']} pairs, Holm-corrected: {board['n_pairs_indistinguishable']} "
        f"indistinguishable. Smallest Holm-adjusted p: {top['smallest_holm_p']:.4g}.")
    add(f"- Without any correction, {top['n_pairs_raw_p_below_alpha']} of {board['n_pairs']} pairs have "
        f"p < {board['alpha']}; the smallest raw p is {top['smallest_raw_p']:.4g}.")
    fl = top["first_vs_last"]
    add(f"- Rank 1 against rank {board['n_systems']}: {100 * fl['diff']:+.1f} points, 95% CI "
        f"{100 * fl['ci_low']:+.1f} to {100 * fl['ci_high']:+.1f} (bootstrap {100 * fl['boot_ci_low']:+.1f} "
        f"to {100 * fl['boot_ci_high']:+.1f}), {fl['n_only_a']} tasks solved only by rank 1 and "
        f"{fl['n_only_b']} only by rank {board['n_systems']}, McNemar p = {fl['p_value']:.4g}.")
    add(f"- Tiers among the top {board['n_systems']}: {board['n_tiers']}.")
    add(f"- Composition: {top['n_top_from_mini_swe_agent']} are mini-SWE-agent runs, "
        f"{top['n_top_naming_a_claude_model']} name a Claude model in their model tag or directory name, "
        f"{top['n_top_checked_by_maintainers']} carry the maintainers' `checked: true` mark.")
    add("")
    add("Adjacent pairs:")
    add("")
    adjacent = pd.DataFrame(board["adjacent"])[
        ["rank", "diff", "ci_low", "ci_high", "n_only_upper", "n_only_lower", "discordance", "p_value", "p_holm"]
    ]
    add(markdown_table(adjacent, {"diff": "+.3f", "ci_low": "+.3f", "ci_high": "+.3f", "discordance": ".3f",
                                  "p_value": ".3f", "p_holm": ".3f"}))
    add("")
    add(f"Whole leaderboard ({full['n_systems']} submissions): {full['n_adjacent_indistinguishable']} of "
        f"{full['n_adjacent_pairs']} adjacent pairs are indistinguishable, and the submissions fall into "
        f"{full['n_tiers']} tiers. Going down from rank 1, the first submission that differs from it at "
        f"raw p < {board['alpha']} is at rank {full['first_rank_with_raw_p_below_alpha_vs_rank_1']}; with "
        f"Holm correction over the {full['n_systems'] - 1} comparisons against rank 1 it is at rank "
        f"{full['first_rank_with_holm_p_below_alpha_vs_rank_1']}.")
    add("")

    add("## What 500 tasks can resolve")
    add("")
    add("Smallest difference in resolved rate between two submissions that a paired test can pick up, "
        "at the task-level disagreement seen between pairs of top-20 submissions:")
    add("")
    table = pd.DataFrame(
        [
            {
                "disagreement": name.replace("top20_pairs_", ""),
                "share of tasks where the pair differs": pct(v["discordance"]),
                "smallest significant difference": pct(v["min_significant_difference"], 2),
                "detectable with 80% power": pct(v["min_detectable_difference_power_0.8"], 2),
            }
            for name, v in det["levels"].items()
        ]
    )
    add(markdown_table(table))
    add("")
    u = det["unpaired_at_median_top20_accuracy"]
    add(f"An unpaired comparison at a resolved rate of {pct(u['accuracy'])} would need "
        f"{pct(u['min_significant_difference'], 2)} to be significant and "
        f"{pct(u['min_detectable_difference_power_0.8'], 2)} for 80% power. The whole top 20 spans "
        f"{pct(det['top20_spread_first_minus_last'])}.")
    add("")

    add("## Tasks with negative or near-zero discrimination among capable submissions")
    add("")
    cf = cap["flags"]
    add(f"Capable means a resolved rate of at least {pct(cap['threshold'], 0)}: {cap['n_systems']} "
        f"submissions ({', '.join(f'{n} from {y}' for y, n in cap['n_per_year'].items())}).")
    add("")
    add(f"- Negative discrimination: {cf['negative']} tasks ({cf['negative_ci_excludes_zero']} with a 95% "
        "interval entirely below zero).")
    add(f"- Discrimination between 0 and {R['settings']['near_zero_discrimination_below']}: {cf['near_zero']}.")
    add(f"- Solved by all capable submissions: {cf['solved_by_all']}. By none: {cf['solved_by_none']}.")
    add("")
    add("Sensitivity to the threshold:")
    add("")
    table = pd.DataFrame(
        [
            {
                "min resolved rate": thr,
                "submissions": v["n_systems"],
                "negative": v["negative"],
                "negative, CI below 0": v["negative_ci_excludes_zero"],
                "near zero": v["near_zero"],
                "solved by all": v["solved_by_all"],
                "solved by none": v["solved_by_none"],
            }
            for thr, v in cap["sensitivity_to_threshold"].items()
        ]
    )
    add(markdown_table(table))
    add("")
    add("The negative tasks, most negative first. `disc` is among capable submissions; `disc, all` and "
        "`2PL slope, all` use all submissions. The year columns give the solve rate among capable "
        "submissions by submission year.")
    add("")
    neg = pd.DataFrame(cap["negative_tasks"])
    years = [c for c in neg.columns if c.startswith("solve_rate_capable_")]
    table = pd.DataFrame(
        {
            "task": neg["task"],
            "solved": neg["n_solved_capable"].astype(str) + "/" + neg["n_capable"].astype(str),
            "disc": neg["discrimination_capable"],
            "95% CI": [f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(neg["disc_ci_low"], neg["disc_ci_high"])],
            "disc, all": neg["discrimination_all"],
            "2PL slope, all": neg["irt_discrimination"],
            **{c.replace("solve_rate_capable_", ""): neg[c] for c in years},
            "label": neg["annotated_difficulty"],
        }
    )
    add(markdown_table(table, {"disc": "+.3f", "disc, all": "+.3f", "2PL slope, all": "+.2f",
                               **{c.replace("solve_rate_capable_", ""): ".2f" for c in years}}))
    add("")
    sh, sc = cap["split_half_replication"], cap["scaffold_split"]
    add("Are these flags noise? Two checks:")
    add("")
    add(f"- Random halves ({sh['n_splits']} splits of the capable submissions): on average "
        f"{sh['mean_n_negative_in_first_half']:.1f} tasks are negative in the first half, and "
        f"{pct(sh['mean_share_also_negative_in_second_half'])} of them are also negative in the second half, "
        f"against {pct(sh['mean_share_negative_in_second_half_overall'])} of all tasks.")
    add(f"- Disjoint groups ({sc['n_mini_swe_agent']} mini-SWE-agent runs against {sc['n_other']} other "
        f"submissions): {sc['n_negative_among_other']} tasks are negative among the others, and "
        f"{pct(sc['share_of_those_also_negative_among_mini'])} of them are also negative among the "
        f"mini-SWE-agent runs, against {pct(sc['share_negative_among_mini_overall'])} of all tasks. The "
        f"correlation of task discrimination between the two groups is {sc['discrimination_correlation']:.2f}.")
    add("")

    dd = R["dif_mini_vs_other"]
    add("## Tasks whose result depends on which group ran them")
    add("")
    add(f"Mantel-Haenszel test among the capable submissions, {dd['n_focal']} mini-SWE-agent runs against "
        f"{dd['n_reference']} others, matched on overall resolved rate, Holm-corrected over 500 tasks: "
        f"{dd['n_flagged']} tasks flagged. An empty odds ratio means it has no finite estimate (one "
        "group solved the task every time).")
    add("")
    if dd["tasks"]:
        add(markdown_table(pd.DataFrame(dd["tasks"]), {"rate_mini": ".3f", "rate_other": ".3f",
                                                       "odds_ratio": ".3g", "chi2": ".1f",
                                                       "p_value": ".2g", "p_holm": ".2g"}))
        add("")

    add("## Cross-check against public lists of problematic tasks")
    add("")
    sizes = R["flagged_set_sizes"]
    add("Flagged sets: " + ", ".join(f"{k} ({v})" for k, v in sizes.items()) + ".")
    add("")
    for name, entry in R["cross_check"].items():
        add(f"### {name}")
        add("")
        add(entry["description"])
        add("")
        add(f"Source: {entry['source']}")
        add("")
        add(f"{entry['n_listed']} tasks listed, {entry['n_listed_in_verified']} of them in SWE-bench "
            f"Verified. Mean discrimination among capable submissions: "
            f"{entry['mean_discrimination_capable_listed']:.3f} for listed tasks, "
            f"{entry['mean_discrimination_capable_other_tasks']:.3f} for the rest.")
        add("")
        table = pd.DataFrame(
            [
                {
                    "flagged set": key,
                    "flagged": v["n_flagged"],
                    "overlap": v["overlap"],
                    "expected by chance": v["expected_by_chance"],
                    "p (hypergeometric)": v["p_hypergeometric"],
                    "tasks": ", ".join(v["overlap_tasks"]),
                }
                for key, v in entry["overlaps"].items()
            ]
        )
        add(markdown_table(table, {"expected by chance": ".2f", "p (hypergeometric)": ".2g"}))
        add("")

    add("## Item response model")
    add("")
    add(f"- {irt['model']}. Converged: {irt['converged']} after {irt['n_iter']} iterations.")
    add(f"- Tasks with a negative fitted slope: {irt['n_tasks_negative_slope']}. Slope quartiles: "
        + ", ".join(f"{q:.2f}" for q in irt["slope_quartiles"]) + ".")
    add(f"- Spearman correlation of fitted ability with resolved rate: "
        f"{irt['ability_accuracy_spearman']:.4f}; of fitted slope with classical discrimination: "
        f"{irt['slope_vs_ctt_discrimination_spearman']:.3f}.")
    add(f"- Effective number of informative tasks: {irt['effective_n_tasks_all_submissions']:.1f} over all "
        f"submissions, {irt['effective_n_tasks_top']:.1f} over the top {board['n_systems']}.")
    add(f"- Tasks holding 90% of the information: {irt['n_tasks_for_90pct_information_all_submissions']} "
        f"over all submissions, {irt['n_tasks_for_90pct_information_top']} over the top "
        f"{board['n_systems']} ({irt['n_tasks_for_50pct_information_top']} tasks hold half of it there).")
    add(f"- Test information peaks at ability {irt['ability_of_peak_information']:+.2f} "
        f"({irt['peak_information']:.1f}). The top {board['n_systems']} sit between "
        f"{irt['top_ability_min']:+.2f} and {irt['top_ability_max']:+.2f} (median "
        f"{irt['top_ability_median']:+.2f}), where the information is "
        f"{irt['information_at_median_top_ability']:.1f}, a ratio of "
        f"{irt['peak_information_over_information_at_median_top_ability']:.2f}.")
    add("")
    add(markdown_table(pd.DataFrame(irt["test_information"]), {"ability": "+.1f", "information": ".1f"}))
    add("")

    add("## Does the order of the top 20 change without the non-discriminating tasks?")
    add("")
    add("Non-discriminating means negative discrimination, discrimination below "
        f"{R['settings']['near_zero_discrimination_below']}, or solved by all or by none of the reference "
        "submissions. In the first variant the reference submissions are the capable ones outside the top "
        "20, so the tasks are not chosen using the submissions being re-ranked.")
    add("")
    for key, label in (
        ("flags_from_capable_systems_outside_top", "Flags from capable submissions outside the top 20"),
        ("flags_from_all_capable_systems", "Flags from all capable submissions"),
    ):
        o = R["ordering"][key]
        b = o["random_removal_baseline"]
        add(f"### {label}")
        add("")
        add(f"- Reference submissions: {o['n_reference_systems']}. Tasks removed: {o['n_tasks_removed']}; "
            f"remaining: {o['n_tasks_remaining']}.")
        add(f"- Kendall tau-b between the two orders: {o['kendall_tau_b']:.3f} (Spearman "
            f"{o['spearman']:.3f}). Removing the same number of random tasks gives a mean tau-b of "
            f"{b['kendall_tau_b_mean']:.3f} (5th to 95th percentile {b['kendall_tau_b_p05']:.3f} to "
            f"{b['kendall_tau_b_p95']:.3f}, {b['n_draws']} draws).")
        add(f"- Submissions whose rank changes: {o['n_systems_whose_rank_changed']} of "
            f"{len(o['table'])}; largest shift {o['max_rank_shift']} places.")
        add(f"- Rank 1 before: {o['top_system_before']}. Rank 1 after: "
            f"{', '.join(o['systems_ranked_first_after'])}.")
        add(f"- Adjacent pairs still indistinguishable: {o['n_adjacent_indistinguishable_after']} of "
            f"{len(o['table']) - 1}; all pairs: {o['n_pairs_indistinguishable_after']} of {o['n_pairs']}.")
        add("")
        add(markdown_table(pd.DataFrame(o["table"]), {"resolved_before": ".3f", "resolved_after": ".3f"}))
        add("")

    add("## Submissions with unusual response patterns")
    add("")
    us = R["unusual_systems_summary"]
    add("A submission is listed when its normalised Guttman error count (hard tasks solved while easier "
        "ones are failed) is more than three standard deviations above what the fitted 2PL model "
        "produces for a system of the same ability. This marks a pattern worth a look. It is not "
        "evidence of anything about how a submission was produced: a submission that has no patch for "
        "some tasks fails easy tasks for that reason alone (see the last two columns), tasks whose result depends on the evaluation "
        "setup produce the same signature, and the model itself fits imperfectly (across all "
        f"submissions the z-scores have a standard deviation of {us['guttman_z_sd']:.2f} where a "
        f"well-fitting model would give about 1, and {pct(us['share_of_systems_with_z_above_2'])} of "
        "submissions are above 2).")
    add("")
    add(f"{us['n_flagged']} submissions are listed; {us['n_flagged_mini_swe_agent']} of them are "
        f"mini-SWE-agent runs (which make up {us['n_mini_swe_agent_in_matrix']} of the "
        f"{R['tasks_all_submissions']['n_submissions']} submissions).")
    add("")
    if R["unusual_systems"]:
        table = pd.DataFrame(R["unusual_systems"])
        for column in ("n_tasks_without_a_patch", "n_instances_in_per_instance_file"):
            table[column] = ["" if row[column] is None else str(row[column]) for row in R["unusual_systems"]]
        add(markdown_table(table, {"resolved_rate": ".3f", "guttman_norm": ".3f", "guttman_expected": ".3f",
                                   "guttman_z": ".2f"}))
    else:
        add("None.")
    add("")

    add("## My reading of individual tasks")
    add("")
    add("These notes are my own reading of the public task text and tests. I did not run the tasks or "
        "read any submission's patch. The explanations are hypotheses.")
    add("")
    for note in R["task_notes"]:
        add(f"### {note['task']}")
        add("")
        add(f"Solved by {note['n_solved_all']} of {note['n_all']} submissions "
            f"({note['n_solved_capable']} of {note['n_capable']} capable ones; "
            f"{pct(note['rate_mini'])} of mini-SWE-agent runs against {pct(note['rate_other'])} of other "
            f"capable submissions). Discrimination among capable submissions: "
            + ("not defined" if note["discrimination_capable"] is None else f"{note['discrimination_capable']:+.3f}")
            + f". Annotated difficulty: {note['annotated_difficulty']}."
            + (" On OpenAI's list of 23." if note["on_openai_infra_list"] else ""))
        add("")
        add(note["reading"])
        add("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
