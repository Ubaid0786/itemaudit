"""One-call audit: writes a Markdown report and the same content as JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .irt import fit_2pl
from .items import FLAG_ALL, FLAG_NEAR_ZERO, FLAG_NEGATIVE, FLAG_NONE, item_stats, kr20, system_fit
from .leaderboard import leaderboard
from .matrix import ResultsMatrix
from .stats import min_detectable_difference, min_significant_difference


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas values to plain JSON types; nan and inf become null."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.reset_index().to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is pd.NA:
        return None
    return value


def markdown_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    """Render a frame as a GitHub-flavoured Markdown table (index not included)."""
    formats = formats or {}

    def cell(column: str, value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if column in formats:
            return format(value, formats[column])
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(c, v) for c, v in zip(frame.columns, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def solve_rate_histogram(solve_rate: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Task counts by solve-rate band. 0 and 1 get their own rows; bands are (low, high]."""
    rate = solve_rate[~np.isnan(solve_rate)]
    rows: list[dict[str, Any]] = [{"band": "exactly 0", "n_tasks": int(np.sum(rate == 0.0))}]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    inner = rate[(rate > 0.0) & (rate < 1.0)]
    for low, high in zip(edges[:-1], edges[1:]):
        last = high == 1.0
        # Rounding to 12 places keeps rates such as 0.3 on the right side of an edge.
        count = int(np.sum((inner > low + 1e-12) & (inner <= high + 1e-12)))
        label = f"{low:.1f} to {high:.1f}" + (" (below 1)" if last else "")
        rows.append({"band": label, "n_tasks": count})
    rows.append({"band": "exactly 1", "n_tasks": int(np.sum(rate == 1.0))})
    return rows


def audit(
    matrix: ResultsMatrix,
    *,
    top: int | None = 20,
    alpha: float = 0.05,
    near_zero: float = 0.1,
    fit_irt: bool = True,
    n_boot: int = 2_000,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Run every analysis and return the results as one JSON-serialisable dict."""
    items = item_stats(matrix, near_zero=near_zero, confidence=1.0 - alpha)
    fit = fit_2pl(matrix) if fit_irt and matrix.n_systems >= 2 and matrix.n_tasks >= 2 else None
    systems = system_fit(matrix, fit=fit, model_reference=fit is not None, seed=seed)
    board = leaderboard(matrix, top=top, alpha=alpha, n_boot=n_boot, seed=seed)

    k = len(board.rows)
    upper = np.triu_indices(k, k=1)
    discordances = board.discordance[upper]
    n_shared = int(np.median([r.n_tasks for r in board.rows])) if k else 0
    detectable: dict[str, Any] = {"n_tasks": n_shared}
    if discordances.size and n_shared:
        for label, q in (("p25", 0.25), ("median", 0.5), ("p75", 0.75)):
            level = float(np.quantile(discordances, q))
            detectable[label] = {
                "discordance": level,
                "min_significant_difference": min_significant_difference(n_shared, level, alpha),
                "min_detectable_difference_power_0.8": min_detectable_difference(
                    n_shared, level, alpha, 0.8
                ),
            }

    flags = items["flag"]
    result: dict[str, Any] = {
        "summary": {
            "n_systems": matrix.n_systems,
            "n_tasks": matrix.n_tasks,
            "n_missing_cells": int((~matrix.observed).sum()),
            "kr20": kr20(matrix),
            "n_tasks_solved_by_all": int((flags == FLAG_ALL).sum()),
            "n_tasks_solved_by_none": int((flags == FLAG_NONE).sum()),
            "n_tasks_negative_discrimination": int((flags == FLAG_NEGATIVE).sum()),
            "n_tasks_near_zero_discrimination": int((flags == FLAG_NEAR_ZERO).sum()),
            "near_zero_threshold": near_zero,
            "solve_rate_histogram": solve_rate_histogram(items["solve_rate"].to_numpy()),
        },
        "leaderboard": board.to_dict(),
        "detectable_difference": detectable,
        "systems": systems,
    }

    if fit is not None:
        items = items.join(fit.items_frame())
        ranked = [fit.systems.index(s) for s in board.systems]
        grid = np.linspace(-3.0, 3.0, 13)
        result["irt"] = {
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "log_likelihood": fit.log_likelihood,
            "n_tasks_negative_slope": int((fit.discrimination < 0).sum()),
            "effective_n_tasks_all_systems": fit.effective_n_items(),
            "effective_n_tasks_ranked_systems": fit.effective_n_items(fit.ability[ranked]),
            "n_tasks_for_90pct_information_all_systems": fit.items_for_information_share(0.9),
            "n_tasks_for_90pct_information_ranked_systems": fit.items_for_information_share(
                0.9, fit.ability[ranked]
            ),
            "test_information": [
                {"ability": float(t), "information": float(i)}
                for t, i in zip(grid, fit.test_information(grid))
            ],
            "abilities": fit.systems_frame(),
        }
    result["items"] = items
    return to_jsonable(result)


def render_markdown(result: dict[str, Any], title: str = "Benchmark audit") -> str:
    """Render the dict from :func:`audit` as Markdown."""
    s = result["summary"]
    board = result["leaderboard"]
    lines = [f"# {title}", ""]
    lines += [
        "## Summary",
        "",
        f"- Systems: {s['n_systems']}",
        f"- Tasks: {s['n_tasks']}",
        f"- Missing cells: {s['n_missing_cells']}",
        "- KR-20 reliability of the total score: "
        + ("not defined" if s["kr20"] is None else f"{s['kr20']:.4f}"),
        f"- Tasks solved by every system: {s['n_tasks_solved_by_all']}",
        f"- Tasks solved by no system: {s['n_tasks_solved_by_none']}",
        f"- Tasks with negative discrimination: {s['n_tasks_negative_discrimination']}",
        f"- Tasks with discrimination between 0 and {s['near_zero_threshold']}: "
        f"{s['n_tasks_near_zero_discrimination']}",
        "",
        "### Solve-rate distribution",
        "",
        markdown_table(pd.DataFrame(s["solve_rate_histogram"])),
        "",
    ]

    rows = pd.DataFrame(board["rows"])
    lines += [f"## Leaderboard ({board['n_systems']} systems ranked)", ""]
    if len(rows):
        table = pd.DataFrame(
            {
                "rank": rows["rank"],
                "system": rows["system"],
                "solved": rows["n_solved"].astype(str) + "/" + rows["n_tasks"].astype(str),
                "accuracy": rows["accuracy"],
                "ci": [f"{lo:.3f} to {hi:.3f}" for lo, hi in zip(rows["ci_low"], rows["ci_high"])],
                "rank interval": rows["rank_low"].astype(str) + " to " + rows["rank_high"].astype(str),
                "bootstrap rank": rows["boot_rank_low"].astype(str) + " to " + rows["boot_rank_high"].astype(str),
                "tier": rows["tier"],
            }
        )
        lines += [markdown_table(table, {"accuracy": ".4f"}), ""]
    lines += [
        f"Rank intervals and tiers use exact McNemar tests on all {board['n_pairs']} pairs with Holm "
        f"correction at alpha = {board['alpha']}. {board['n_pairs_indistinguishable']} of those pairs "
        f"are not distinguishable. The ranked systems fall into {board['n_tiers']} tier(s).",
        "",
        "### Adjacent ranks",
        "",
        f"{board['n_adjacent_indistinguishable']} of {board['n_adjacent_pairs']} adjacent pairs are "
        f"statistically indistinguishable (Holm-corrected over the adjacent pairs).",
        "",
    ]
    adjacent = pd.DataFrame(board["adjacent"])
    if len(adjacent):
        table = adjacent[
            ["rank", "system_upper", "system_lower", "diff", "ci_low", "ci_high",
             "n_only_upper", "n_only_lower", "p_value", "p_holm", "distinguishable"]
        ]
        lines += [
            markdown_table(
                table,
                {"diff": "+.4f", "ci_low": "+.4f", "ci_high": "+.4f", "p_value": ".4g", "p_holm": ".4g"},
            ),
            "",
        ]

    det = result["detectable_difference"]
    if "median" in det:
        lines += [
            "### Smallest difference this benchmark can resolve",
            "",
            f"For two systems on {det['n_tasks']} shared tasks, at the disagreement levels seen "
            "among the ranked systems:",
            "",
        ]
        table = pd.DataFrame(
            [
                {
                    "disagreement level": label,
                    "share of tasks where the two differ": det[label]["discordance"],
                    "smallest significant difference": det[label]["min_significant_difference"],
                    "detectable with 80% power": det[label]["min_detectable_difference_power_0.8"],
                }
                for label in ("p25", "median", "p75")
            ]
        )
        lines += [markdown_table(table, {c: ".4f" for c in table.columns[1:]}), ""]

    items = pd.DataFrame(result["items"])
    lines += ["## Suspect tasks", ""]
    negative = items[items["flag"] == FLAG_NEGATIVE].sort_values("discrimination")
    if len(negative):
        columns = ["task", "n_solved", "n", "solve_rate", "discrimination", "disc_ci_low", "disc_ci_high"]
        formats = {c: ".3f" for c in columns[3:]}
        if "irt_discrimination" in negative:
            columns.append("irt_discrimination")
            formats["irt_discrimination"] = ".3f"
        lines += [
            "Tasks that weaker systems solve more often than stronger ones "
            f"(most negative first, up to 50 of {len(negative)}):",
            "",
            markdown_table(negative[columns].head(50), formats),
            "",
        ]
    else:
        lines += ["No task has negative discrimination.", ""]

    if "irt" in result:
        irt = result["irt"]
        lines += [
            "## Item response model (2PL)",
            "",
            f"- Converged: {irt['converged']} after {irt['n_iter']} iterations",
            f"- Tasks with a negative fitted slope: {irt['n_tasks_negative_slope']}",
            f"- Effective number of informative tasks over all systems: "
            f"{irt['effective_n_tasks_all_systems']:.1f}",
            f"- Effective number of informative tasks over the ranked systems: "
            f"{irt['effective_n_tasks_ranked_systems']:.1f}",
            f"- Tasks holding 90% of the information over all systems: "
            f"{irt['n_tasks_for_90pct_information_all_systems']}",
            f"- Tasks holding 90% of the information over the ranked systems: "
            f"{irt['n_tasks_for_90pct_information_ranked_systems']}",
            "",
            "### Test information",
            "",
            markdown_table(pd.DataFrame(irt["test_information"]), {"ability": "+.1f", "information": ".2f"}),
            "",
        ]

    systems = pd.DataFrame(result["systems"])
    unusual = systems[systems["flag"] != ""].sort_values("guttman_z", ascending=False)
    lines += ["## Systems with unusual response patterns", ""]
    if len(unusual):
        lines += [
            "These systems solve hard tasks while failing easy ones more than the fitted model "
            "expects of a system with their score. That is a reason to look, not a finding about "
            "the system.",
            "",
            markdown_table(
                unusual[
                    ["system", "accuracy", "guttman_norm", "guttman_expected", "guttman_z",
                     "n_hard_solved", "n_easy_failed"]
                ],
                {"accuracy": ".4f", "guttman_norm": ".4f", "guttman_expected": ".4f", "guttman_z": ".2f"},
            ),
            "",
        ]
    else:
        lines += ["None flagged.", ""]
    return "\n".join(lines)


def report(
    matrix: ResultsMatrix,
    out_dir: str | Path = ".",
    *,
    name: str = "report",
    title: str = "Benchmark audit",
    top: int | None = 20,
    alpha: float = 0.05,
    near_zero: float = 0.1,
    fit_irt: bool = True,
    n_boot: int = 2_000,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Audit ``matrix`` and write ``<name>.md`` and ``<name>.json`` into ``out_dir``.

    Returns the audit as a dict (the content of the JSON file).
    """
    result = audit(
        matrix, top=top, alpha=alpha, near_zero=near_zero, fit_irt=fit_irt, n_boot=n_boot, seed=seed
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    (out / f"{name}.md").write_text(render_markdown(result, title), encoding="utf-8", newline="\n")
    return result
