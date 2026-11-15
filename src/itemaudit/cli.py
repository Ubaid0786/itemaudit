"""Command line interface: ``itemaudit report | compare | fetch``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .matrix import load_results
from .report import report
from .stats import compare


def _cmd_report(args: argparse.Namespace) -> int:
    matrix = load_results(args.results)
    result = report(
        matrix,
        args.out,
        name=args.name,
        title=args.title or f"Audit of {Path(args.results).name}",
        top=args.top,
        alpha=args.alpha,
        fit_irt=not args.no_irt,
        seed=args.seed,
    )
    summary = result["summary"]
    board = result["leaderboard"]
    print(f"{summary['n_systems']} systems x {summary['n_tasks']} tasks")
    print(
        f"adjacent ranks indistinguishable: {board['n_adjacent_indistinguishable']} "
        f"of {board['n_adjacent_pairs']} (top {board['n_systems']})"
    )
    print(
        f"tasks solved by all: {summary['n_tasks_solved_by_all']}, by none: "
        f"{summary['n_tasks_solved_by_none']}, negative discrimination: "
        f"{summary['n_tasks_negative_discrimination']}"
    )
    out = Path(args.out)
    print(f"wrote {out / (args.name + '.md')} and {out / (args.name + '.json')}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    matrix = load_results(args.results)
    result = compare(
        matrix.row(args.system_a),
        matrix.row(args.system_b),
        names=(args.system_a, args.system_b),
        confidence=1.0 - args.alpha,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .datasets import fetch_swebench_verified

    built = fetch_swebench_verified(args.data_dir, ref=args.ref, refresh=args.refresh)
    p = built.provenance
    print(f"source commit: {p['source_commit_sha']} ({p['source_commit_date']})")
    print(
        f"{p['n_submission_directories']} submission directories, {p['n_included']} used, "
        f"{p['n_excluded']} left out ({p['n_excluded_no_per_task_results']} without per-task results, "
        f"{p['n_excluded_inconsistent_with_reported_score']} contradicting their own reported score)"
    )
    print(f"wrote {Path(args.data_dir) / 'swebench_verified_matrix.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itemaudit",
        description="Audit an agent benchmark from its results matrix (systems x tasks, pass/fail).",
    )
    parser.add_argument("--version", action="version", version=f"itemaudit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="write a Markdown and JSON audit of a results CSV")
    p_report.add_argument("results", help="CSV: first column system name, one column per task")
    p_report.add_argument("--out", default=".", help="output directory (default: current)")
    p_report.add_argument("--name", default="report", help="output file stem (default: report)")
    p_report.add_argument("--title", default=None)
    p_report.add_argument("--top", type=int, default=20, help="rank and test the top N systems")
    p_report.add_argument("--alpha", type=float, default=0.05)
    p_report.add_argument("--no-irt", action="store_true", help="skip the 2PL fit")
    p_report.add_argument("--seed", type=int, default=0)
    p_report.set_defaults(func=_cmd_report)

    p_compare = sub.add_parser("compare", help="paired comparison of two systems")
    p_compare.add_argument("results")
    p_compare.add_argument("system_a")
    p_compare.add_argument("system_b")
    p_compare.add_argument("--alpha", type=float, default=0.05)
    p_compare.add_argument("--seed", type=int, default=0)
    p_compare.add_argument("--json", action="store_true", help="print JSON instead of text")
    p_compare.set_defaults(func=_cmd_compare)

    p_fetch = sub.add_parser("fetch", help="download a public leaderboard as a results CSV")
    p_fetch.add_argument("dataset", choices=["swebench-verified"])
    p_fetch.add_argument("--data-dir", default="data", help="where to write (default: ./data)")
    p_fetch.add_argument("--ref", default="main", help="commit or branch of swe-bench/experiments")
    p_fetch.add_argument("--refresh", action="store_true", help="ignore the local cache")
    p_fetch.set_defaults(func=_cmd_fetch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"itemaudit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
