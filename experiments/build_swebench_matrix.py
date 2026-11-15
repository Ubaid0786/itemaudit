"""Build data/swebench_verified_matrix.csv from the public SWE-bench leaderboard files.

Downloads only the per-submission result and metadata files of
swe-bench/experiments (evaluation/verified) through the GitHub API, caches
them under data/raw/, and writes:

    data/swebench_verified_matrix.csv        submissions x tasks, 1 = resolved
    data/swebench_verified_submissions.csv   metadata and counts per submission
    data/swebench_verified_build.json        provenance and counts

Rerunning reuses the cache. Pass --refresh to download again, and --ref to
build from a different commit of the source repository.

    python experiments/build_swebench_matrix.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from itemaudit.datasets import fetch_swebench_verified

# The commit of swe-bench/experiments the committed results were computed from.
PINNED_COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=PINNED_COMMIT, help="commit, branch or tag of swe-bench/experiments")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    built = fetch_swebench_verified(args.data_dir, ref=args.ref, refresh=args.refresh)
    print(json.dumps(built.provenance, indent=2))
    print(f"matrix: {built.matrix.n_systems} submissions x {built.matrix.n_tasks} tasks")


if __name__ == "__main__":
    main()
