"""Loaders for public leaderboards. Currently: SWE-bench Verified.

The source is the ``swe-bench/experiments`` repository on GitHub, which holds
one directory per leaderboard submission under ``evaluation/verified``. Only
the small per-submission result and metadata files are downloaded (through
the Git blobs API), never the repository itself. Downloaded files are parsed
as JSON/YAML data and nothing in them is executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .matrix import ResultsMatrix

EXPERIMENTS_REPO = "swe-bench/experiments"
VERIFIED_PATH = "evaluation/verified"
HF_DATASET = "princeton-nlp/SWE-bench_Verified"
N_VERIFIED_TASKS = 500

# A submission whose per-task file disagrees with the score in its own
# metadata by more than this many percentage points is treated as malformed.
MAX_SCORE_GAP_PCT = 1.0

# Files worth fetching from a submission directory, relative to it.
_WANTED = ("metadata.yaml", "metadata.yml", "results/results.json", "per_instance_details.json")


class FetchError(RuntimeError):
    pass


# -- transport -------------------------------------------------------------


def _github_get(endpoint: str, *, raw: bool = False) -> bytes:
    """GET a GitHub REST endpoint, via the ``gh`` CLI if present, else HTTPS.

    The HTTPS fallback uses ``GITHUB_TOKEN`` or ``GH_TOKEN`` when set;
    without a token GitHub's anonymous rate limit (60 requests an hour) is
    too low for a full fetch.
    """
    accept = "application/vnd.github.raw" if raw else "application/vnd.github+json"
    gh = shutil.which("gh")
    if gh:
        proc = subprocess.run(
            [gh, "api", endpoint, "-H", f"Accept: {accept}"],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise FetchError(f"gh api {endpoint} failed: {proc.stderr.decode(errors='replace').strip()}")
        return proc.stdout
    request = urllib.request.Request(f"https://api.github.com/{endpoint}", headers={"Accept": accept})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise FetchError(f"GET {endpoint} failed: {exc}") from exc


def _http_json(url: str, attempts: int = 6) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "itemaudit"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(5.0 * (attempt + 1))
    raise FetchError(f"GET {url} failed: {last}")


def git_blob_sha(content: bytes) -> str:
    """The SHA-1 Git assigns to a blob with this content."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


# -- download --------------------------------------------------------------


def _subtree_sha(repo: str, tree_sha: str, path: str) -> str:
    """Walk ``path`` down from a tree and return the SHA of the final directory."""
    for part in path.split("/"):
        tree = json.loads(_github_get(f"repos/{repo}/git/trees/{tree_sha}"))
        match = [e for e in tree["tree"] if e["path"] == part and e["type"] == "tree"]
        if not match:
            raise FetchError(f"directory {part!r} not found in {repo}")
        tree_sha = match[0]["sha"]
    return tree_sha


def download_swebench_verified(raw_dir: Path, *, ref: str = "main", refresh: bool = False) -> dict[str, Any]:
    """Download result and metadata files into ``raw_dir`` and return the manifest.

    Files already cached with the right Git blob SHA are not downloaded again.
    With a cached manifest and ``refresh=False`` nothing is fetched at all, so
    a rebuild from cache works offline and stays pinned to the cached commit
    (unless ``ref`` is a full commit SHA different from the cached one).
    """
    source_dir = raw_dir / "swe-bench-experiments"
    manifest_path = source_dir / "MANIFEST.json"
    if manifest_path.exists() and not refresh:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pinned_elsewhere = len(ref) == 40 and ref != manifest["commit_sha"]
        if not pinned_elsewhere and all((source_dir / path).exists() for path in manifest["files"]):
            return manifest

    commit = json.loads(_github_get(f"repos/{EXPERIMENTS_REPO}/commits/{ref}"))
    commit_sha = commit["sha"]
    root_tree = commit["commit"]["tree"]["sha"]
    verified_tree = _subtree_sha(EXPERIMENTS_REPO, root_tree, VERIFIED_PATH)
    listing = json.loads(
        _github_get(f"repos/{EXPERIMENTS_REPO}/git/trees/{verified_tree}?recursive=1")
    )
    if listing.get("truncated"):
        raise FetchError("GitHub truncated the tree listing; the fetch would be incomplete")

    submissions = sorted(
        e["path"] for e in listing["tree"] if e["type"] == "tree" and "/" not in e["path"]
    )
    files = {
        e["path"]: e["sha"]
        for e in listing["tree"]
        if e["type"] == "blob" and e["path"].split("/", 1)[-1] in _WANTED
    }

    def fetch(item: tuple[str, str]) -> None:
        path, sha = item
        target = source_dir / path
        if target.exists() and git_blob_sha(target.read_bytes()) == sha:
            return
        content = _github_get(f"repos/{EXPERIMENTS_REPO}/git/blobs/{sha}", raw=True)
        if git_blob_sha(content) != sha:
            raise FetchError(f"{path}: downloaded content does not match blob {sha}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files.items()))

    manifest = {
        "repository": EXPERIMENTS_REPO,
        "path": VERIFIED_PATH,
        "commit_sha": commit_sha,
        "commit_date": commit["commit"]["committer"]["date"],
        "tree_sha": verified_tree,
        "submissions": submissions,
        "files": files,
    }
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def download_verified_task_index(raw_dir: Path, *, refresh: bool = False) -> dict[str, Any]:
    """Fetch the 500 task ids (plus repo, date and difficulty label) from Hugging Face."""
    path = raw_dir / "swebench_verified_instances.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    info = _http_json(f"https://huggingface.co/api/datasets/{HF_DATASET}")
    rows: list[dict[str, Any]] = []
    for offset in range(0, N_VERIFIED_TASKS, 100):
        query = urllib.parse.urlencode(
            {"dataset": HF_DATASET, "config": "default", "split": "test", "offset": offset, "length": 100}
        )
        page = _http_json(f"https://datasets-server.huggingface.co/rows?{query}")
        for entry in page["rows"]:
            row = entry["row"]
            rows.append(
                {
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "created_at": row["created_at"],
                    "difficulty": row.get("difficulty"),
                }
            )
    index = {
        "dataset": HF_DATASET,
        "revision": info.get("sha"),
        "last_modified": info.get("lastModified"),
        "instances": sorted(rows, key=lambda r: r["instance_id"]),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return index


# -- parsing ---------------------------------------------------------------


def parse_results_json(data: Any) -> set[str]:
    """Resolved ids from a ``results/results.json`` (older submission format)."""
    if not isinstance(data, dict) or "resolved" not in data:
        raise ValueError("no 'resolved' key")
    resolved = data["resolved"]
    if isinstance(resolved, dict):
        return {str(k) for k, v in resolved.items() if v}
    if isinstance(resolved, list) and all(isinstance(x, str) for x in resolved):
        return set(resolved)
    raise ValueError("'resolved' is neither a list of ids nor a mapping")


def parse_per_instance_details(data: Any) -> tuple[set[str], set[str]]:
    """``(resolved ids, all ids)`` from a ``per_instance_details.json`` (newer format)."""
    if not isinstance(data, dict) or not data:
        raise ValueError("not a non-empty mapping")
    resolved: set[str] = set()
    for instance, detail in data.items():
        if not isinstance(detail, dict) or not isinstance(detail.get("resolved"), bool):
            raise ValueError(f"{instance}: no boolean 'resolved' field")
        if detail["resolved"]:
            resolved.add(str(instance))
    return resolved, {str(k) for k in data}


def _load_metadata(directory: Path) -> tuple[dict[str, Any], str]:
    for name in ("metadata.yaml", "metadata.yml"):
        path = directory / name
        if path.exists():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                return {}, f"unparseable {name}: {type(exc).__name__}"
            return (loaded, "") if isinstance(loaded, dict) else ({}, f"{name} is not a mapping")
    return {}, "no metadata file"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value)


@dataclass(frozen=True)
class SwebenchVerified:
    """The built matrix with per-submission metadata and provenance."""

    matrix: ResultsMatrix
    submissions: pd.DataFrame
    provenance: dict[str, Any]


def build_swebench_verified(raw_dir: Path, manifest: dict[str, Any], task_index: dict[str, Any]) -> SwebenchVerified:
    """Turn the cached files into a submissions x tasks matrix.

    A task a submission does not list as resolved counts as a failure, which
    is how the leaderboard scores it. Submissions with no per-task results,
    with results that cannot be parsed, or whose per-task results contradict
    the score in their own metadata by more than ``MAX_SCORE_GAP_PCT`` points
    are left out of the matrix and recorded with the reason.
    """
    source_dir = raw_dir / "swe-bench-experiments"
    tasks = [row["instance_id"] for row in task_index["instances"]]
    task_set = set(tasks)
    if len(tasks) != N_VERIFIED_TASKS or len(task_set) != N_VERIFIED_TASKS:
        raise FetchError(f"expected {N_VERIFIED_TASKS} distinct task ids, got {len(task_set)}")

    resolved_by_submission: dict[str, set[str]] = {}
    records: list[dict[str, Any]] = []
    for name in manifest["submissions"]:
        directory = source_dir / name
        meta, meta_problem = _load_metadata(directory)
        info = meta.get("info") or {}
        tags = meta.get("tags") or {}
        record: dict[str, Any] = {
            "submission": name,
            "date": name[:8] if name[:8].isdigit() else "",
            "name": _as_text(info.get("name")),
            "agent": _as_text(tags.get("agent")),
            "model": _as_text(tags.get("model")),
            "model_org": _as_text(tags.get("model_org")),
            "org": _as_text(tags.get("org")),
            "os_model": _as_text(tags.get("os_model")),
            "os_system": _as_text(tags.get("os_system")),
            "checked": _as_text(tags.get("checked")),
            "attempts": _as_text((tags.get("system") or {}).get("attempts"))
            if isinstance(tags.get("system"), dict)
            else "",
            "model_release_date": _as_text(info.get("model_release_date")),
            "reported_resolved_pct": info.get("resolved"),
            "metadata_problem": meta_problem,
        }

        sources: dict[str, set[str]] = {}
        problems: list[str] = []
        n_listed: int | None = None
        n_no_generation: int | None = None
        results_path = directory / "results" / "results.json"
        details_path = directory / "per_instance_details.json"
        if results_path.exists():
            try:
                raw_results = json.loads(results_path.read_text(encoding="utf-8"))
                sources["results.json"] = parse_results_json(raw_results)
                # Tasks for which the submission produced no patch at all. The
                # leaderboard scores them as failures, and so does the matrix.
                no_generation = raw_results.get("no_generation")
                if isinstance(no_generation, list):
                    n_no_generation = len(set(map(str, no_generation)) & task_set)
            except (ValueError, json.JSONDecodeError) as exc:
                problems.append(f"results.json: {exc}")
        if details_path.exists():
            try:
                solved, listed = parse_per_instance_details(json.loads(details_path.read_text(encoding="utf-8")))
                sources["per_instance_details.json"] = solved
                n_listed = len(listed & task_set)
            except (ValueError, json.JSONDecodeError) as exc:
                problems.append(f"per_instance_details.json: {exc}")

        if not sources:
            record.update(
                included=False,
                source_format="",
                exclusion_reason="; ".join(problems) if problems else "no per-task results file",
            )
            records.append(record)
            continue

        # Both formats present: results.json is what the leaderboard was built
        # from historically, so it wins; any disagreement is recorded.
        source_format = "results.json" if "results.json" in sources else "per_instance_details.json"
        resolved = sources[source_format]
        unknown = resolved - task_set
        resolved = resolved & task_set
        if len(sources) == 2 and sources["results.json"] != sources["per_instance_details.json"]:
            problems.append(
                "formats disagree on "
                f"{len(sources['results.json'] ^ sources['per_instance_details.json'])} tasks"
            )
        computed_pct = 100.0 * len(resolved) / N_VERIFIED_TASKS
        reported = record["reported_resolved_pct"]
        gap = float(reported) - computed_pct if isinstance(reported, (int, float)) else None
        consistent = gap is None or abs(gap) <= MAX_SCORE_GAP_PCT
        if consistent:
            resolved_by_submission[name] = resolved
        record.update(
            included=consistent,
            source_format=source_format,
            exclusion_reason=""
            if consistent
            else f"per-task file gives {computed_pct:.1f}% resolved but metadata reports {reported}%",
            n_resolved=len(resolved),
            resolved_pct=computed_pct,
            n_unknown_ids=len(unknown),
            n_instances_listed=n_listed,
            n_no_generation=n_no_generation,
            reported_minus_computed=gap,
            parse_notes="; ".join(problems),
        )
        records.append(record)

    matrix = ResultsMatrix.from_resolved(resolved_by_submission, tasks)
    submissions = pd.DataFrame(records).set_index("submission")
    included = submissions[submissions["included"]]
    excluded = submissions[~submissions["included"]]
    gaps = included["reported_minus_computed"].dropna().astype(float)
    no_generation_counts = pd.to_numeric(included["n_no_generation"], errors="coerce")
    provenance = {
        "source_repository": f"https://github.com/{manifest['repository']}",
        "source_path": manifest["path"],
        "source_commit_sha": manifest["commit_sha"],
        "source_commit_date": manifest["commit_date"],
        "task_list_dataset": task_index["dataset"],
        "task_list_revision": task_index["revision"],
        "n_submission_directories": len(manifest["submissions"]),
        "n_source_files": len(manifest["files"]),
        "n_included": int(len(included)),
        "n_excluded": int(len(excluded)),
        "n_excluded_no_per_task_results": int((excluded["source_format"] == "").sum()),
        "n_excluded_inconsistent_with_reported_score": int((excluded["source_format"] != "").sum()),
        "excluded": {name: row["exclusion_reason"] for name, row in excluded.iterrows()},
        "n_from_results_json": int((included["source_format"] == "results.json").sum()),
        "n_from_per_instance_details": int(
            (included["source_format"] == "per_instance_details.json").sum()
        ),
        "n_with_unknown_ids": int((included["n_unknown_ids"] > 0).sum()),
        "n_with_parse_notes": int((included["parse_notes"] != "").sum()),
        "n_reporting_no_generation_count": int(no_generation_counts.notna().sum()),
        "n_with_no_generation_on_10_or_more_tasks": int((no_generation_counts >= 10).sum()),
        "max_no_generation": int(no_generation_counts.max()) if no_generation_counts.notna().any() else 0,
        "n_with_reported_score": int(len(gaps)),
        "n_included_reported_score_differs_over_0.1pt": int((gaps.abs() > 0.1).sum()),
        "max_score_gap_pct_allowed": MAX_SCORE_GAP_PCT,
        "n_tasks": len(tasks),
        "n_metadata_problems": int((submissions["metadata_problem"] != "").sum()),
    }
    return SwebenchVerified(matrix=matrix, submissions=submissions, provenance=provenance)


def fetch_swebench_verified(
    data_dir: str | Path = "data",
    *,
    ref: str = "main",
    refresh: bool = False,
) -> SwebenchVerified:
    """Download (or reuse the cache in ``data_dir/raw``), build and save the matrix.

    Writes ``swebench_verified_matrix.csv``, ``swebench_verified_submissions.csv``
    and ``swebench_verified_build.json`` into ``data_dir``.
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    manifest = download_swebench_verified(raw_dir, ref=ref, refresh=refresh)
    task_index = download_verified_task_index(raw_dir, refresh=refresh)
    built = build_swebench_verified(raw_dir, manifest, task_index)
    data_dir.mkdir(parents=True, exist_ok=True)
    built.matrix.to_csv(data_dir / "swebench_verified_matrix.csv")
    built.submissions.to_csv(data_dir / "swebench_verified_submissions.csv", lineterminator="\n")
    (data_dir / "swebench_verified_build.json").write_text(
        json.dumps(built.provenance, indent=2) + "\n", encoding="utf-8"
    )
    return built


def load_verified_task_index(data_dir: str | Path = "data") -> pd.DataFrame:
    """The cached task index (repo, creation date, difficulty label) as a frame."""
    path = Path(data_dir) / "raw" / "swebench_verified_instances.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(index["instances"]).set_index("instance_id")


__all__ = [
    "SwebenchVerified",
    "FetchError",
    "fetch_swebench_verified",
    "build_swebench_verified",
    "download_swebench_verified",
    "download_verified_task_index",
    "load_verified_task_index",
    "parse_results_json",
    "parse_per_instance_details",
    "git_blob_sha",
]
