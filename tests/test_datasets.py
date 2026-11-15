"""The SWE-bench loader, tested on a fake cache. Nothing here touches the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from itemaudit import datasets
from itemaudit.datasets import (
    build_swebench_verified,
    fetch_swebench_verified,
    git_blob_sha,
    parse_per_instance_details,
    parse_results_json,
)

TASKS = [f"repo__proj-{i:04d}" for i in range(datasets.N_VERIFIED_TASKS)]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _metadata(name: str, resolved: float | None = None, checked: object = True) -> str:
    info: dict[str, object] = {"name": name}
    if resolved is not None:
        info["resolved"] = resolved
    meta = {
        "info": info,
        "tags": {"agent": "Agent X", "model": ["model-a", "model-b"], "checked": checked, "system": {"attempts": "1"}},
    }
    return yaml.safe_dump(meta)


@pytest.fixture
def fake_cache(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    raw = tmp_path / "raw"
    source = raw / "swe-bench-experiments"

    # 1. Old format: list of resolved ids, one of them not a Verified task.
    _write(source / "20240101_old" / "metadata.yaml", _metadata("Old"))
    _write(
        source / "20240101_old" / "results" / "results.json",
        json.dumps({"resolved": TASKS[:100] + ["other__repo-1"], "no_generation": TASKS[400:]}),
    )
    # 2. New format, consistent with its reported score (250 / 500 = 50.0).
    details = {t: {"resolved": i < 250, "cost": 0.1} for i, t in enumerate(TASKS)}
    _write(source / "20250101_new" / "metadata.yml", _metadata("New", 50.0, checked=None))
    _write(source / "20250101_new" / "per_instance_details.json", json.dumps(details))
    # 3. Metadata only.
    _write(source / "20250102_meta_only" / "metadata.yaml", _metadata("Meta only", 61.2))
    # 4. Per-task file contradicts the reported score.
    zeros = {t: {"resolved": False} for t in TASKS}
    _write(source / "20250103_inconsistent" / "metadata.yaml", _metadata("Inconsistent", 70.0))
    _write(source / "20250103_inconsistent" / "per_instance_details.json", json.dumps(zeros))
    # 5. Broken JSON.
    _write(source / "20250104_broken" / "metadata.yaml", _metadata("Broken"))
    _write(source / "20250104_broken" / "results" / "results.json", "{not json")
    # 6. Both formats, agreeing; resolved given as a mapping.
    both = {t: {"resolved": i < 10} for i, t in enumerate(TASKS)}
    _write(source / "20250105_both" / "metadata.yaml", _metadata("Both", 2.0))
    _write(source / "20250105_both" / "results" / "results.json", json.dumps({"resolved": {t: True for t in TASKS[:10]}}))
    _write(source / "20250105_both" / "per_instance_details.json", json.dumps(both))
    # 7. No metadata at all, results missing the key.
    _write(source / "20250106_nokey" / "results" / "results.json", json.dumps({"generated": TASKS[:5]}))

    submissions = sorted(p.name for p in source.iterdir())
    manifest = {
        "repository": "swe-bench/experiments",
        "path": "evaluation/verified",
        "commit_sha": "0" * 40,
        "commit_date": "2026-01-01T00:00:00Z",
        "tree_sha": "1" * 40,
        "submissions": submissions,
        "files": {},
    }
    task_index = {
        "dataset": "example/dataset",
        "revision": "abc",
        "instances": [{"instance_id": t, "repo": "repo/proj", "created_at": "", "difficulty": None} for t in TASKS],
    }
    return raw, manifest, task_index


def test_build_counts_and_reasons(fake_cache: tuple[Path, dict[str, object], dict[str, object]]) -> None:
    raw, manifest, task_index = fake_cache
    built = build_swebench_verified(raw, manifest, task_index)
    p = built.provenance
    assert p["n_submission_directories"] == 7
    assert p["n_included"] == 3 and p["n_excluded"] == 4
    assert p["n_excluded_no_per_task_results"] == 3
    assert p["n_excluded_inconsistent_with_reported_score"] == 1
    assert p["n_from_results_json"] == 2 and p["n_from_per_instance_details"] == 1
    assert p["n_with_unknown_ids"] == 1
    assert p["n_metadata_problems"] == 1
    assert p["source_commit_sha"] == "0" * 40
    reasons = p["excluded"]
    assert reasons["20250102_meta_only"] == "no per-task results file"
    assert "70.0%" in reasons["20250103_inconsistent"] and "0.0%" in reasons["20250103_inconsistent"]
    assert reasons["20250104_broken"].startswith("results.json:")
    assert "no 'resolved' key" in reasons["20250106_nokey"]


def test_build_matrix_content(fake_cache: tuple[Path, dict[str, object], dict[str, object]]) -> None:
    raw, manifest, task_index = fake_cache
    built = build_swebench_verified(raw, manifest, task_index)
    matrix = built.matrix
    assert matrix.systems == ("20240101_old", "20250101_new", "20250105_both")
    assert matrix.tasks == tuple(TASKS)
    assert matrix.observed.all()  # an unlisted task is a failure, not missing
    assert list(matrix.values.sum(axis=1)) == [100, 250, 10]
    row = built.submissions.loc["20240101_old"]
    assert row["n_resolved"] == 100 and row["n_unknown_ids"] == 1
    assert row["n_no_generation"] == 100
    assert built.provenance["n_with_no_generation_on_10_or_more_tasks"] == 1
    assert built.provenance["max_no_generation"] == 100
    assert row["model"] == "model-a; model-b" and row["agent"] == "Agent X"
    assert row["date"] == "20240101" and row["attempts"] == "1"
    assert built.submissions.loc["20250101_new", "reported_minus_computed"] == pytest.approx(0.0)
    assert built.submissions.loc["20250105_both", "source_format"] == "results.json"
    assert built.submissions.loc["20250105_both", "parse_notes"] == ""


def test_disagreeing_formats_are_noted(fake_cache: tuple[Path, dict[str, object], dict[str, object]]) -> None:
    raw, manifest, task_index = fake_cache
    path = raw / "swe-bench-experiments" / "20250105_both" / "per_instance_details.json"
    path.write_text(json.dumps({t: {"resolved": i < 12} for i, t in enumerate(TASKS)}), encoding="utf-8")
    built = build_swebench_verified(raw, manifest, task_index)
    assert built.submissions.loc["20250105_both", "parse_notes"] == "formats disagree on 2 tasks"
    assert built.provenance["n_with_parse_notes"] == 1


def test_build_rejects_wrong_task_count(fake_cache: tuple[Path, dict[str, object], dict[str, object]]) -> None:
    raw, manifest, task_index = fake_cache
    short = dict(task_index, instances=task_index["instances"][:10])  # type: ignore[index]
    with pytest.raises(datasets.FetchError):
        build_swebench_verified(raw, manifest, short)


def test_fetch_uses_the_cache_without_network(
    fake_cache: tuple[Path, dict[str, object], dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, manifest, task_index = fake_cache
    data_dir = raw.parent
    (raw / "swe-bench-experiments" / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (raw / "swebench_verified_instances.json").write_text(json.dumps(task_index), encoding="utf-8")

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(datasets, "_github_get", no_network)
    monkeypatch.setattr(datasets, "_http_json", no_network)
    built = fetch_swebench_verified(data_dir)
    assert built.matrix.n_systems == 3
    from itemaudit import load_results

    again = load_results(data_dir / "swebench_verified_matrix.csv")
    assert again.systems == built.matrix.systems
    assert (again.values == built.matrix.values).all()
    saved = json.loads((data_dir / "swebench_verified_build.json").read_text(encoding="utf-8"))
    assert saved["n_included"] == 3
    assert datasets.load_verified_task_index(data_dir).shape[0] == datasets.N_VERIFIED_TASKS


def test_parse_results_json_variants() -> None:
    assert parse_results_json({"resolved": ["a", "b"]}) == {"a", "b"}
    assert parse_results_json({"resolved": {"a": True, "b": False}}) == {"a"}
    assert parse_results_json({"resolved": []}) == set()
    for bad in ({"generated": []}, ["a"], {"resolved": [1, 2]}, {"resolved": "a"}):
        with pytest.raises(ValueError):
            parse_results_json(bad)


def test_parse_per_instance_details_variants() -> None:
    resolved, listed = parse_per_instance_details({"a": {"resolved": True}, "b": {"resolved": False, "cost": 1}})
    assert resolved == {"a"} and listed == {"a", "b"}
    for bad in ({}, [], {"a": {"cost": 1}}, {"a": {"resolved": "yes"}}, {"a": True}):
        with pytest.raises(ValueError):
            parse_per_instance_details(bad)


def test_git_blob_sha_reference_values() -> None:
    # Well-known Git object ids for the empty blob and for "hello\n".
    assert git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
