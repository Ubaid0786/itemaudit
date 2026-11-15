from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import simulate_2pl
from itemaudit import ResultsMatrix, report
from itemaudit.cli import main
from itemaudit.report import audit, markdown_table, render_markdown, solve_rate_histogram, to_jsonable


@pytest.fixture
def results_csv(tmp_path: Path) -> Path:
    matrix, *_ = simulate_2pl(30, 40, seed=12)
    values = matrix.values.copy()
    values[:, 0] = 1.0  # solved by all
    values[:, 1] = 0.0  # solved by none
    values[:, 2] = 1 - values[:, 2]  # reverse keyed
    values[3, 5] = np.nan
    path = tmp_path / "results.csv"
    ResultsMatrix(matrix.systems, matrix.tasks, values).to_csv(path)
    return path


def test_report_writes_markdown_and_strict_json(tmp_path: Path, results_csv: Path) -> None:
    from itemaudit import load_results

    out = tmp_path / "out"
    result = report(load_results(results_csv), out, top=10, n_boot=200)
    text = (out / "report.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    loaded = json.loads(text)
    assert loaded == result
    assert loaded["summary"]["n_systems"] == 30 and loaded["summary"]["n_tasks"] == 40
    assert loaded["summary"]["n_missing_cells"] == 1
    assert loaded["summary"]["n_tasks_solved_by_all"] == 1
    assert loaded["summary"]["n_tasks_solved_by_none"] == 1
    assert loaded["summary"]["n_tasks_negative_discrimination"] >= 1
    assert loaded["summary"]["kr20"] is None  # undefined with a missing cell
    assert loaded["leaderboard"]["n_systems"] == 10
    assert len(loaded["items"]) == 40 and len(loaded["systems"]) == 30
    assert {"irt_discrimination", "discrimination", "flag"} <= set(loaded["items"][0])
    flagged = {row["task"] for row in loaded["items"] if row["flag"] == "negative_discrimination"}
    assert "t2" in flagged

    markdown = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## Summary", "## Leaderboard", "### Adjacent ranks", "## Suspect tasks", "## Item response model"):
        assert heading in markdown
    assert "| t2 |" in markdown


def test_report_without_irt_and_with_two_systems(tmp_path: Path) -> None:
    values = np.array([[1, 1, 0, 1], [1, 0, 0, 0]], dtype=float)
    matrix = ResultsMatrix(("a", "b"), tuple(f"t{j}" for j in range(4)), values)
    result = report(matrix, tmp_path, name="tiny", fit_irt=False, n_boot=50)
    assert "irt" not in result
    assert result["leaderboard"]["n_adjacent_pairs"] == 1
    assert (tmp_path / "tiny.md").exists() and (tmp_path / "tiny.json").exists()
    with_irt = audit(matrix, n_boot=50)
    assert "irt" in with_irt
    assert "No task has negative discrimination" in render_markdown(with_irt)


def test_to_jsonable_handles_numpy_and_non_finite_values() -> None:
    value = {"a": np.float64(1.5), "b": np.nan, "c": [np.int64(2), np.inf], "d": np.array([1.0, np.nan]), 5: np.True_}
    assert to_jsonable(value) == {"a": 1.5, "b": None, "c": [2, None], "d": [1.0, None], "5": True}


def test_solve_rate_histogram_accounts_for_every_task() -> None:
    rate = np.array([0.0, 0.0, 0.05, 0.1, 0.3, 0.30000000000000004, 0.55, 0.999, 1.0, np.nan])
    rows = solve_rate_histogram(rate)
    counts = {row["band"]: row["n_tasks"] for row in rows}
    assert sum(counts.values()) == 9
    assert counts["exactly 0"] == 2 and counts["exactly 1"] == 1
    assert counts["0.0 to 0.1"] == 2  # .05 and the boundary value .1
    assert counts["0.2 to 0.3"] == 2  # both spellings of .3
    assert counts["0.9 to 1.0 (below 1)"] == 1


def test_markdown_table_formats_and_escapes() -> None:
    import pandas as pd

    frame = pd.DataFrame({"name": ["a|b", "c"], "value": [0.12345, np.nan]})
    table = markdown_table(frame, {"value": ".2f"})
    lines = table.splitlines()
    assert lines[0] == "| name | value |"
    assert lines[2] == "| a\\|b | 0.12 |"
    assert lines[3] == "| c |  |"


def test_cli_report(tmp_path: Path, results_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli"
    code = main(["report", str(results_csv), "--out", str(out), "--name", "audit", "--top", "5"])
    assert code == 0
    assert (out / "audit.md").exists() and (out / "audit.json").exists()
    printed = capsys.readouterr().out
    assert "30 systems x 40 tasks" in printed
    assert "adjacent ranks indistinguishable" in printed


def test_cli_compare(results_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", str(results_csv), "s0", "s1"]) == 0
    assert "McNemar exact p" in capsys.readouterr().out
    assert main(["compare", str(results_csv), "s0", "s1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["system_a"] == "s0" and payload["n_tasks"] == 40


def test_cli_errors_are_reported_not_raised(results_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", str(results_csv), "s0", "nobody"]) == 2
    assert "nobody" in capsys.readouterr().err
    assert main(["report", str(results_csv.parent / "missing.csv")]) == 2


def test_cli_requires_a_known_command() -> None:
    with pytest.raises(SystemExit):
        main(["fetch", "not-a-dataset"])
    with pytest.raises(SystemExit):
        main([])
