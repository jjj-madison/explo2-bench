"""Independent aggregation and result-integrity checks for the release reports."""

from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from explo2_bench.analysis.baseline_results import summarize_results
from explo2_bench.data import canonical_sha256


def synthetic_rows():
    rows = []
    for index, outcomes in enumerate(((True, False), (False, False), (True, True), (True, True))):
        for replicate, success in enumerate(outcomes):
            row = dict(map_id=f"m{index}", version="v2", episode_sha256=f"hash{index}",
                       exploration_profile="lower" if index < 2 else "higher", dag_profile="easier",
                       method="random_walk", alpha=1, replicate=replicate, success=success,
                       steps=4 if success else 10, goal_activation_step=4 if success else None,
                       task_nodes=2, activated_count=2 if success else 1, discovered_count=2,
                       coverage=0.8, invalid_actions=0, planning_cpu_seconds=0.01,
                       planning_wall_seconds=0.02, planning_cpu_seconds_per_action=0.001,
                       planner_diagnostics={}, exploration_error_rate=0.5,
                       exploitation_error_rate=None, union_error_rate=0.5)
            rows.append(row)
    return rows


def write_run(tmp_path, rows=None):
    rows = synthetic_rows() if rows is None else rows
    path = tmp_path / "results.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    selection = [(f"m{index}", f"hash{index}") for index in range(4)]
    manifest = dict(status="complete", identity="synthetic-test", version="v2", map_count=4, row_count=8,
                    selection_sha256=canonical_sha256(selection),
                    results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    config=dict(methods=["random_walk"], alphas=[1], stochastic_replicates=2,
                                max_search_states=10, seed=1))
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    return path


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def test_replicates_are_not_independent_maps_and_conditional_success_is_counted(tmp_path):
    result = summarize_results(write_run(tmp_path), figures=False)
    overall = {row["metric"]: row for row in read_csv(tmp_path / "summary.csv")}
    # Cell A: per-map success means 0.5 and 0 -> mean .25 and SE .25.
    # Cell B: per-map success means 1 and 1 -> mean 1 and SE 0.
    # Equal cell mean .625, stratified SE sqrt(.25^2 + 0^2)/2=.125.
    assert float(overall["success"]["mean"]) == pytest.approx(0.625)
    assert float(overall["success"]["standard_error"]) == pytest.approx(0.125)
    assert overall["success"]["maps"] == "4"
    assert overall["success"]["runs"] == "8"
    assert overall["successful_actions"]["maps"] == "3"
    assert overall["successful_actions"]["runs"] == "5"
    assert float(overall["successful_actions"]["mean"]) == 4
    assert overall["exploitation_error_rate"]["maps"] == "0"
    assert overall["exploitation_error_rate"]["mean"] == ""
    assert result["result_rows"] == 8
    assert "Successful maps / runs" in (tmp_path / "README.md").read_text()


def test_summary_rejects_modified_file_before_using_rows(tmp_path):
    path = write_run(tmp_path)
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        summarize_results(path, figures=False)


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_method", "wrong_replicate", "mixed_hash", "wrong_version"])
def test_summary_checks_expected_keys_and_identity_even_with_matching_file_hash(tmp_path, change):
    rows = deepcopy(synthetic_rows())
    if change == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif change == "missing":
        rows.pop(1)
    elif change == "wrong_method":
        rows[0]["method"] = "online_planner"
    elif change == "wrong_replicate":
        rows[0]["replicate"] = 9
    elif change == "mixed_hash":
        rows[1]["episode_sha256"] = "other"
    else:
        rows[0]["version"] = "v3"
    with pytest.raises(ValueError):
        summarize_results(write_run(tmp_path, rows), figures=False)


@pytest.mark.parametrize("field", ["row_count", "map_count", "selection_sha256"])
def test_summary_checks_complete_manifest_counts_and_selection(tmp_path, field):
    path = write_run(tmp_path)
    manifest_path = tmp_path / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = "other" if field == "selection_sha256" else 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        summarize_results(path, figures=False)


@pytest.mark.parametrize("valid,matching,shown", [(True, True, True), (False, True, False), (True, False, False)])
def test_report_only_attaches_a_passed_audit_for_its_exact_results(tmp_path, valid, matching, shown):
    directory = tmp_path / "campaign" / "v2"
    directory.mkdir(parents=True)
    path = write_run(directory)
    manifest = json.loads((directory / "run.json").read_text())
    audit = {
        "valid": valid,
        "versions": {"v2": {
            "result_sha256": manifest["results_sha256"] if matching else "another-campaign",
            "rows_checked": 8, "summary_means_checked": 6,
            "maps_checked": 4, "replay_maps": 1,
            "replay_trajectories": 2, "replay_rows": 2,
        }},
    }
    (directory.parent / "metric-audit.json").write_text(json.dumps(audit))

    summarize_results(path, figures=False)
    report = (directory / "README.md").read_text()

    assert ("[upstream metric audit](../metric-audit.json)" in report) is shown
    if shown:
        assert "**8 recorded rows**" in report
        assert "**2 trajectories** on **1 selected map**" in report
        assert "remaining trajectories have not been replayed" in report
        assert "Timing fields are excluded" in report
