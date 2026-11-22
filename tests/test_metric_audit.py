"""Independent checks for auditing archived rates and sampled trajectory replay."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_script = Path(__file__).resolve().parents[1] / "scripts" / "audit_baseline_metrics.py"
_spec = importlib.util.spec_from_file_location("audit_baseline_metrics", _script)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def archived_rows():
    # Setting A has two maps; setting B has one. A map with one undefined
    # replicate must not lose its weight relative to the other maps.
    specifications = (
        ("a0", "lower", 0, 4, 4), ("a0", "lower", 1, 0, 4),
        ("a1", "lower", 0, 0, 10), ("a1", "lower", 1, 0, 0),
        ("b0", "higher", 0, 8, 8), ("b0", "higher", 1, 8, 8),
    )
    return [
        {
            "map_id": map_id, "version": "v2", "episode_sha256": f"hash-{map_id}",
            "method": "random_walk", "replicate": replicate, "alpha": 1,
            "exploration_profile": exploration, "dag_profile": "easier", "steps": 10,
            "exploration_errors": errors, "exploration_eligible_steps": eligible,
            "exploration_error_rate": errors / eligible if eligible else None,
            "exploitation_errors": 0, "exploitation_eligible_steps": 0,
            "exploitation_error_rate": None, "union_errors": errors,
            "union_error_rate": errors / 10,
        }
        for map_id, exploration, replicate, errors, eligible in specifications
    ]


def summary_tables():
    by_setting = []
    for exploration, exploration_mean, union_mean in (("lower", 0.25, 0.1), ("higher", 1.0, 0.8)):
        for metric, mean in (("exploration_error_rate", exploration_mean),
                             ("exploitation_error_rate", None), ("union_error_rate", union_mean)):
            by_setting.append(dict(method="random_walk", alpha=1, exploration_profile=exploration,
                                   dag_profile="easier", metric=metric, mean=mean))
    summary = [dict(method="random_walk", alpha=1, metric=metric, mean=mean)
               for metric, mean in (("exploration_error_rate", 0.625),
                                    ("exploitation_error_rate", None), ("union_error_rate", 0.45))]
    return {"by_setting.csv": by_setting, "summary.csv": summary}


def write_archive(directory, *, rows=None, tables=None):
    rows = archived_rows() if rows is None else rows
    tables = summary_tables() if tables is None else tables
    results = directory / "results.jsonl.gz"
    with gzip.open(results, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for name, table in tables.items():
        with (directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    result_digest = hashlib.sha256(results.read_bytes()).hexdigest()
    manifest = {
        "status": "complete", "row_count": len(rows),
        "map_count": len({row["map_id"] for row in rows}),
        "results_sha256": result_digest, "code_sha256": "synthetic-evaluator",
        "config": {"alphas": [1], "methods": ["random_walk"], "stochastic_replicates": 2,
                   "seed": 1, "max_search_states": 100},
    }
    (directory / "run.json").write_text(json.dumps(manifest))
    (directory / "summary.json").write_text(json.dumps({
        "source_results_sha256": result_digest,
        "table_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in tables},
    }))


def test_archive_checks_map_then_setting_means_and_preserves_null(tmp_path):
    write_archive(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    _, references, report = audit.inspect_archive(tmp_path, {"a0": {}}, tmp_path / "ref")
    assert report["rows_checked"] == 6
    assert report["maps_checked"] == 3
    assert report["summary_means_checked"] == 9
    assert report["summary_mean_mismatches"] == []
    assert set(references) == {"a0"}
    with gzip.open(references["a0"], "rt") as stream:
        reference_rows = json.load(stream)
    assert len(reference_rows) == 2
    assert all(row["exploitation_error_rate"] is None for row in reference_rows)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}


@pytest.mark.parametrize("custom_results", [False, True])
def test_cli_selects_result_archives_separately_from_benchmark_data(tmp_path, monkeypatch, custom_results):
    data_root = tmp_path / "benchmark-repository"
    canonical_results = data_root / "analysis" / "baselines"
    custom_results_root = tmp_path / "outputs" / "baselines"
    selected_root = custom_results_root if custom_results else canonical_results
    unused_root = canonical_results if custom_results else custom_results_root
    for root in (selected_root, unused_root):
        directory = root / "v2"
        directory.mkdir(parents=True)
        write_archive(directory)
    # A silently ignored --results-root (or an incorrect new default) must fail.
    (unused_root / "v2" / "results.jsonl.gz").write_bytes(b"wrong archive")
    rows = archived_rows()
    records = list({row["map_id"]: {
        "map_id": row["map_id"], "episode_sha256": row["episode_sha256"],
        "generation": {"exploration_profile": row["exploration_profile"], "dag_profile": row["dag_profile"]},
    } for row in rows}.values())
    data_lookups = []

    def load_records(version, *, root):
        data_lookups.append((version, root))
        return iter(records)

    def evaluate(record, method, *, replicate, **kwargs):
        return [row for row in rows if row["map_id"] == record["map_id"]
                and row["method"] == method and row["replicate"] == replicate]

    monkeypatch.setattr(audit, "load_benchmark", load_records)
    monkeypatch.setattr(audit, "evaluate_record", evaluate)
    monkeypatch.setattr(audit, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(audit, "verify_metric_dependency", lambda: {})
    monkeypatch.setattr(audit, "evaluator_source_sha256", lambda: "synthetic-evaluator")
    report_path = tmp_path / "reports" / "audit.json"
    arguments = [str(_script), "--root", str(data_root), "--versions", "v2", "--output", str(report_path)]
    if custom_results:
        arguments += ["--results-root", str(custom_results_root)]
    monkeypatch.setattr(sys, "argv", arguments)
    assert audit.main() == 0
    report = json.loads(report_path.read_text())
    assert report["valid"] is True
    assert data_lookups == [("v2", data_root)]
    assert report["versions"]["v2"]["rows_checked"] == 6
    assert report["versions"]["v2"]["replay_maps"] == 2
    assert {row["map_id"] for row in report["versions"]["v2"]["replays"]} == {"a0", "b0"}
    assert report["versions"]["v2"]["result_sha256"] == audit.file_sha256(selected_root / "v2" / "results.jsonl.gz")


def test_archive_requires_every_inventory_selected_replay_map(tmp_path):
    write_archive(tmp_path)
    with pytest.raises(ValueError, match="Replay maps missing from archive"):
        audit.inspect_archive(tmp_path, {"not-in-archive": {}}, tmp_path / "ref")


@pytest.mark.parametrize("change", ["rate", "denominator", "zero_as_zero", "nonzero_without_eligible"])
def test_archive_rejects_incorrect_counts_rates_and_undefined_values(tmp_path, change):
    rows = archived_rows()
    if change == "rate":
        rows[0]["exploration_error_rate"] = 0.4
    elif change == "denominator":
        rows[0]["exploration_eligible_steps"] = 10
    elif change == "zero_as_zero":
        rows[3]["exploration_error_rate"] = 0.0
    else:
        rows[3]["exploration_errors"] = 1
    write_archive(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="Invalid archived rate/count"):
        audit.inspect_archive(tmp_path, {}, tmp_path / "ref")


@pytest.mark.parametrize("table", ["by_setting.csv", "summary.csv"])
@pytest.mark.parametrize("change", ["missing_defined_mean", "zero_for_undefined_mean"])
def test_archive_distinguishes_missing_csv_values_from_zero(tmp_path, table, change):
    tables = summary_tables()
    metric = "exploration_error_rate" if change == "missing_defined_mean" else "exploitation_error_rate"
    row = next(row for row in tables[table] if row["metric"] == metric)
    row["mean"] = None if change == "missing_defined_mean" else 0.0
    write_archive(tmp_path, tables=tables)
    _, _, report = audit.inspect_archive(tmp_path, {}, tmp_path / "ref")
    assert len(report["summary_mean_mismatches"]) == 1
    mismatch = report["summary_mean_mismatches"][0]
    assert mismatch["table"] == table
    assert mismatch["metric"] == metric
    if change == "missing_defined_mean":
        assert mismatch["saved"] is None
        assert mismatch["recomputed"] > 0
    else:
        assert mismatch["saved"] == 0.0
        assert mismatch["recomputed"] is None


@pytest.mark.parametrize("pooled_mean", [0.5, 0.6])
def test_archive_rejects_map_weighted_or_replicate_weighted_overall_mean(tmp_path, pooled_mean):
    tables = summary_tables()
    tables["summary.csv"][0]["mean"] = pooled_mean
    write_archive(tmp_path, tables=tables)
    _, _, report = audit.inspect_archive(tmp_path, {}, tmp_path / "ref")
    assert len(report["summary_mean_mismatches"]) == 1
    assert report["summary_mean_mismatches"][0]["recomputed"] == pytest.approx(0.625)


@pytest.mark.parametrize("table", ["by_setting.csv", "summary.csv"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "unexpected"])
def test_archive_requires_complete_unique_error_rate_summary_keys(tmp_path, table, change):
    tables = summary_tables()
    if change == "missing":
        tables[table].pop(0)
    elif change == "duplicate":
        tables[table].append(deepcopy(tables[table][0]))
    else:
        tables[table][0]["method"] = "other_method"
    write_archive(tmp_path, tables=tables)
    with pytest.raises(ValueError, match="summary key"):
        audit.inspect_archive(tmp_path, {}, tmp_path / "ref")


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_archive_requires_every_configured_replicate(tmp_path, change):
    rows = archived_rows()
    if change == "missing":
        rows.pop(1)
    else:
        rows[1] = deepcopy(rows[0])
    write_archive(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="archived result keys"):
        audit.inspect_archive(tmp_path, {}, tmp_path / "ref")


def test_replay_detects_consistent_but_wrong_denominator_and_ignores_timing(tmp_path, monkeypatch):
    replayed = deepcopy(archived_rows()[0])
    replayed["planning_cpu_seconds"] = 20.0
    saved = deepcopy(replayed)
    saved["planning_cpu_seconds"] = 10.0
    # The saved ratio is internally consistent, but uses six extra steps.
    saved["exploration_eligible_steps"] = 10
    saved["exploration_error_rate"] = 0.4
    monkeypatch.setattr(audit, "evaluate_record", lambda *args, **kwargs: [replayed])
    config = {"alphas": [1], "seed": 1, "max_search_states": 100}
    reference_path = tmp_path / "reference.json.gz"
    with gzip.open(reference_path, "wt") as stream:
        json.dump([saved], stream)
    result = audit.replay_map({"map_id": "a0", "episode_sha256": "hash-a0"}, config, reference_path)
    assert result["rows_checked"] == 1
    assert {mismatch["field"] for mismatch in result["mismatches"]} == {
        "exploration_eligible_steps", "exploration_error_rate",
    }


@pytest.mark.parametrize("change", ["missing", "duplicate", "unexpected"])
def test_replay_requires_every_archived_cutoff_once(tmp_path, monkeypatch, change):
    first = deepcopy(archived_rows()[0])
    second = {**first, "alpha": 2}
    saved = [first, second]
    if change == "missing":
        replayed = [first]
    elif change == "duplicate":
        replayed = [first, first, second]
    else:
        replayed = [first, {**second, "alpha": 3}]
    monkeypatch.setattr(audit, "evaluate_record", lambda *args, **kwargs: replayed)
    reference_path = tmp_path / "reference.json.gz"
    with gzip.open(reference_path, "wt") as stream:
        json.dump(saved, stream)
    config = {"alphas": [1, 2], "seed": 1, "max_search_states": 100}
    with pytest.raises(ValueError):
        audit.replay_map({"map_id": "a0", "episode_sha256": "hash-a0"}, config, reference_path)
