#!/usr/bin/env python3
"""Audit archived rates and replay a declared sample with pinned upstream scoring.

Reads the archived results without rewriting them. The default replay selects
the first inventory map in every exploration/task-DAG setting; --all-maps
replays the complete inventory. Timing fields are not deterministic comparisons.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

from explo2_bench.data import canonical_sha256, load_benchmark
from explo2_bench.baselines import STOCHASTIC_BASELINES
from explo2_bench.evaluation import evaluate_record, evaluator_source_sha256
from explo2_bench.upstream import verify_metric_dependency


RATE_FIELDS = {
    "exploration_error_rate": ("exploration_errors", "exploration_eligible_steps"),
    "exploitation_error_rate": ("exploitation_errors", "exploitation_eligible_steps"),
    "union_error_rate": ("union_errors", "steps"),
}
TIMING_FIELDS = {
    "planning_cpu_seconds", "planning_wall_seconds", "planning_cpu_seconds_per_action",
    "full_rollout_scoring_cpu_seconds", "execution_id",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equivalent(left, right):
    if isinstance(left, float) or isinstance(right, float):
        return (isinstance(left, (int, float)) and isinstance(right, (int, float))
                and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12))
    return left == right


def select_records(version, root, maps_per_setting):
    counts = defaultdict(int)
    selected = {}
    for record in load_benchmark(version, root=root):
        setting = (record["generation"]["exploration_profile"], record["generation"]["dag_profile"])
        if maps_per_setting is None or counts[setting] < maps_per_setting:
            selected[record["map_id"]] = record
            counts[setting] += 1
    return selected


def inspect_archive(directory, selected, reference_dir):
    """Check every rate and independently rebuild error-rate summary means."""
    directory = Path(directory)
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "run.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    result_path = directory / "results.jsonl.gz"
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete archived run: {directory}")
    result_hash = file_sha256(result_path)
    if result_hash != manifest["results_sha256"] or result_hash != summary["source_results_sha256"]:
        raise ValueError(f"Archived result checksum mismatch: {directory}")
    for name, digest in summary["table_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Archived summary checksum mismatch: {directory / name}")

    references = {}
    settings = defaultdict(list)
    observed_settings = set()
    expected_row_keys = {(method, alpha, replicate)
                         for method in manifest["config"]["methods"]
                         for alpha in manifest["config"]["alphas"]
                         for replicate in range(manifest["config"]["stochastic_replicates"]
                                                if method in STOCHASTIC_BASELINES else 1)}
    row_count = 0
    map_count = 0
    mismatches = []
    with gzip.open(result_path, "rt") as stream:
        for map_id, group in groupby((json.loads(line) for line in stream), key=lambda row: row["map_id"]):
            rows = list(group)
            keys = {(row["method"], row["alpha"], row["replicate"]) for row in rows}
            if len(keys) != len(rows) or keys != expected_row_keys:
                raise ValueError(f"Incomplete or duplicate archived result keys: {map_id}")
            map_count += 1
            row_count += len(rows)
            if map_id in selected:
                reference_path = reference_dir / f"{map_id}.json.gz"
                with gzip.open(reference_path, "wt") as reference_stream:
                    json.dump(rows, reference_stream)
                references[map_id] = reference_path
            per_map = defaultdict(list)
            for row in rows:
                observed_settings.add((row["exploration_profile"], row["dag_profile"]))
                for rate, (numerator, denominator) in RATE_FIELDS.items():
                    count, eligible = row[numerator], row[denominator]
                    expected = count / eligible if eligible else None
                    if not (0 <= count <= eligible <= row["steps"]) or not equivalent(row[rate], expected):
                        raise ValueError(f"Invalid archived rate/count: {map_id}, {row['method']}, {rate}")
                    if row[rate] is not None:
                        per_map[(row["method"], row["alpha"], row["exploration_profile"],
                                 row["dag_profile"], rate)].append(row[rate])
            for key, rates in per_map.items():
                settings[key].append(statistics.fmean(rates))

    setting_means = {key: statistics.fmean(values) for key, values in settings.items()}
    overall = defaultdict(list)
    for (method, alpha, exploration, dag, rate), mean in setting_means.items():
        overall[(method, alpha, rate)].append(mean)
    means_checked = 0
    all_setting_keys = {(method, alpha, exploration, dag, rate)
                        for method in manifest["config"]["methods"]
                        for alpha in manifest["config"]["alphas"]
                        for exploration, dag in observed_settings
                        for rate in RATE_FIELDS}
    all_overall_keys = {(method, alpha, rate) for method in manifest["config"]["methods"]
                        for alpha in manifest["config"]["alphas"] for rate in RATE_FIELDS}
    for name in ("by_setting.csv", "summary.csv"):
        seen_keys = set()
        with (directory / name).open(newline="") as stream:
            for row in csv.DictReader(stream):
                rate = row["metric"]
                if rate not in RATE_FIELDS:
                    continue
                base = (row["method"], float(row["alpha"]))
                if name == "by_setting.csv":
                    key = (*base, row["exploration_profile"], row["dag_profile"], rate)
                    expected = setting_means.get(key)
                else:
                    key = (*base, rate)
                    values = overall.get(key, [])
                    expected = statistics.fmean(values) if values else None
                if key in seen_keys:
                    raise ValueError(f"Duplicate error-rate summary key in {directory / name}: {key}")
                seen_keys.add(key)
                actual = float(row["mean"]) if row["mean"] else None
                means_checked += 1
                if not equivalent(actual, expected):
                    mismatches.append({"table": name, "method": row["method"], "alpha": row["alpha"],
                                       "metric": rate, "saved": actual, "recomputed": expected})
        expected_keys = all_setting_keys if name == "by_setting.csv" else all_overall_keys
        if seen_keys != expected_keys:
            raise ValueError(f"Missing or unexpected error-rate summary keys in {directory / name}")
    if row_count != manifest["row_count"] or map_count != manifest["map_count"]:
        raise ValueError(f"Archived inventory count mismatch: {directory}")
    if set(references) != set(selected):
        raise ValueError(f"Replay maps missing from archive: {directory}")
    return manifest, references, {
        "result_sha256": result_hash, "recorded_evaluator_sha256": manifest["code_sha256"],
        "rows_checked": row_count, "maps_checked": map_count,
        "rate_arithmetic_mismatches": 0, "summary_means_checked": means_checked,
        "summary_mean_mismatches": mismatches,
    }


def replay_map(record, config, reference_path):
    started = time.perf_counter()
    with gzip.open(reference_path, "rt") as stream:
        references = json.load(stream)
    expected = {(row["method"], row["replicate"], row["alpha"]): row for row in references}
    trajectories = sorted({(row["method"], row["replicate"]) for row in references})
    mismatches = []
    checked = 0
    seen_keys = set()
    for method, replicate in trajectories:
        rows = evaluate_record(record, method, alphas=config["alphas"], replicate=replicate,
                               seed=config["seed"], max_search_states=config["max_search_states"])
        for actual in rows:
            key = (method, replicate, actual["alpha"])
            if key in seen_keys or key not in expected:
                raise ValueError(f"Duplicate or unexpected replay cutoff: {record['map_id']}, {key}")
            seen_keys.add(key)
            saved = expected[key]
            checked += 1
            for field in sorted(set(saved) | set(actual)):
                if field in TIMING_FIELDS:
                    continue
                if field not in saved or field not in actual or not equivalent(saved[field], actual[field]):
                    mismatches.append({"method": method, "replicate": replicate, "alpha": actual["alpha"],
                                       "field": field, "saved": saved.get(field), "replayed": actual.get(field)})
    if seen_keys != set(expected):
        raise ValueError(f"Missing replay cutoffs: {record['map_id']}")
    return {"map_id": record["map_id"], "episode_sha256": record["episode_sha256"],
            "trajectories": len(trajectories), "rows_checked": checked, "mismatches": mismatches,
            "elapsed_seconds": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Benchmark repository or data directory; default: this repository")
    parser.add_argument("--results-root", type=Path,
                        help="Directory containing v2/v3 result archives; default: <root>/analysis/baselines")
    parser.add_argument("--versions", nargs="+", choices=("v2", "v3"), default=("v2", "v3"))
    parser.add_argument("--maps-per-setting", type=int, default=1)
    parser.add_argument("--all-maps", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("analysis/baselines/metric-audit.json"),
                        help="Audit report destination; does not select result archives")
    args = parser.parse_args()
    if args.maps_per_setting < 1 or args.workers < 1:
        parser.error("maps-per-setting and workers must be positive")
    results_root = args.results_root if args.results_root is not None else args.root / "analysis/baselines"
    started = time.perf_counter()
    dependency = verify_metric_dependency()
    report = {
        "schema_version": 1, "metric_dependency": dependency,
        "audit_script_sha256": file_sha256(Path(__file__)),
        "current_evaluator_sha256": evaluator_source_sha256(),
        "selection": "all inventory maps" if args.all_maps else "first inventory maps in each exploration/task-DAG setting",
        "maps_per_setting": None if args.all_maps else args.maps_per_setting,
        "workers": args.workers,
        "scope": "All archived rows and error-rate summary means; deterministic replay only for the explicitly listed maps. Timing fields are excluded from replay comparisons.",
        "versions": {},
    }
    valid = True
    for version in args.versions:
        records = select_records(version, args.root, report["maps_per_setting"])
        replayed = []
        with tempfile.TemporaryDirectory(prefix="metric-audit-") as reference_dir:
            manifest, references, archive = inspect_archive(results_root / version,
                                                            records, reference_dir)
            print(f"{version}: checked {archive['rows_checked']:,} archived rows; replaying {len(records)} maps", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                tasks = [pool.submit(replay_map, record, manifest["config"], references[map_id])
                         for map_id, record in records.items()]
                for future in as_completed(tasks):
                    result = future.result()
                    replayed.append(result)
                    if len(replayed) % 10 == 0 or len(replayed) == len(records):
                        print(f"{version}: replayed {len(replayed)}/{len(records)} maps", flush=True)
        replayed.sort(key=lambda row: row["map_id"])
        mismatches = sum(len(row["mismatches"]) for row in replayed)
        valid &= not archive["summary_mean_mismatches"] and not mismatches
        report["versions"][version] = {
            **archive, "recorded_evaluator_matches_current": manifest["code_sha256"] == report["current_evaluator_sha256"],
            "replay_maps": len(replayed), "replay_trajectories": sum(row["trajectories"] for row in replayed),
            "replay_rows": sum(row["rows_checked"] for row in replayed), "replay_mismatches": mismatches,
            "replay_selection_sha256": canonical_sha256([(row["map_id"], row["episode_sha256"]) for row in replayed]),
            "replays": replayed,
        }
    report["valid"] = bool(valid)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Audit {'passed' if valid else 'FAILED'}: {args.output}", flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
