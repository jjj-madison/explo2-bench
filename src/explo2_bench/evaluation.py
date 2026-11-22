"""Resumable evaluation with one budget-blind rollout per map/policy/replicate.

All lower action cutoffs are scored from the same longest trajectory. Planning
time includes policy reset and act calls, but excludes environment and scoring.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from symbolic_environment.metrics import compute_metrics

from ._core.environment import SymbolicEnvironment
from ._core.types import Trajectory
from .baselines import BASELINE_NAMES, BASELINE_REVISION, STOCHASTIC_BASELINES, create_agent
from .data import canonical_json, canonical_sha256, episode_from_record, load_benchmark
from .upstream import metric_dependency_provenance, verify_metric_dependency

DEFAULT_ALPHAS = (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10)
EVALUATION_REVISION = 3


def action_budgets(open_cells: int, alphas=DEFAULT_ALPHAS) -> list[tuple[float, int]]:
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be nonempty and unique")
    values = []
    for alpha in alphas:
        if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alphas must be finite positive numbers")
        budget = math.floor(alpha * open_cells)
        if budget < 1:
            raise ValueError("Every action budget must be at least one")
        values.append((alpha, budget))
    return sorted(values)


def policy_seed(map_id: str, method: str, replicate: int, seed: int) -> int:
    key = canonical_json([seed, map_id, method, replicate])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def evaluate_record(record: dict, method: str, *, alphas=DEFAULT_ALPHAS,
                    replicate: int = 0, seed: int = 20261010,
                    max_search_states: int = 10_000) -> list[dict]:
    """Evaluate one policy and return one row per action cutoff.

    Timing is empirical, not bitwise reproducible. State counts, outcomes and
    action sequences are deterministic for a fixed code revision and seed.
    """
    instance = episode_from_record(record)
    open_cells = len(instance.grid.open_cells())
    cutoffs = action_budgets(open_cells, alphas)
    instance = replace(instance, hidden_budget=max(b for _, b in cutoffs))
    rng_seed = policy_seed(record["map_id"], method, replicate, seed)
    agent = create_agent(method, seed=rng_seed, max_search_states=max_search_states)
    env = SymbolicEnvironment(instance)
    wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
    agent.reset()
    cpu_ns, wall_ns = time.process_time_ns() - cpu_start, time.perf_counter_ns() - wall_start
    observation = env.reset()
    checkpoints = {0: (cpu_ns, wall_ns, dict(agent.diagnostics))}
    wanted = {budget for _, budget in cutoffs}
    while not env.state.done:
        wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
        action = agent.act(observation)
        cpu_ns += time.process_time_ns() - cpu_start
        wall_ns += time.perf_counter_ns() - wall_start
        observation = env.step(action)
        if env.state.t in wanted or env.state.done:
            checkpoints[env.state.t] = (cpu_ns, wall_ns, dict(agent.diagnostics))
    trajectory = Trajectory(list(env.history), env.goal_id in env.state.activated,
                            env.state.terminated_reason, env.state.t)
    scoring_start = time.process_time_ns()
    metrics = compute_metrics(instance, trajectory)
    scoring_cpu_seconds = (time.process_time_ns() - scoring_start) / 1e9
    errors = {error.t: error for error in metrics.step_errors}
    start = trajectory.records[0]
    visited = {start.position}
    goal_step = 0 if env.goal_id in start.activated else None
    totals = dict(exploration_errors=0, exploitation_errors=0, union_errors=0,
                  exploration_eligible_steps=0, exploitation_eligible_steps=0,
                  invalid_actions=0)
    rows = []
    index = 0
    for alpha, budget in cutoffs:
        observed = min(budget, trajectory.total_steps)
        while index < observed:
            index += 1
            rec = trajectory.records[index]
            visited.add(rec.position)
            totals["invalid_actions"] += not rec.valid
            if goal_step is None and env.goal_id in rec.activated:
                goal_step = rec.t
            error = errors.get(rec.t)
            if error is not None:
                totals["exploration_errors"] += error.exploration_error
                totals["exploitation_errors"] += error.exploitation_error
                totals["union_errors"] += bool(error.exploration_error or error.exploitation_error)
                totals["exploration_eligible_steps"] += error.regime in ("explore_only", "mixed_non_goal")
                totals["exploitation_eligible_steps"] += error.regime in ("exploit_only", "mixed_goal", "mixed_non_goal")
        rec = trajectory.records[observed]
        cpu, wall, diagnostics = checkpoints[observed]
        success = goal_step is not None and goal_step <= budget
        rows.append({
            "map_id": record["map_id"], "version": record["version"],
            "episode_sha256": record["episode_sha256"], "split": record["split"],
            **record["generation"], "method": method, "replicate": replicate,
            "policy_seed": rng_seed, "alpha": alpha, "action_budget": budget,
            "open_cells": open_cells, "task_nodes": len(instance.task_graph.nodes),
            "success": success, "steps": observed,
            "terminated_reason": "goal_activated" if success else "budget_exhausted",
            "goal_activation_step": goal_step if success else None,
            "discovered_count": len(rec.discovered), "activated_count": len(rec.activated),
            "visited_cells": len(visited), "coverage": len(visited) / open_cells,
            **totals,
            "exploration_error_rate": (totals["exploration_errors"] / totals["exploration_eligible_steps"]
                                       if totals["exploration_eligible_steps"] else None),
            "exploitation_error_rate": (totals["exploitation_errors"] / totals["exploitation_eligible_steps"]
                                        if totals["exploitation_eligible_steps"] else None),
            "union_error_rate": totals["union_errors"] / observed if observed else None,
            "planning_cpu_seconds": cpu / 1e9, "planning_wall_seconds": wall / 1e9,
            "planning_cpu_seconds_per_action": cpu / 1e9 / observed if observed else None,
            "planner_diagnostics": diagnostics,
            "full_rollout_scoring_cpu_seconds": scoring_cpu_seconds,
        })
    return rows


def _evaluate_map_job(job):
    record, config, shard_path, identity, execution_id = job
    rows = []
    for method in config["methods"]:
        reps = config["stochastic_replicates"] if method in STOCHASTIC_BASELINES else 1
        for replicate in range(reps):
            rows.extend(evaluate_record(record, method, alphas=config["alphas"], replicate=replicate,
                                        seed=config["seed"], max_search_states=config["max_search_states"]))
    for row in rows:
        row["execution_id"] = execution_id
    payload = {"identity": identity, "episode_sha256": record["episode_sha256"], "rows": rows}
    path = Path(shard_path)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))
    temporary.replace(path)
    return record["map_id"], len(rows)


def evaluator_source_sha256() -> str:
    """Fingerprint local and installed upstream code before resuming a campaign."""
    package = Path(__file__).parent
    paths = [Path(__file__), package / "data.py", package / "environment.py", package / "upstream.py",
             *sorted((package / "baselines").glob("*.py")),
             *sorted((package / "_core").rglob("*.py"))]
    local = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths}
    return canonical_sha256({"local_sources": local, "metric_dependency": metric_dependency_provenance()})


def run_evaluation(version: str, *, root=None, output_root="outputs/baselines", config=None,
                   workers=1, limit=None, split=None) -> Path:
    """Write atomic per-map checkpoints, a complete compressed table and manifest."""
    dependency = verify_metric_dependency()
    config = dict(config or {"alphas": list(DEFAULT_ALPHAS), "methods": list(BASELINE_NAMES),
                            "stochastic_replicates": 3, "seed": 20261010, "max_search_states": 10_000})
    action_budgets(100, config["alphas"])
    if not config["methods"] or len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("methods must be nonempty and unique")
    for method in config["methods"]:
        create_agent(method, max_search_states=config["max_search_states"])
    for key in ("stochastic_replicates", "seed"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < (1 if key == "stochastic_replicates" else 0):
            raise ValueError(f"Invalid {key}")
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    from itertools import islice
    records = list(islice(load_benchmark(version, root=root, split=split), limit))
    if not records:
        raise ValueError("No maps selected")
    out = Path(output_root) / version
    shards = out / ".checkpoints"
    shards.mkdir(parents=True, exist_ok=True)
    selection = [(record["map_id"], record["episode_sha256"]) for record in records]
    identity = canonical_sha256({"config": config, "code": evaluator_source_sha256(),
                                "selection": selection, "revision": EVALUATION_REVISION})
    run_path = out / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["identity"] != identity:
        raise ValueError(f"Different evaluation already exists at {out}; use a new output_root")
    manifest = {"identity": identity, "status": "running", "version": version, "config": config,
                "map_count": len(records), "split_filter": split, "limit": limit,
                "selection_sha256": canonical_sha256(selection), "code_sha256": evaluator_source_sha256(),
                "baseline_revision": BASELINE_REVISION, "evaluation_revision": EVALUATION_REVISION,
                "metric_dependency": dependency,
                "python": platform.python_version(), "platform": platform.platform(),
                "host": platform.node(), "processor": platform.processor(),
                "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "workers": workers, "logical_cpus": os.cpu_count(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "cost_scope": "policy reset and act only; CPU process time and wall elapsed time",
                "timing_caveat": "Parallel host contention affects wall time; timing is not deterministic."}
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        manifest["started_utc"] = previous["started_utc"]
        executions = previous["executions"]
    else:
        executions = []
    execution = {key: manifest[key] for key in (
        "host", "processor", "platform", "python", "workers", "logical_cpus",
        "cpu_affinity_count")}
    execution["started_utc"] = datetime.now(timezone.utc).isoformat()
    execution_id = canonical_sha256(execution)[:16]
    execution["execution_id"] = execution_id
    executions.append(execution)
    manifest["executions"] = executions
    run_path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for record in records:
        path = shards / (record["map_id"] + ".json.gz")
        if path.exists():
            with gzip.open(path, "rt") as handle:
                cached = json.load(handle)
            if cached["identity"] != identity or cached["episode_sha256"] != record["episode_sha256"]:
                raise ValueError(f"Invalid checkpoint: {path}")
        else:
            jobs.append((record, config, str(path), identity, execution_id))
    print(f"{version}: {len(records)-len(jobs)}/{len(records)} maps cached; {workers} workers", flush=True)
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            _evaluate_map_job(job)
            if i % 10 == 0 or i == len(jobs):
                print(f"{version}: completed {i}/{len(jobs)} pending maps", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(_evaluate_map_job, job) for job in jobs]
            for i, future in enumerate(as_completed(pending), 1):
                future.result()
                if i % 25 == 0 or i == len(jobs):
                    print(f"{version}: completed {i}/{len(jobs)} pending maps", flush=True)
    destination = out / "results.jsonl.gz"
    temporary = out / "results.jsonl.gz.tmp"
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for record in records:
            with gzip.open(shards / (record["map_id"] + ".json.gz"), "rt") as shard:
                for row in json.load(shard)["rows"]:
                    handle.write(canonical_json(row) + "\n")
                    count += 1
    temporary.replace(destination)
    manifest.update(status="complete", row_count=count,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    results_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    run_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination
