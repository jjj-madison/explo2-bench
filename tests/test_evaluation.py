from dataclasses import replace
import gzip
import json
from pathlib import Path
import runpy
import sys

import pytest

from explo2_bench.baselines import BASELINE_NAMES, create_agent
from explo2_bench.data import make_record
from explo2_bench.generation import generate
from explo2_bench.evaluation import action_budgets, evaluate_record, policy_seed, run_evaluation
from explo2_bench._core.environment import SymbolicEnvironment
from symbolic_environment.metrics import compute_metrics


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
def test_default_output_preserves_published_analysis(tmp_path, monkeypatch, entrypoint):
    from explo2_bench import evaluation

    record = make_record("v2", generate("v2", "lower", "easier", 3))
    monkeypatch.setattr(evaluation, "load_benchmark", lambda *args, **kwargs: iter([record]))
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / "analysis" / "baselines" / "v2"
    canonical.mkdir(parents=True)
    (canonical / "run.json").write_text(json.dumps({"identity": "published-reference-run"}))
    (canonical / "results.jsonl.gz").write_bytes(b"published results sentinel")
    published = {path.relative_to(canonical): path.read_bytes() for path in canonical.rglob("*") if path.is_file()}
    config = dict(alphas=[1], methods=["frontier_revisit"],
                  stochastic_replicates=1, seed=1, max_search_states=10)

    if entrypoint == "api":
        path = run_evaluation("v2", config=config, limit=1)
        assert path == Path("outputs/baselines/v2/results.jsonl.gz")
    else:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_baselines.py"
        monkeypatch.setattr(sys, "argv", [str(script), "--version", "v2", "--config", str(config_path),
                                         "--limit", "1", "--no-figures"])
        runpy.run_path(str(script), run_name="__main__")

    output = tmp_path / "outputs" / "baselines" / "v2"
    with gzip.open(output / "results.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 1
    assert rows[0]["map_id"] == record["map_id"]
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    assert {path.relative_to(canonical): path.read_bytes() for path in canonical.rglob("*") if path.is_file()} == published
    assert not (canonical / ".checkpoints").exists()
    if entrypoint == "cli":
        assert (output / "summary.csv").is_file()


@pytest.mark.parametrize("method", BASELINE_NAMES)
def test_prefix_matches_independent_cutoff(method):
    episode = generate("v2", "lower", "easier", 3)
    record = make_record("v2", episode)
    rows = evaluate_record(record, method, alphas=(0.5, 1, 2, 10), max_search_states=20)
    for row in rows:
        instance = replace(episode, hidden_budget=row["action_budget"])
        env = SymbolicEnvironment(instance)
        agent = create_agent(method, seed=policy_seed(record["map_id"], method, 0, 20261010), max_search_states=20)
        trajectory = env.run(agent)
        metrics = compute_metrics(instance, trajectory)
        assert row["steps"] == trajectory.total_steps
        assert row["success"] == trajectory.success
        assert row["exploration_errors"] == metrics.total_exploration_errors
        assert row["exploitation_errors"] == metrics.total_exploitation_errors
        assert row["activated_count"] == len(trajectory.records[-1].activated)
        assert row["visited_cells"] == len({r.position for r in trajectory.records})
        assert row["planner_diagnostics"] == agent.diagnostics
    assert record["episode"]["hidden_budget"] == episode.hidden_budget


@pytest.mark.parametrize("alphas", [[], [1, 1], [True], [float("nan")], [float("inf")], [0], [-1], [0.01]])
def test_bad_budgets(alphas):
    with pytest.raises(ValueError):
        action_budgets(5, alphas)


def test_odd_cells_use_exact_floor():
    assert action_budgets(5, [0.5, 1.5, 10]) == [(0.5, 2), (1.5, 7), (10, 50)]


def test_resume_and_configuration_rejection(tmp_path):
    config = dict(alphas=[0.5, 1], methods=["random_walk", "frontier_revisit"],
                  stochastic_replicates=3, seed=1, max_search_states=10)
    path = run_evaluation("v2", config=config, output_root=tmp_path, limit=1)
    with gzip.open(path, "rt") as handle:
        before = [json.loads(line) for line in handle]
    assert len(before) == 8
    run_evaluation("v2", config=config, output_root=tmp_path, limit=1)
    with gzip.open(path, "rt") as handle:
        assert before == [json.loads(line) for line in handle]
    config["seed"] = 2
    with pytest.raises(ValueError, match="Different evaluation"):
        run_evaluation("v2", config=config, output_root=tmp_path, limit=1)
