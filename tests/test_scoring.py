"""The public scoring API uses the same metric contract as baseline evaluation."""

from copy import deepcopy
import json

import pytest

from explo2_bench import generate, make_env, score_episode
from explo2_bench._core.types import Action, GridSpec, TaskGraph, TaskNode
from explo2_bench.baselines import create_agent
from explo2_bench.data import make_record
from explo2_bench.evaluation import evaluate_record, policy_seed


def corridor_record(*, goal_at_start=False):
    episode = generate("v2", "lower", "easier", 0)
    episode.grid = GridSpec(3, 1, set(), (0, 0))
    episode.task_graph = TaskGraph({"G": TaskNode("G", "Goal", [set()], 0)}, "G")
    episode.required_nodes = {"G"}
    episode.discovery_map = {(0, 0) if goal_at_start else (2, 0): {"G"}}
    episode.hidden_budget = 9
    episode.metadata["private_test_marker"] = "PRIVATE_WITNESS"
    return make_record("v2", episode)


def test_scoring_requires_public_environment_reset_and_termination():
    with pytest.raises(TypeError, match="BenchmarkEnv"):
        score_episode({})
    env = make_env(corridor_record(), action_budget=3)
    with pytest.raises(ValueError, match="Reset"):
        score_episode(env)
    env.reset()
    with pytest.raises(ValueError, match="completed"):
        score_episode(env)


def test_invalid_action_and_success_on_last_turn_are_scored_without_private_records():
    env = make_env(corridor_record(), action_budget=3)
    env.reset()
    env.step("malformed")
    env.step({"move": "right"})
    env.step({"move": "right"})
    history = deepcopy(env._env.history)
    score = score_episode(env)
    assert score["success"] is True
    assert score["steps"] == score["action_budget"] == score["goal_activation_step"] == 3
    assert score["invalid_actions"] == score["union_errors"] == score["exploration_errors"] == 1
    assert score["exploration_eligible_steps"] == 3
    assert score["exploration_error_rate"] == pytest.approx(1 / 3)
    assert score["exploitation_eligible_steps"] == 0
    assert score["exploitation_error_rate"] is None
    assert score["coverage"] == score["activated_fraction"] == score["discovered_fraction"] == 1
    assert env._env.history == history
    assert score_episode(env) == score
    encoded = json.dumps(score)
    for private in ("PRIVATE_WITNESS", "canary", "seed", "discovery_map", "task_graph", "trajectory", "records"):
        assert private not in encoded


def test_failed_budget_and_zero_action_success_have_explicit_missing_rates():
    failure = make_env(corridor_record(), action_budget=1)
    failure.reset()
    failure.step({"move": "right"})
    scores = score_episode(failure)
    assert scores["success"] is False
    assert scores["terminated_reason"] == "budget_exhausted"
    assert scores["goal_activation_step"] is None
    assert scores["coverage"] == pytest.approx(2 / 3)
    success = make_env(corridor_record(goal_at_start=True))
    success.reset()
    scores = score_episode(success)
    assert scores["success"] is True
    assert scores["steps"] == scores["goal_activation_step"] == 0
    assert scores["union_error_rate"] is None
    assert scores["exploration_error_rate"] is None
    assert scores["exploitation_error_rate"] is None


@pytest.mark.parametrize("method", ["random_walk", "frontier_revisit"])
@pytest.mark.parametrize("alpha", [0.5, 3])
def test_public_scores_match_budget_runner_results(method, alpha):
    record = make_record("v2", generate("v2", "lower", "easier", 3))
    expected = evaluate_record(record, method, alphas=[alpha])[0]
    env = make_env(record, alpha=alpha)
    # Drive the same public wrapper with the native baseline adapter in this
    # test; ordinary LM clients use reset/step with the public JSON contract.
    agent = create_agent(method, seed=policy_seed(record["map_id"], method, 0, 20261010))
    env._env.run(agent)
    scores = score_episode(env)
    for key in set(scores) & set(expected):
        assert scores[key] == expected[key], key


def test_goal_pending_steps_never_dilute_exploration_error_rate(monkeypatch):
    """Two exploration errors use the four exploration-eligible actions.

    Discover the goal before its prerequisite, then spend six actions with the
    goal ready. Three of those actions move away from the goal. Goal-pending
    actions contribute only to the exploitation denominator.
    """
    episode = generate("v2", "lower", "easier", 0)
    episode.grid = GridSpec(8, 1, set(), (0, 0))
    episode.task_graph = TaskGraph({
        "R": TaskNode("R", "Root", [set()], 0),
        "G": TaskNode("G", "Goal", [{"R"}], 1),
    }, "G")
    episode.discovery_map = {(1, 0): {"G"}, (2, 0): {"R"}}
    episode.required_nodes = {"R", "G"}
    episode.hidden_budget = 10
    record = make_record("v2", episode)
    moves = [None, None, "right", "right", "right", "right", "right", "left", "left", "left"]
    env = make_env(record)
    env.reset()
    for move in moves:
        env.step("malformed" if move is None else {"move": move})
    scores = score_episode(env)
    assert scores["steps"] == 10
    assert scores["mixed_goal_steps"] == 6
    assert scores["exploration_errors"] == 2
    assert scores["exploration_eligible_steps"] == 4
    assert scores["exploration_error_rate"] == 0.5
    assert scores["exploitation_errors"] == 3
    assert scores["exploitation_eligible_steps"] == 6
    assert scores["exploitation_error_rate"] == 0.5

    class ScriptedAgent:
        diagnostics = {}

        def reset(self):
            self.actions = iter(moves)

        def act(self, observation):
            direction = next(self.actions)
            if direction is None:
                return Action("invalid", "")
            x, y = observation.position
            return Action.move((x + (1 if direction == "right" else -1), y))

    from explo2_bench import evaluation
    monkeypatch.setattr(evaluation, "create_agent", lambda *args, **kwargs: ScriptedAgent())
    prefix, complete = evaluate_record(record, "scripted", alphas=[0.5, 1.25])
    for row in (prefix, complete):
        assert row["exploration_eligible_steps"] == 4
        assert row["exploration_error_rate"] == 0.5
    assert prefix["steps"] == 4
    assert prefix["exploitation_eligible_steps"] == 0
    assert prefix["exploitation_error_rate"] is None
    assert complete["steps"] == 10
    assert complete["exploitation_eligible_steps"] == 6
    assert complete["exploitation_error_rate"] == 0.5
