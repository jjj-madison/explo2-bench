from __future__ import annotations

import unittest
import json
from copy import deepcopy
from pathlib import Path

from explo2_bench._core.environment import SymbolicEnvironment
from explo2_bench.baselines.base import AgentEpisodeFailure
from explo2_bench._core.types import (
    Action,
    ConstructiveKnobs,
    EpisodeInstance,
    ExplorationLevel,
    GridSpec,
    TaskDAGDifficulty,
    TaskGraph,
    TaskNode,
)
from explo2_bench.environment import (
    LMCallableAgent, make_env, safe_observation, observation_to_text,
)


def public_record(instance):
    native = instance.to_dict()
    native["configuration_version"] = "v2"
    native["metadata"].update({"feasibility_witness": [[99, 99]], "secret": "PRIVATE"})
    return {"schema_version": 1, "version": "v2", "episode": native,
            "map_id": "PRIVATE_MAP", "generation": {"seed": 13579},
            "canary": {"project": "CANARY_PRIVATE"}, "split": None}


def _node(
    indicator_id: str,
    *,
    visible_name: str | None = None,
    prerequisites: list[set[str]] | None = None,
    depth: int = 0,
) -> TaskNode:
    return TaskNode(
        indicator_id=indicator_id,
        visible_name=indicator_id if visible_name is None else visible_name,
        prerequisites=[] if prerequisites is None else prerequisites,
        depth=depth,
    )


def _instance(
    *,
    nodes: dict[str, TaskNode],
    goal_id: str,
    discovery_map: dict[tuple[int, int], set[str]],
    start: tuple[int, int] = (0, 0),
    width: int = 3,
    height: int = 3,
    hidden_budget: int = 5,
) -> EpisodeInstance:
    return EpisodeInstance(
        episode_id="episode",
        exploration_level=ExplorationLevel.HIGH,
        task_dag_difficulty=TaskDAGDifficulty.EASY,
        seed=0,
        knobs=ConstructiveKnobs(
            alpha_budget=1.0,
            num_nodes=len(nodes),
            dag_density_over_grid=len(nodes) / max(1, width * height),
            grid_aspect_ratio=width / max(1, height),
            max_nodes_per_layer=max(
                1,
                max(
                    sum(1 for node in nodes.values() if node.depth == depth)
                    for depth in {node.depth for node in nodes.values()}
                ),
            ),
            path_width_probs={1: 1.0},
            option_count_probs={
                max(1, max((len(node.prerequisites) for node in nodes.values()), default=0)): 1.0
            },
            dependency_count_probs={
                max(
                    1,
                    max(
                        (len(option) for node in nodes.values() for option in node.prerequisites),
                        default=0,
                    ),
                ): 1.0
            },
            goal_dependency_count_probs={
                max(
                    1,
                    max(
                        (len(option) for node in nodes.values() for option in node.prerequisites),
                        default=0,
                    ),
                ): 1.0
            },
        ),
        grid=GridSpec(width=width, height=height, blocked=set(), start=start),
        task_graph=TaskGraph(nodes=nodes, goal_id=goal_id),
        discovery_map=discovery_map,
        required_nodes={goal_id},
        hidden_budget=hidden_budget,
    )


class _LoopAgent:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, observation: object) -> Action:
        del observation
        raise NotImplementedError


class _BackAndForthAgent(_LoopAgent):
    def act(self, observation: object) -> Action:
        position = observation.position  # type: ignore[attr-defined]
        return Action.move((1, 0) if position == (0, 0) else (0, 0))


class _FailingAgent(_LoopAgent):
    def act(self, observation: object) -> Action:
        del observation
        raise AgentEpisodeFailure(
            "excessive_reasoning_context_exhausted",
            "maximum context length reached",
        )


class EnvironmentTest(unittest.TestCase):
    def test_episode_loading_uses_canonical_exploration_levels(self) -> None:
        native = _instance(
            nodes={"goal": _node("goal")}, goal_id="goal",
            discovery_map={(0, 0): {"goal"}},
        ).to_dict()
        for raw, expected in ((" HIGH ", ExplorationLevel.HIGH),
                              ("medium", ExplorationLevel.MEDIUM),
                              ("Low", ExplorationLevel.LOW),
                              (" custom ", ExplorationLevel.CUSTOM)):
            with self.subTest(level=raw):
                restored = EpisodeInstance.from_dict({**native, "exploration_level": raw})
                self.assertEqual(restored.exploration_level, expected)
                self.assertEqual(restored.to_dict(), {**native, "exploration_level": expected.value})
        for raw in ("easy", "hard"):
            with self.subTest(level=raw), self.assertRaises(ValueError):
                EpisodeInstance.from_dict({**native, "exploration_level": raw})
        invalid_record = {**native, "grid_difficulty": native["exploration_level"]}
        del invalid_record["exploration_level"]
        with self.assertRaises(KeyError):
            EpisodeInstance.from_dict(invalid_record)

    def test_exploit_only_reset_reveals_full_map_graph_and_true_locations(self) -> None:
        nodes = {
            "tool": _node("tool", visible_name="TOOL", depth=0),
            "goal": _node("goal", visible_name="GOAL", prerequisites=[{"tool"}], depth=1),
        }
        instance = _instance(
            nodes=nodes,
            goal_id="goal",
            discovery_map={(1, 0): {"tool"}, (2, 2): {"goal"}},
        )

        env = SymbolicEnvironment(instance, exploit_only=True)
        obs = env.reset()

        self.assertEqual(env.state.discovered, {"tool", "goal"})
        self.assertEqual(obs.newly_discovered_visible, ["GOAL", "TOOL"])
        self.assertTrue(obs.exploit_only)
        self.assertEqual(len(obs.open_cells), 9)
        self.assertEqual(obs.state_positions_visible, {"GOAL": (2, 2), "TOOL": (1, 0)})
        self.assertEqual(obs.all_prerequisites_visible["GOAL"], [["TOOL"]])
        self.assertEqual(obs.goal_visible, "GOAL")

    def test_exploit_only_reveal_does_not_activate_remote_states(self) -> None:
        nodes = {
            "tool": _node("tool", depth=0),
            "goal": _node("goal", prerequisites=[{"tool"}], depth=1),
        }
        instance = _instance(
            nodes=nodes,
            goal_id="goal",
            discovery_map={(1, 0): {"tool"}, (2, 0): {"goal"}},
            width=3,
            height=1,
        )

        env = SymbolicEnvironment(instance, exploit_only=True)
        env.reset()
        self.assertEqual(env.state.activated, set())

        env.step(Action.move((1, 0)))
        self.assertEqual(env.state.activated, {"tool"})
        env.step(Action.move((2, 0)))
        self.assertEqual(env.state.activated, {"tool", "goal"})

    def test_reset_goal_at_start_terminates_and_prevents_further_steps(self) -> None:
        instance = _instance(
            nodes={"goal": _node("goal")},
            goal_id="goal",
            discovery_map={(0, 0): {"goal"}},
        )

        env = SymbolicEnvironment(instance)
        obs = env.reset()

        self.assertTrue(env.state.done)
        self.assertEqual(env.state.terminated_reason, "goal_activated_at_start")
        self.assertEqual(obs.newly_discovered_visible, ["goal"])
        self.assertEqual(obs.newly_activated_visible, ["goal"])
        self.assertEqual(env.history[-1].action, "START")

        with self.assertRaisesRegex(RuntimeError, "already terminated"):
            env.step(Action.move((1, 0)))

    def test_invalid_move_is_recorded_without_changing_position(self) -> None:
        instance = _instance(
            nodes={"goal": _node("goal")},
            goal_id="goal",
            discovery_map={},
            hidden_budget=3,
        )

        env = SymbolicEnvironment(instance)
        env.reset()
        obs = env.step(Action(kind="move", value="bad-coord"))

        self.assertEqual(obs.position, (0, 0))
        self.assertEqual(env.state.position, (0, 0))
        self.assertFalse(env.history[-1].valid)
        self.assertEqual(env.history[-1].note, "invalid move")
        self.assertFalse(env.state.done)

    def test_activation_requires_visiting_the_current_cell(self) -> None:
        nodes = {
            "tool": _node("tool", depth=0),
            "goal": _node("goal", prerequisites=[{"tool"}], depth=1),
        }
        instance = _instance(
            nodes=nodes,
            goal_id="goal",
            discovery_map={(1, 0): {"tool"}, (1, 1): {"goal"}},
        )

        env = SymbolicEnvironment(instance)
        env.reset()

        first_obs = env.step(Action.move((1, 0)))
        self.assertEqual(first_obs.newly_discovered_visible, ["tool"])
        self.assertEqual(first_obs.newly_activated_visible, ["tool"])
        self.assertEqual(env.state.activated, {"tool"})
        self.assertFalse(env.state.done)

        second_obs = env.step(Action.move((1, 1)))
        self.assertEqual(second_obs.newly_discovered_visible, ["goal"])
        self.assertEqual(second_obs.newly_activated_visible, ["goal"])
        self.assertEqual(env.state.activated, {"tool", "goal"})
        self.assertTrue(env.state.done)
        self.assertEqual(env.state.terminated_reason, "goal_activated")

    def test_single_visit_can_chain_local_activation_safely(self) -> None:
        nodes = {
            "tool": _node("tool", depth=0),
            "goal": _node("goal", prerequisites=[{"tool"}], depth=1),
        }
        instance = _instance(
            nodes=nodes,
            goal_id="goal",
            discovery_map={(1, 0): {"tool", "goal"}},
        )

        env = SymbolicEnvironment(instance)
        env.reset()
        obs = env.step(Action.move((1, 0)))

        self.assertEqual(sorted(obs.newly_discovered_visible), ["goal", "tool"])
        self.assertEqual(sorted(obs.newly_activated_visible), ["goal", "tool"])
        self.assertEqual(env.state.activated, {"tool", "goal"})
        self.assertTrue(env.state.done)
        self.assertEqual(env.state.terminated_reason, "goal_activated")

    def test_discovery_payload_reports_visible_prerequisites_and_ancestors(self) -> None:
        nodes = {
            "alt": _node("alt", visible_name="ALT", depth=0),
            "tool": _node("tool", visible_name="TOOL", depth=0),
            "mid": _node("mid", visible_name="MID", prerequisites=[{"alt", "tool"}], depth=1),
            "goal": _node("goal", visible_name="GOAL", prerequisites=[{"mid"}], depth=2),
        }
        instance = _instance(
            nodes=nodes,
            goal_id="goal",
            discovery_map={(0, 0): {"alt", "tool"}, (1, 0): {"mid"}},
            width=2,
            height=1,
        )

        env = SymbolicEnvironment(instance)
        env.reset()
        obs = env.step(Action.move((1, 0)))

        self.assertEqual(obs.newly_discovered_visible, ["MID"])
        self.assertEqual(obs.newly_discovered_prerequisites_visible, {"MID": [["ALT", "TOOL"]]})
        self.assertEqual(obs.newly_discovered_ancestors_visible, {"MID": ["GOAL"]})

    def test_unknown_action_is_recorded_and_can_exhaust_budget(self) -> None:
        instance = _instance(
            nodes={"goal": _node("goal")},
            goal_id="goal",
            discovery_map={},
            hidden_budget=1,
        )

        env = SymbolicEnvironment(instance)
        env.reset()
        obs = env.step(Action(kind="wait", value=""))

        self.assertEqual(obs.position, (0, 0))
        self.assertEqual(env.history[-1].note, "unknown action")
        self.assertFalse(env.history[-1].valid)
        self.assertTrue(env.state.done)
        self.assertEqual(env.state.terminated_reason, "budget_exhausted")

    def test_reset_clears_history_and_runtime_state(self) -> None:
        instance = _instance(
            nodes={"goal": _node("goal")},
            goal_id="goal",
            discovery_map={(1, 0): {"goal"}},
            width=2,
            height=1,
            hidden_budget=3,
        )

        env = SymbolicEnvironment(instance)
        env.reset()
        env.step(Action.move((1, 0)))
        env.reset()

        self.assertEqual(env.state.t, 0)
        self.assertEqual(env.state.position, (0, 0))
        self.assertEqual(len(env.history), 1)
        self.assertEqual(env.history[0].action, "START")

    def test_run_resets_agent_and_returns_budget_exhausted_trajectory(self) -> None:
        instance = _instance(
            nodes={"goal": _node("goal")},
            goal_id="goal",
            discovery_map={},
            width=2,
            height=1,
            hidden_budget=2,
        )
        agent = _BackAndForthAgent()

        trajectory = SymbolicEnvironment(instance).run(agent)

        self.assertEqual(agent.reset_calls, 1)
        self.assertFalse(trajectory.success)
        self.assertEqual(trajectory.terminated_reason, "budget_exhausted")
        self.assertEqual(trajectory.total_steps, 2)
        self.assertEqual([record.action for record in trajectory.records], ["START", "MOVE 1,0", "MOVE 0,0"])

    def test_run_scores_agent_terminal_failure_without_crashing_suite(self) -> None:
        instance = _instance(
            nodes={"goal": _node("goal")},
            goal_id="goal",
            discovery_map={},
            hidden_budget=5,
        )

        trajectory = SymbolicEnvironment(instance).run(_FailingAgent())

        self.assertFalse(trajectory.success)
        self.assertEqual(
            trajectory.terminated_reason,
            "excessive_reasoning_context_exhausted",
        )
        self.assertEqual(trajectory.total_steps, 1)
        self.assertEqual(trajectory.records[-1].action, "AGENT_FAILURE")
        self.assertFalse(trajectory.records[-1].valid)



class PublicInterfaceTest(unittest.TestCase):
    def record(self):
        return public_record(_instance(
            nodes={"goal": _node("goal")}, goal_id="goal",
            discovery_map={(2, 0): {"goal"}}, width=3, height=1, hidden_budget=9,
        ))

    def test_safe_observations_and_info_exclude_private_metadata(self):
        record = self.record()
        original = deepcopy(record)
        env = make_env(record, action_budget=3)
        obs = env.reset()
        self.assertEqual(obs["legal_moves"], ["right"])
        result = env.step({"move": "right"})
        encoded = json.dumps([obs, result.observation, result.info])
        for secret in ("PRIVATE", "13579", "hidden_budget", "feasibility_witness", "map_id", "canary"):
            self.assertNotIn(secret, encoded)
        self.assertEqual(record, original)
        # Modifying an allowed collection cannot corrupt core observations.
        core = env._env._make_observation(set(), set())
        safe = safe_observation(core)
        safe["discovered"].append("injected")
        self.assertNotIn("injected", core.discovered_visible)
        self.assertNotIn("PRIVATE", observation_to_text({**obs, "canary": "PRIVATE"}))

    def test_invalid_actions_charge_exactly_one_turn_and_success_wins_at_limit(self):
        env = make_env(self.record(), action_budget=3)
        env.reset()
        result = env.step('not json')
        self.assertFalse(result.info["valid"])
        self.assertEqual(result.observation["t"], 1)
        self.assertEqual(result.observation["position"], [0, 0])
        self.assertFalse(env.step({"move": "right"}).done)
        result = env.step({"move": "right"})
        self.assertTrue(result.done)
        self.assertTrue(result.success)
        self.assertEqual(result.info["terminated_reason"], "goal_activated")
        with self.assertRaises(RuntimeError):
            env.step({"move": "left"})

    def test_alpha_is_exact_floor_and_rejects_sub_one_budget(self):
        env = make_env(self.record(), alpha=0.7)
        self.assertEqual(env._env.instance.hidden_budget, 2)
        for kwargs in ({"alpha": 0.1}, {"alpha": True}, {"alpha": float("inf")},
                       {"action_budget": True}, {"action_budget": 0},
                       {"alpha": 1, "action_budget": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_env(self.record(), **kwargs)

    def test_malformed_callbacks_are_invalid_actions_and_exceptions_are_distinct(self):
        for response in ({"move": "right", "extra": 1}, {"move": "north"}, [], "```json\n{}\n```"):
            env = make_env(self.record(), action_budget=1)
            agent = LMCallableAgent(lambda obs: response)
            trajectory = env._env.run(agent)
            self.assertFalse(trajectory.success)
            self.assertEqual(trajectory.total_steps, 1)
            self.assertEqual(trajectory.terminated_reason, "budget_exhausted")
            self.assertEqual(agent.diagnostics["malformed_actions"], 1)
        def broken(_):
            raise RuntimeError("PRIVATE_PROVIDER_MESSAGE")
        env = make_env(self.record())
        trajectory = env._env.run(LMCallableAgent(broken))
        self.assertEqual(trajectory.terminated_reason, "callback_error")
        self.assertNotIn("PRIVATE_PROVIDER_MESSAGE", trajectory.records[-1].note)


def metric_golden_cases():
    """Deterministic paths with revisits, invalid actions, mixed regimes and OR."""
    import random
    class MixedActions:
        def reset(self):
            self.rng = random.Random(431)
            self.n = 0
        def act(self, obs):
            self.n += 1
            if self.n % 7 == 0:
                return Action("invalid", "")
            return Action.move(self.rng.choice(sorted(obs.neighbors)))
    instance = _instance(
        nodes={"root": _node("root"), "a": _node("a", prerequisites=[{"root"}]),
               "b": _node("b", prerequisites=[{"a"}]),
               "goal": _node("goal", prerequisites=[{"a", "b"}, {"b", "root"}])},
        goal_id="goal", discovery_map={(0, 0): {"root"}, (2, 0): {"a"},
                                       (0, 1): {"b"}, (2, 2): {"goal"}},
        width=3, height=3, hidden_budget=90,
    )
    cases = []
    for mapping in (instance.discovery_map, {(0, 0): {"root"}, (2, 0): {"a"},
                                            (2, 2): {"b"}, (0, 1): {"goal"}}):
        instance = deepcopy(instance)
        instance.discovery_map = mapping
        trajectory = SymbolicEnvironment(instance).run(MixedActions())
        cases.extend((instance, trajectory, flags) for flags in (
            {}, {"all_cells_seen": True},
            {"initial_visited": {(0, 0), (0, 1), (1, 0)}},
        ))
    return cases


def test_upstream_metrics_match_preserved_reference_outputs():
    """Compare every metric field with the stored reference-output hashes.

    Six supported cases cover invalid actions, target progress, cycles, repeated
    edges, mixed/goal regimes, full-known graphs and initial visited overrides.
    The fixture records the source implementation and eight-case reference
    fingerprint. Two cases requiring the private exploit-only metric interface
    are excluded from this public-interface check.
    """
    import hashlib
    from symbolic_environment.metrics import compute_metrics
    reference = json.loads((Path(__file__).parent / "fixtures/metric_reference.json").read_text())
    values = [compute_metrics(i, t, **flags).to_dict()
              for i, t, flags in metric_golden_cases()]
    for value, expected in zip(values, reference["cases"], strict=True):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == expected["metrics_sha256"], expected["case_index"]
