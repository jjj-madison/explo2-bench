from collections import deque
from copy import deepcopy

import unittest

from explo2_bench.baselines.online_planner import OnlinePlannerAgent
from explo2_bench._core.environment import SymbolicEnvironment
from explo2_bench._core.types import (
    Action, ConstructiveKnobs, EpisodeInstance, ExplorationLevel, GridSpec,
    Observation, TaskDAGDifficulty, TaskGraph, TaskNode,
)


def known_environment(goal_options, *, parent_options=None):
    options = {
        "seed": [set()], "A": [{"seed"}], "B": [{"seed"}], "goal": goal_options,
    }
    if parent_options:
        options.update(parent_options)
    positions = {"seed": (0, 0), "A": (0, 1), "B": (2, 1), "goal": (2, 0)}
    nodes = {
        name: TaskNode(name, name, prerequisites, 0)
        for name, prerequisites in options.items()
    }
    instance = EpisodeInstance(
        episode_id="known-planner-toy", exploration_level=ExplorationLevel.CUSTOM,
        task_dag_difficulty=TaskDAGDifficulty.CUSTOM, seed=0,
        knobs=ConstructiveKnobs(
            alpha_budget=2.0, num_nodes=4, dag_density_over_grid=0.5,
            grid_aspect_ratio=1.5, max_nodes_per_layer=4,
            path_width_probs={1: 1.0}, option_count_probs={1: 1.0},
            dependency_count_probs={1: 1.0}, goal_dependency_count_probs={1: 1.0},
        ),
        grid=GridSpec(3, 2, set(), (0, 0)), task_graph=TaskGraph(nodes, "goal"),
        discovery_map={position: {node} for node, position in positions.items()},
        required_nodes=set(nodes), hidden_budget=128,
    )
    env = SymbolicEnvironment(instance)
    env.reset()
    # Simulate an already explored map without giving the agent an environment.
    env.state.discovered.update(nodes)
    return env, positions, options


def warm_agent(env, positions, options, **kwargs):
    agent = OnlinePlannerAgent(**kwargs)
    cells = env.instance.grid.open_cells()
    agent.memory.known_neighbors = {cell: set(env.instance.grid.neighbors(cell)) for cell in cells}
    agent.memory.visited.update(cells)
    agent.memory.discovered_at.update(positions)
    agent.memory.prerequisites.update(deepcopy(options))
    return agent


def observe(env):
    return env._make_observation(set(), set())


def shortest_actual_environment_solution(env):
    """Tiny oracle runs actual environment transitions, without planner helpers."""
    queue = deque([(deepcopy(env), 0)])
    seen = {(env.state.position, frozenset(env.state.activated))}
    while queue:
        current, distance = queue.popleft()
        for neighbor in current.instance.grid.neighbors(current.state.position):
            child = deepcopy(current)
            child.step(Action.move(neighbor))
            if child.goal_id in child.state.activated:
                return distance + 1
            key = (child.state.position, frozenset(child.state.activated))
            if key not in seen:
                seen.add(key)
                queue.append((child, distance + 1))
    return None


def run_known(agent, env):
    observation = observe(env)
    while not env.state.done:
        action = agent.act(observation)
        observation = env.step(action)
        assert env.history[-1].valid
    return env.state.t


class OnlinePlannerAgentTests(unittest.TestCase):
    def test_known_and_or_plan_matches_environment_oracle(self):
        cases = [
            ([{"A", "B"}], None),
            ([{"A"}, {"B"}], None),
            ([{"A", "B"}, {"A"}], {"B": [{"A"}]}),
            ([{"B"}], {"B": [{"A", "seed"}]}),
        ]
        for goal_options, parent_options in cases:
            with self.subTest(goal_options=goal_options, parent_options=parent_options):
                env, positions, options = known_environment(goal_options, parent_options=parent_options)
                optimum = shortest_actual_environment_solution(env)
                agent = warm_agent(env, positions, options)
                self.assertEqual(run_known(agent, env), optimum)
                self.assertIn(env.goal_id, env.state.activated)
                self.assertEqual(agent.diagnostics["plan_searches"], 1)
                self.assertEqual(agent.diagnostics["cached_actions"], optimum - 1)
                self.assertGreater(agent.diagnostics["search_states_generated"], 0)
                self.assertNotIn("search_states", agent.diagnostics)
                self.assertEqual(sum(agent.memory.visit_count.values()), optimum)

    def test_local_activation_closure_handles_colocated_dependencies(self):
        env, positions, options = known_environment([{"B"}], parent_options={"B": [{"A"}]})
        positions["B"] = positions["A"]
        env.instance.discovery_map = {(0, 0): {"seed"}, (0, 1): {"A", "B"}, (2, 0): {"goal"}}
        optimum = shortest_actual_environment_solution(env)
        agent = warm_agent(env, positions, options)
        assert run_known(agent, env) == optimum
        assert env.goal_id in env.state.activated
        assert agent.diagnostics["plan_searches"] == 1


    def test_unknown_dependencies_and_missing_metadata_stay_unmet(self):
        for options in ({"goal": [{"unknown"}]}, {}):
            agent = OnlinePlannerAgent()
            agent.memory.discovered_at["goal"] = (1, 0)
            agent.memory.prerequisites.update(options)
            observation = Observation(
                t=0, position=(0, 0), neighbors=[(-1, 0), (1, 0)],
                newly_discovered_visible=[], newly_activated_visible=[],
                discovered_visible=["goal"], activated_visible=[], goal_visible="goal",
            )
            assert agent.act(observation) == Action.move((-1, 0))
            assert agent.diagnostics["search_states_generated"] == 0
            assert agent.diagnostics["fallback_actions"] == 1


    def test_cap_bounds_generated_states_and_fallback_finishes(self):
        env, positions, options = known_environment([{"A", "B"}])
        agent = warm_agent(env, positions, options, max_search_states=2)
        steps = run_known(agent, env)
        assert env.goal_id in env.state.activated
        assert agent.diagnostics["search_limit_hits"] > 0
        assert agent.diagnostics["peak_search_states"] <= 2
        assert agent.diagnostics["search_states_generated"] <= 2 * agent.diagnostics["plan_searches"]
        assert agent.diagnostics["search_expansions"] <= agent.diagnostics["search_states_generated"]
        assert agent.diagnostics["fallback_actions"] > 0
        assert sum(agent.memory.visit_count.values()) == steps


    def test_failed_search_is_not_repeated_on_ordinary_movement(self):
        env, positions, options = known_environment([{"unknown"}])
        agent = warm_agent(env, positions, options)
        first = observe(env)
        agent.act(first)
        second = deepcopy(first)
        second.position = (1, 0)
        second.neighbors = env.instance.grid.neighbors(second.position)
        second.t += 1
        agent.act(second)
        assert agent.diagnostics["plan_searches"] == 1
        assert agent.diagnostics["fallback_actions"] == 2


    def test_capped_search_is_not_repeated_on_ordinary_movement(self):
        env, positions, options = known_environment([{"A", "B"}])
        agent = warm_agent(env, positions, options, max_search_states=1)
        first = observe(env)
        agent.act(first)
        second = deepcopy(first)
        second.position = (1, 0)
        second.neighbors = env.instance.grid.neighbors(second.position)
        second.t += 1
        agent.act(second)
        assert agent.diagnostics["plan_searches"] == 1
        assert agent.diagnostics["search_limit_hits"] == 1


    def test_new_knowledge_and_unexpected_activation_invalidate_cached_plan(self):
        for change in ("edge", "prerequisite", "activation"):
            env, positions, options = known_environment([{"A", "B"}])
            agent = warm_agent(env, positions, options)
            observation = env.step(agent.act(observe(env)))
            if change == "edge":
                observation.neighbors.append((-1, 1))
            elif change == "prerequisite":
                observation.newly_discovered_prerequisites_visible["goal"] = [["A"]]
            else:
                observation.activated_visible.append("B")
            action = agent.act(observation)
            assert action.kind == "move"
            assert agent.diagnostics["plan_searches"] == 2
            assert agent.diagnostics["cached_actions"] == 0


    def test_reset_clears_cached_plan_memory_and_diagnostics(self):
        env, positions, options = known_environment([{"A", "B"}])
        agent = warm_agent(env, positions, options, max_search_states=77)
        agent.act(observe(env))
        assert agent.diagnostics["plan_searches"] == 1
        agent.reset()
        assert agent.memory.known_neighbors == {}
        assert agent.memory.discovered_at == {}
        assert not agent._plan
        assert agent._failed_search_key is None
        assert all(value == 0 for value in agent.diagnostics.values())
        assert "search_states" not in agent.diagnostics
        assert agent.max_search_states == 77



    def test_invalid_limits_rejected(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "positive integer"):
                OnlinePlannerAgent(max_search_states=limit)



from explo2_bench.baselines import BASELINE_NAMES, create_agent
from explo2_bench.baselines.sweep_then_exploit import SweepThenExploitAgent
from explo2_bench.baselines.utility import immediate_unlock_count


class PublicBaselineTests(unittest.TestCase):
    def test_all_eight_are_legal_deterministic_on_reset_and_ignore_private_extras(self):
        self.assertEqual(len(BASELINE_NAMES), 8)
        for name in BASELINE_NAMES:
            with self.subTest(name=name):
                env, _, _ = known_environment([{"A", "B"}])
                agent = create_agent(name, seed=31, max_search_states=20)
                one = env.run(agent)
                two = env.run(agent)
                self.assertEqual(one.to_dict(), two.to_dict())
                self.assertTrue(all(r.valid for r in one.records))
                clean = env.reset()
                dirty = deepcopy(clean)
                dirty.seed = 99
                dirty.hidden_budget = 1
                dirty.canary = "PRIVATE"
                dirty.feasibility_witness = [(2, 1)]
                dirty.open_cells = [(-9, -9)]
                dirty.state_positions_visible = {"goal": (-9, -9)}
                agent.reset()
                first = agent.act(clean)
                agent.reset()
                self.assertEqual(first, agent.act(dirty))

    def test_sweep_uses_bounded_cached_plans_and_delays_task_revisits(self):
        env, positions, options = known_environment([{"A", "B"}])
        for cap in (1, 1000):
            env, positions, options = known_environment([{"A", "B"}])
            agent = SweepThenExploitAgent(max_search_states=cap)
            cells = env.instance.grid.open_cells()
            agent.memory.known_neighbors = {c: set(env.instance.grid.neighbors(c)) for c in cells}
            agent.memory.visited.update(cells)
            agent.memory.discovered_at.update(positions)
            agent.memory.prerequisites.update(deepcopy(options))
            steps = run_known(agent, env)
            self.assertTrue(env.state.done)
            self.assertIn("goal", env.state.activated)
            self.assertLessEqual(agent.diagnostics["peak_search_states"], cap)
            self.assertEqual(sum(agent.memory.visit_count.values()), steps)
            if cap == 1:
                self.assertGreater(agent.diagnostics["search_limit_hits"], 0)
            else:
                self.assertEqual(agent.diagnostics["plan_searches"], 1)
                self.assertEqual(agent.diagnostics["cached_actions"], steps - 1)
        # A ready goal nearby does not interrupt the exploration phase.
        env, positions, options = known_environment([{"seed"}])
        agent = SweepThenExploitAgent()
        agent.memory.discovered_at.update(positions)
        agent.memory.prerequisites.update(options)
        first = agent.act(observe(env))
        self.assertEqual(first, Action.move((0, 1)))
        self.assertEqual(agent._phase, "sweep")

    def test_immediate_unlock_count_respects_and_or_and_excludes_already_ready(self):
        env, positions, options = known_environment([{"A", "B"}])
        agent = create_agent("dependency_greedy")
        agent.memory.prerequisites = {
            "A": [{"seed"}], "B": [{"seed"}],
            "and": [{"A", "B"}], "or": [{"A"}, {"B"}],
            "ready": [{"seed"}], "multi": [{"A"}, {"A", "B"}],
        }
        obs = observe(env)
        obs.discovered_visible = ["seed", "A", "B", "and", "or", "ready", "multi", "unknown"]
        self.assertEqual(immediate_unlock_count(agent.memory, obs, "A"), 2)
        obs.activated_visible.append("B")
        self.assertEqual(immediate_unlock_count(agent.memory, obs, "A"), 2)

    def test_new_policies_prioritize_ready_goal_and_keep_target_until_ready_goal(self):
        for name in ("dependency_greedy", "balanced_utility"):
            agent = create_agent(name)
            env, positions, options = known_environment([{"A", "B"}])
            cells = env.instance.grid.open_cells()
            agent.memory.known_neighbors = {c: set(env.instance.grid.neighbors(c)) for c in cells}
            agent.memory.visited.update(cells)
            agent.memory.discovered_at.update(positions)
            agent.memory.prerequisites.update(options)
            obs = observe(env)
            agent.act(obs)
            self.assertEqual(agent._target, ("task", "A"))
            # Recompute scores while still en route; retain the original target.
            altered = deepcopy(obs)
            altered.position = (1, 0)
            altered.neighbors = env.instance.grid.neighbors(altered.position)
            agent.act(altered)
            self.assertEqual(agent._target, ("task", "A"))
            altered.activated_visible += ["A", "B"]
            self.assertEqual(agent.act(altered), Action.move((2, 0)))
            self.assertEqual(agent._target, ("task", "goal"))
