"""Evaluator-only scoring of completed public-interface episodes."""

from __future__ import annotations

from typing import Any

from symbolic_environment.metrics import compute_metrics
from ._core.types import Trajectory
from .environment import BenchmarkEnv


def score_episode(env: BenchmarkEnv) -> dict[str, Any]:
    """Return aggregate benchmark measurements after an episode has ended.

    Scoring uses the evaluator's full map, but returns no map, seed, witness,
    task identity, or trajectory. Scores include the final action allowance and
    global cell/task counts: keep them outside the agent observation stream.
    Rates and eligible-step denominators match the baseline evaluation runner.
    A zero denominator gives ``None``. Policy/model timing is the caller's
    responsibility and is not inferred from environment history.
    """
    if not isinstance(env, BenchmarkEnv):
        raise TypeError("score_episode expects a BenchmarkEnv returned by make_env")
    core = env._env
    if not core.history:
        raise ValueError("Reset and complete an episode before scoring")
    if not core.state.done:
        raise ValueError("Only completed episodes can be scored")
    trajectory = Trajectory(records=list(core.history), success=env.success,
                            terminated_reason=core.state.terminated_reason,
                            total_steps=core.state.t)
    metrics = compute_metrics(core.instance, trajectory)
    exploration_eligible = metrics.exploration_steps
    exploitation_eligible = metrics.exploitation_steps
    union_errors = sum(bool(error.exploration_error or error.exploitation_error)
                       for error in metrics.step_errors)
    visited_cells = len({record.position for record in trajectory.records})
    open_cells = len(core.instance.grid.open_cells())
    task_nodes = len(core.instance.task_graph.nodes)
    goal_step = next((record.t for record in trajectory.records if core.goal_id in record.activated), None)
    return {
        "success": trajectory.success, "terminated_reason": trajectory.terminated_reason,
        "steps": trajectory.total_steps, "action_budget": core.instance.hidden_budget,
        "goal_activation_step": goal_step,
        "open_cells": open_cells, "task_nodes": task_nodes,
        "discovered_count": metrics.discovered_count, "activated_count": metrics.activated_count,
        "discovered_fraction": metrics.discovered_count / task_nodes if task_nodes else None,
        "activated_fraction": metrics.activated_count / task_nodes if task_nodes else None,
        "visited_cells": visited_cells, "coverage": visited_cells / open_cells if open_cells else None,
        "invalid_actions": sum(not record.valid for record in trajectory.records if record.action != "START"),
        "exploration_errors": metrics.total_exploration_errors,
        "exploitation_errors": metrics.total_exploitation_errors, "union_errors": union_errors,
        "exploration_eligible_steps": exploration_eligible,
        "exploitation_eligible_steps": exploitation_eligible,
        "exploration_error_rate": metrics.total_exploration_errors / exploration_eligible if exploration_eligible else None,
        "exploitation_error_rate": metrics.total_exploitation_errors / exploitation_eligible if exploitation_eligible else None,
        "union_error_rate": union_errors / trajectory.total_steps if trajectory.total_steps else None,
        "explore_only_steps": metrics.explore_only_steps, "exploit_only_steps": metrics.exploit_only_steps,
        "mixed_steps": metrics.mixed_steps, "mixed_goal_steps": metrics.mixed_goal_steps,
        "mixed_non_goal_steps": metrics.mixed_non_goal_steps,
    }


__all__ = ["score_episode"]
