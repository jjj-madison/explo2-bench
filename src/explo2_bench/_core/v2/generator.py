from __future__ import annotations

from ..generator import _generate_constructive_episode
from ..types import ExplorationLevel as _ConstructiveExplorationLevel
from ..types import TaskDAGDifficulty as _ConstructiveTaskDAGDifficulty
from .config import get_knobs
from .types import (
    CONFIGURATION_VERSION,
    ConstructiveKnobs,
    EpisodeInstance,
    ExplorationLevel,
    TaskDAGDifficulty,
)


def generate_episode(
    task_dag_difficulty: TaskDAGDifficulty,
    exploration_level: ExplorationLevel,
    seed: int,
    knobs: ConstructiveKnobs | None = None,
) -> EpisodeInstance:
    if not isinstance(task_dag_difficulty, TaskDAGDifficulty):
        raise TypeError(
            "task_dag_difficulty must be a explo2_bench._core.v2."
            "TaskDAGDifficulty"
        )
    if not isinstance(exploration_level, ExplorationLevel):
        raise TypeError(
            "exploration_level must be a explo2_bench._core.v2.ExplorationLevel"
        )
    resolved_knobs = knobs or get_knobs(task_dag_difficulty, exploration_level)
    if resolved_knobs.alpha_budget < 0:
        raise ValueError(
            f"alpha_budget must be >= 0, got {resolved_knobs.alpha_budget}"
        )

    base_instance = _generate_constructive_episode(
        task_dag_difficulty=_ConstructiveTaskDAGDifficulty.CUSTOM,
        exploration_level=_ConstructiveExplorationLevel.CUSTOM,
        seed=seed,
        knobs=resolved_knobs,
    )

    episode_id = (
        f"exploration_level_{exploration_level.value}-"
        f"task_dag_{task_dag_difficulty.value}-s{seed:04d}"
    )
    metadata = dict(base_instance.metadata)
    metadata.update(
        {
            "configuration_version": CONFIGURATION_VERSION,
            "exploration_level": exploration_level.value,
            "task_dag_difficulty": task_dag_difficulty.value,
        }
    )

    return EpisodeInstance(
        episode_id=episode_id,
        exploration_level=exploration_level,
        task_dag_difficulty=task_dag_difficulty,
        seed=seed,
        knobs=resolved_knobs,
        grid=base_instance.grid,
        task_graph=base_instance.task_graph,
        discovery_map=base_instance.discovery_map,
        required_nodes=base_instance.required_nodes,
        hidden_budget=int(
            len(base_instance.grid.open_cells()) * resolved_knobs.alpha_budget
        ),
        metadata=metadata,
    )


__all__ = ["generate_episode"]
