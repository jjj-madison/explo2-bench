from __future__ import annotations

from dataclasses import asdict, dataclass

from .types import ConstructiveKnobs, ExplorationLevel, TaskDAGDifficulty


@dataclass(frozen=True)
class DefaultGenerationConfig:
    alpha_budget: float
    grid_aspect_ratio: float


@dataclass(frozen=True)
class TaskDAGPreset:
    num_nodes: int
    max_nodes_per_layer: int
    option_count_probs: dict[int, float]
    dependency_count_probs: dict[int, float]
    goal_dependency_count_probs: dict[int, float]
    parent_depth_bias: float


@dataclass(frozen=True)
class ExplorationPreset:
    dag_density_over_grid: float
    path_width_probs: dict[int, float]


def _uniform_count_probs(max_count: int) -> dict[int, float]:
    weight = 1.0 / max_count
    return {count: weight for count in range(1, max_count + 1)}


DEFAULT_GENERATION_CONFIG = DefaultGenerationConfig(
    alpha_budget=3.0,
    grid_aspect_ratio=1.0,
)


GENERATED_TASK_DAG_DIFFICULTIES: tuple[TaskDAGDifficulty, ...] = (
    TaskDAGDifficulty.HARDER,
    TaskDAGDifficulty.HARD,
    TaskDAGDifficulty.MEDIUM,
    TaskDAGDifficulty.EASY,
    TaskDAGDifficulty.EASIER,
)

TASK_DAG_CONFIGS: dict[TaskDAGDifficulty, TaskDAGPreset] = {
    TaskDAGDifficulty.HARDER: TaskDAGPreset(
        num_nodes=10,
        max_nodes_per_layer=3,
        option_count_probs={1: 0.4, 2: 0.6},
        dependency_count_probs=_uniform_count_probs(4),
        goal_dependency_count_probs=_uniform_count_probs(4),
        parent_depth_bias=1.0,
    ),
    TaskDAGDifficulty.HARD: TaskDAGPreset(
        num_nodes=8,
        max_nodes_per_layer=3,
        option_count_probs={1: 0.6, 2: 0.4},
        dependency_count_probs=_uniform_count_probs(3),
        goal_dependency_count_probs=_uniform_count_probs(3),
        parent_depth_bias=1.0,
    ),
    TaskDAGDifficulty.MEDIUM: TaskDAGPreset(
        num_nodes=6,
        max_nodes_per_layer=3,
        option_count_probs={1: 0.8, 2: 0.2},
        dependency_count_probs=_uniform_count_probs(2),
        goal_dependency_count_probs=_uniform_count_probs(2),
        parent_depth_bias=1.0,
    ),
    TaskDAGDifficulty.EASY: TaskDAGPreset(
        num_nodes=4,
        max_nodes_per_layer=3,
        option_count_probs={1: 1.0},
        dependency_count_probs=_uniform_count_probs(2),
        goal_dependency_count_probs=_uniform_count_probs(2),
        parent_depth_bias=1.0,
    ),
    TaskDAGDifficulty.EASIER: TaskDAGPreset(
        num_nodes=2,
        max_nodes_per_layer=1,
        option_count_probs={1: 1.0},
        dependency_count_probs={1: 1.0},
        goal_dependency_count_probs={1: 1.0},
        parent_depth_bias=1.0,
    ),
}


GENERATED_EXPLORATION_LEVELS: tuple[ExplorationLevel, ...] = (
    ExplorationLevel.HIGHER,
    ExplorationLevel.HIGH,
    ExplorationLevel.MEDIUM,
    ExplorationLevel.LOW,
    ExplorationLevel.LOWER,
)

EXPLORATION_CONFIGS: dict[ExplorationLevel, ExplorationPreset] = {
    ExplorationLevel.HIGHER: ExplorationPreset(
        dag_density_over_grid=0.05,
        path_width_probs={2: 0.4, 3: 0.6},
    ),
    ExplorationLevel.HIGH: ExplorationPreset(
        dag_density_over_grid=0.075,
        path_width_probs={1: 0.3, 2: 0.45, 3: 0.25},
    ),
    ExplorationLevel.MEDIUM: ExplorationPreset(
        dag_density_over_grid=0.1,
        path_width_probs={1: 0.5, 2: 0.5},
    ),
    ExplorationLevel.LOW: ExplorationPreset(
        dag_density_over_grid=0.15,
        path_width_probs={1: 1.0},
    ),
    ExplorationLevel.LOWER: ExplorationPreset(
        dag_density_over_grid=0.4,
        path_width_probs={1: 1.0},
    ),
}


def _build_knobs(
    task_dag_difficulty: TaskDAGDifficulty,
    exploration_level: ExplorationLevel,
) -> ConstructiveKnobs:
    return ConstructiveKnobs(
        **asdict(DEFAULT_GENERATION_CONFIG),
        **asdict(TASK_DAG_CONFIGS[task_dag_difficulty]),
        **asdict(EXPLORATION_CONFIGS[exploration_level]),
    )


PRESET_KNOBS: dict[tuple[TaskDAGDifficulty, ExplorationLevel], ConstructiveKnobs] = {
    (task_dag_difficulty, exploration_level): _build_knobs(
        task_dag_difficulty,
        exploration_level,
    )
    for task_dag_difficulty in GENERATED_TASK_DAG_DIFFICULTIES
    for exploration_level in GENERATED_EXPLORATION_LEVELS
}


def get_knobs(
    task_dag_difficulty: TaskDAGDifficulty,
    exploration_level: ExplorationLevel,
) -> ConstructiveKnobs:
    if not isinstance(task_dag_difficulty, TaskDAGDifficulty):
        raise TypeError(
            "task_dag_difficulty must be a explo2_bench._core.v2."
            "TaskDAGDifficulty"
        )
    if not isinstance(exploration_level, ExplorationLevel):
        raise TypeError(
            "exploration_level must be a explo2_bench._core.v2.ExplorationLevel"
        )
    key = (task_dag_difficulty, exploration_level)
    if key not in PRESET_KNOBS:
        raise KeyError(
            "No v2 preset knobs for "
            f"task_dag_difficulty={task_dag_difficulty.value}, "
            f"exploration_level={exploration_level.value}"
        )
    return PRESET_KNOBS[key]


__all__ = [
    "DEFAULT_GENERATION_CONFIG",
    "EXPLORATION_CONFIGS",
    "GENERATED_EXPLORATION_LEVELS",
    "GENERATED_TASK_DAG_DIFFICULTIES",
    "PRESET_KNOBS",
    "TASK_DAG_CONFIGS",
    "DefaultGenerationConfig",
    "ExplorationPreset",
    "TaskDAGPreset",
    "get_knobs",
]
