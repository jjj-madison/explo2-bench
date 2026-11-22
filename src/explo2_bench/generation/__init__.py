"""Reproduce v2/v3 maps from the released generation settings and seeds."""

from functools import lru_cache

from .._core.v2 import config as v2_config
from .._core.v2.generator import generate_episode as generate_v2
from .._core.v2.types import ExplorationLevel as V2Exploration
from .._core.v2.types import TaskDAGDifficulty as V2Difficulty
from .._core.v3.config import knobs_for, load_settings
from .._core.v3.generator import generate_episode as generate_v3


@lru_cache(maxsize=1)
def _v3_settings():
    return load_settings()


def generate(version: str, exploration: str, dag: str, seed: int):
    """Generate a native episode; no hidden benchmark data is needed.

    Generated episodes use budgets of 3x open cells for v2 and 2x for v3.
    Environment runners configure evaluation budgets.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if version == "v2":
        return generate_v2(V2Difficulty(dag), V2Exploration(exploration), seed)
    if version == "v3":
        return generate_v3(dag, exploration, seed, knobs_for(_v3_settings(), exploration, dag))
    raise ValueError("version must be 'v2' or 'v3'")


def settings(version: str) -> dict:
    """Return the exact expanded preset settings used by :func:`generate`."""
    if version == "v3":
        from copy import deepcopy
        return deepcopy(_v3_settings())
    if version != "v2":
        raise ValueError("version must be 'v2' or 'v3'")
    from dataclasses import asdict
    return {
        "configuration_version": "v2",
        "preset_count": 25,
        "axis_order": {
            "exploration_levels": [level.value for level in v2_config.GENERATED_EXPLORATION_LEVELS],
            "task_dag_difficulties": [level.value for level in v2_config.GENERATED_TASK_DAG_DIFFICULTIES],
        },
        "defaults": asdict(v2_config.DEFAULT_GENERATION_CONFIG),
        "exploration_levels": {key.value: asdict(value) for key, value in v2_config.EXPLORATION_CONFIGS.items()},
        "task_dag_difficulties": {key.value: asdict(value) for key, value in v2_config.TASK_DAG_CONFIGS.items()},
        "presets": [
            {"exploration_level": exploration.value, "task_dag_difficulty": dag.value,
             "knobs": v2_config.get_knobs(dag, exploration).to_dict()}
            for exploration in v2_config.GENERATED_EXPLORATION_LEVELS
            for dag in v2_config.GENERATED_TASK_DAG_DIFFICULTIES
        ],
    }


__all__ = ["generate", "settings"]
