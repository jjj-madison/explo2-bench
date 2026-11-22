import random

from .._core.types import Action, Observation


class RandomWalkAgent:
    """Choose uniformly among legal neighbors without using task or map memory."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng.seed(self._seed)

    def act(self, observation: Observation) -> Action:
        if observation.neighbors:
            return Action.move(self._rng.choice(sorted(observation.neighbors)))
        raise RuntimeError(f"No valid MOVE action available at position {observation.position}.")
