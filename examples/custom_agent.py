"""Run an observation-only agent; no model provider or network is needed.

From a source checkout: python examples/custom_agent.py --version v2
"""
import argparse
from collections import Counter

from explo2_bench import load_benchmark, make_env, score_episode


class LeastVisitedAgent:
    def __init__(self):
        self.visits = Counter()

    def __call__(self, observation):
        x, y = observation["position"]
        self.visits[(x, y)] += 1
        offsets = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
        legal = observation["legal_moves"]
        move = min(legal, key=lambda direction: (
            self.visits[(x + offsets[direction][0], y + offsets[direction][1])], direction
        ))
        return {"move": move}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    record = next(load_benchmark(args.version, root=args.root))
    env = make_env(record, alpha=2)
    agent = LeastVisitedAgent()
    observation = env.reset()
    while not env.done:
        result = env.step(agent(observation))
        observation = result.observation
    print(score_episode(env))


if __name__ == "__main__":
    main()
