"""Replace model_callback with your LM client; this demo makes no API calls.

The callback receives only public observations. Keep conversation history in
your callback and report its history/context policy alongside evaluation results.
For the core Agent protocol, wrap the same callback in LMCallableAgent.
"""
import argparse
import json

from explo2_bench import SYSTEM_PROMPT, load_benchmark, make_env, observation_to_text, score_episode


def model_callback(observation):
    # Example request construction. Replace the response below with your client:
    # response = client(messages=[system, user], response_schema=ACTION_SCHEMA)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_to_text(observation)},
    ]
    del messages  # The runnable example uses a deterministic dummy response.
    return json.dumps({"move": observation["legal_moves"][0]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    record = next(load_benchmark(args.version, root=args.root))
    env = make_env(record, alpha=2)
    observation = env.reset()
    while not env.done:
        result = env.step(model_callback(observation))
        observation = result.observation
    # Evaluator-only: do not add aggregate scores to the agent's observations.
    print({"callback": "dummy; no LM called", "scores": score_episode(env)})


if __name__ == "__main__":
    main()
