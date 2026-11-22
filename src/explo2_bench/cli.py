"""Inspect benchmark data, validate maps, or play an episode interactively."""
import argparse
import json
from .data import load_benchmark, validate_benchmark


def main():
    parser = argparse.ArgumentParser(prog="explo2-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("info", "validate", "play"):
        child = sub.add_parser(command, help={
            "info": "Show the benchmark manifest, including counts and checksums",
            "validate": "Check benchmark files and map consistency",
            "play": "Play the first map in the selected benchmark using JSON actions",
        }[command])
        child.add_argument("version", choices=["v2", "v3"], help="Benchmark collection")
        child.add_argument("--root", help="Repository or benchmarks directory; default: automatic data lookup")
        if command == "validate":
            child.add_argument("--regenerate", action="store_true",
                               help="Regenerate every map and compare its episode checksum")
        if command == "play":
            child.add_argument("--alpha", type=float, default=2,
                               help="Action allowance multiplier: floor(alpha × open cells); default: 2")
    args = parser.parse_args()
    if args.command == "info":
        from .data import benchmark_root
        print((benchmark_root(args.root) / args.version / "manifest.json").read_text())
    elif args.command == "validate":
        print(json.dumps(validate_benchmark(args.version, root=args.root, regenerate=args.regenerate), indent=2))
    else:
        from .environment import make_env, observation_to_text, SYSTEM_PROMPT
        env = make_env(next(load_benchmark(args.version, root=args.root)), alpha=args.alpha)
        print(SYSTEM_PROMPT)
        print(observation_to_text(env.reset()))
        while not env.done:
            result = env.step(input('Action JSON> '))
            print(observation_to_text(result.observation))
        print(json.dumps({"success": env.success}))
