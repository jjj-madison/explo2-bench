#!/usr/bin/env python3
"""Check a wheel installation, including bundled data, scoring, and rendering.

Run with the installed environment's Python from outside the source checkout.
The analysis extra must be installed. All generated figures are temporary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import explo2_bench
from explo2_bench import load_benchmark, make_env, score_episode
from explo2_bench.data import benchmark_root, validate_record
from explo2_bench.rendering import render_map, render_task_graph
from explo2_bench.upstream import verify_metric_dependency


def main() -> None:
    environment = Path(sys.prefix).resolve()
    package = Path(explo2_bench.__file__).resolve()
    data = benchmark_root().resolve()
    for label, path in (("Package", package), ("Benchmark data", data)):
        if not path.is_relative_to(environment):
            raise RuntimeError(f"{label} must come from the installed environment: {path}")
    verify_metric_dependency()

    checked = []
    with tempfile.TemporaryDirectory(prefix="explo2-installation-") as directory:
        for version, expected_count in (("v2", 250), ("v3", 8100)):
            manifest = json.loads((data / version / "manifest.json").read_text(encoding="utf-8"))
            if manifest["map_count"] != expected_count:
                raise RuntimeError(f"Unexpected map count for {version}: {manifest['map_count']}")
            for filename, expected in manifest["files_sha256"].items():
                digest = hashlib.sha256()
                with (data / version / filename).open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(f"Installed benchmark checksum differs: {version}/{filename}")
            command = Path(sys.executable).with_name("explo2-bench.exe" if sys.platform == "win32" else "explo2-bench")
            result = subprocess.run([str(command), "info", version], check=True, capture_output=True, text=True)
            if json.loads(result.stdout) != manifest:
                raise RuntimeError(f"Console command returned different benchmark information: {version}")
            records = load_benchmark(version)
            try:
                record = next(records)
            finally:
                records.close()
            validate_record(record, regenerate=True)
            env = make_env(record, action_budget=2)
            observation = env.reset()
            while not env.done:
                result = env.step({"move": observation["legal_moves"][0]})
                observation = result.observation
            scores = score_episode(env)
            if scores["invalid_actions"] or not 0 <= scores["steps"] <= 2:
                raise RuntimeError(f"Unexpected environment or scoring result: {scores}")

            for name, render in (("map", render_map), ("dag", render_task_graph)):
                output = Path(directory) / f"{version}_{name}"
                render(record, output)
                for suffix in (".png", ".pdf"):
                    figure = output.with_suffix(suffix)
                    if not figure.is_file() or figure.stat().st_size == 0:
                        raise RuntimeError(f"Figure was not generated: {figure}")
            checked.append({"version": version, "map_count": expected_count,
                            "map_id": record["map_id"], "steps": scores["steps"]})
    print(json.dumps({"package": str(package), "data": str(data), "checked": checked}, indent=2))


if __name__ == "__main__":
    main()
