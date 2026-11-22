#!/usr/bin/env python3
"""Regenerate and freeze complete maps from the committed seeds and settings.

Run from the repository root after installing the package. Artifacts have
deterministic encoding.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

from explo2_bench.data import PROJECT_CANARY, RELEASE_CANARIES, canonical_json, canonical_sha256, make_record, validate_record
from explo2_bench.generation import generate, settings


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(version: str, output: Path) -> dict:
    destination = output / version
    with (destination / "seeds.csv").open(newline="", encoding="utf-8") as handle:
        seeds = list(csv.DictReader(handle))
    recorded_settings = json.loads((destination / "settings.json").read_text())
    if canonical_sha256(recorded_settings) != canonical_sha256(settings(version)):
        raise ValueError("Generation settings differ from recorded release settings")
    ids = [row["map_id"] for row in seeds]
    if len(ids) != len(set(ids)):
        raise ValueError("Seed inventory has duplicate map IDs")
    expected_count = {"v2": 250, "v3": 8100}[version]
    if len(seeds) != expected_count:
        raise ValueError(f"Expected {expected_count} seeds for {version}")
    splits, matched = {}, 0
    target = destination / "maps.jsonl.gz"
    temporary = destination / "maps.jsonl.gz.tmp"
    started = time.monotonic()
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for index, row in enumerate(seeds):
                episode = generate(version, row["exploration_profile"], row["dag_profile"], int(row["seed"]))
                record = make_record(version, episode, split=row["split"] or None)
                if record["map_id"] != row["map_id"]:
                    raise ValueError("Map ID differs from seed inventory")
                if row["source_episode_sha256"]:
                    if record["episode_sha256"] != row["source_episode_sha256"]:
                        raise ValueError(f"Source episode hash differs for {record['map_id']}")
                    matched += 1
                validate_record(record)
                handle.write((canonical_json(record) + "\n").encode("utf-8"))
                key = record["split"] or "unassigned"
                splits[key] = splits.get(key, 0) + 1
                if (index + 1) % 500 == 0:
                    print(f"{version}: {index + 1}/{len(seeds)} maps", flush=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1, "version": version, "release": f"{version}-r1", "map_count": len(seeds),
        "setting_count": len({(row["exploration_profile"], row["dag_profile"]) for row in seeds}),
        "split_counts": splits, "native_action_budget_multiplier": 3 if version == "v2" else 2,
        "episode_checksum_encoding": "SHA256 of UTF-8 JSON with sort_keys=True, separators=(',', ':'), allow_nan=False",
        "map_file_format": "gzip-compressed JSON Lines; one complete common-schema record per line",
        "canary": {"project": PROJECT_CANARY, "release": RELEASE_CANARIES[version]},
        "source_episode_hashes_verified": matched,
        "generation_python_version": platform.python_version(),
        "files_sha256": {name: digest(destination / name) for name in ("maps.jsonl.gz", "seeds.csv", "settings.json", "provenance.json")},
    }
    write_json(destination / "manifest.json", manifest)
    print(f"{version}: frozen {len(seeds)} maps in {time.monotonic()-started:.1f}s; {matched} source hashes verified", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks"),
                        help="Directory containing release inputs and receiving regenerated maps; default: benchmarks")
    parser.add_argument("--version", choices=("v2", "v3", "all"), default="all",
                        help="Benchmark collection to regenerate; default: all")
    args = parser.parse_args()
    for version in ("v2", "v3") if args.version == "all" else (args.version,):
        freeze(version, args.output)


if __name__ == "__main__":
    main()
