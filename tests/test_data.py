"""Data integrity, reference identity, and deterministic generation checks."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import unittest

from explo2_bench.data import (
    benchmark_root, canonical_json, canonical_sha256, episode_from_record,
    load_benchmark, make_record, validate_benchmark, validate_record,
)
from explo2_bench.generation import generate


class ReleaseDataTests(unittest.TestCase):
    def test_inventory_and_integrity(self):
        for version, count, splits in (
            ("v2", 250, {"test": 250}),
            ("v3", 8100, {"train": 180, "validation": 60, "test": 360, "unassigned": 7500}),
        ):
            result = validate_benchmark(version)
            self.assertEqual(result["map_count"], count)
            self.assertEqual(result["split_counts"], splits)
            self.assertTrue(result["valid"])

    def test_v2_entire_release_is_test_split(self):
        expected_ids = {
            f"v2-{exploration}-{dag}-s{seed}"
            for exploration in ("higher", "high", "medium", "low", "lower")
            for dag in ("harder", "hard", "medium", "easy", "easier")
            for seed in range(10)
        }
        records = list(load_benchmark("v2"))
        test_records = list(load_benchmark("v2", split="test"))
        self.assertEqual(len(records), 250)
        self.assertEqual(len(test_records), 250)
        self.assertEqual({record["map_id"] for record in records}, expected_ids)
        self.assertEqual({record["map_id"] for record in test_records}, expected_ids)
        self.assertEqual({record["split"] for record in records}, {"test"})
        for split in ("train", "validation", "unassigned"):
            with self.subTest(split=split):
                self.assertEqual(list(load_benchmark("v2", split=split)), [])

        with (benchmark_root() / "v2/seeds.csv").open(newline="") as handle:
            seeds = list(csv.DictReader(handle))
        self.assertEqual(len(seeds), 250)
        self.assertEqual({row["map_id"] for row in seeds}, expected_ids)
        self.assertEqual({row["split"] for row in seeds}, {"test"})

    def test_v3_reference_hashes_and_split_membership(self):
        with (benchmark_root() / "v3/seeds.csv").open(newline="") as handle:
            source = {row["map_id"]: row for row in csv.DictReader(handle)}
        memberships = {name: set() for name in ("train", "validation", "test", "unassigned")}
        for record in load_benchmark("v3"):
            self.assertEqual(record["episode_sha256"], source[record["map_id"]]["source_episode_sha256"])
            self.assertEqual(record["split"] or "", source[record["map_id"]]["split"])
            memberships[record["split"] or "unassigned"].add(record["map_id"])
        self.assertEqual({key: len(value) for key, value in memberships.items()},
                         {"train": 180, "validation": 60, "test": 360, "unassigned": 7500})
        self.assertFalse(memberships["train"] & memberships["test"])
        self.assertEqual({record["map_id"] for record in load_benchmark("v3", split="test")}, memberships["test"])

    def test_generation_roundtrip_and_reference_hash(self):
        expected_first_v3 = "7770a64a1094c3d11f9c5f8e52946ce2e9e478efed969450d8189d26cb1098eb"
        first = next(load_benchmark("v3"))
        self.assertEqual(first["episode_sha256"], expected_first_v3)
        for version, exploration, dag, seed in (("v2", "higher", "harder", 8),
                                                ("v2", "lower", "easier", 0),
                                                ("v3", "x-lowest", "x-easiest", 3470614351),
                                                ("v3", "x-highest", "x-hardest", 8)):
            episode = generate(version, exploration, dag, seed)
            record = json.loads(canonical_json(make_record(version, episode)))
            restored = episode_from_record(record)
            self.assertEqual(canonical_json(restored.to_dict()), canonical_json(episode.to_dict()))
            self.assertTrue(validate_record(record, regenerate=True)["regenerated"])

    def test_v3_regeneration_matches_frozen_hashes_for_all_exploration_profiles(self):
        # Frozen hashes catch cross-version float differences that roundtrips miss.
        representatives = {}
        with (benchmark_root() / "v3/seeds.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dag_profile"] == "x-easiest":
                    representatives.setdefault(row["exploration_profile"], row)
        self.assertEqual(len(representatives), 9)
        for exploration, row in representatives.items():
            with self.subTest(exploration=exploration):
                episode = generate("v3", exploration, row["dag_profile"], int(row["seed"]))
                self.assertEqual(canonical_sha256(episode.to_dict()), row["source_episode_sha256"])

    def test_reject_modified_or_mismatched_record(self):
        original = next(load_benchmark("v2"))
        for mutate in (lambda row: row["episode"].update(hidden_budget=1),
                       lambda row: row["generation"].update(seed=999),
                       lambda row: row["canary"].update(project="removed"),
                       lambda row: row.update(version="v3")):
            record = deepcopy(original)
            mutate(record)
            with self.assertRaises((ValueError, TypeError, KeyError)):
                validate_record(record)

    def test_reject_disconnected_grid_even_with_recomputed_checksum(self):
        record = deepcopy(next(load_benchmark("v2")))
        # Enlarge the grid and isolate its old cells with a complete wall column.
        grid = record["episode"]["grid"]
        previous_width = grid["width"]
        grid["width"] += 2
        grid["blocked"].extend([[previous_width, y] for y in range(grid["height"])])
        record["episode_sha256"] = canonical_sha256(record["episode"])
        with self.assertRaisesRegex(ValueError, "connected"):
            validate_record(record)

    def test_reject_unknown_versions_and_invalid_seeds(self):
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                generate("v2", "higher", "harder", seed)
        with self.assertRaises(ValueError):
            list(load_benchmark("v1"))
        with self.assertRaises(ValueError):
            list(load_benchmark("v3", split="unknown"))

    def test_reject_boolean_coordinate_even_with_valid_checksum(self):
        record = deepcopy(next(load_benchmark("v2")))
        record["episode"]["grid"]["start"][0] = False
        record["episode_sha256"] = canonical_sha256(record["episode"])
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_record(record)


if __name__ == "__main__":
    unittest.main()
