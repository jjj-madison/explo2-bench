"""Analytic fixtures verify graph semantics, geometry, and equivalence classes."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from explo2_bench._core.types import GridSpec, TaskGraph, TaskNode
from explo2_bench.analysis import maps
from explo2_bench.analysis.diversity import DAGRegistry


def example_episode():
    nodes = {
        "a": TaskNode("a", "Apple", [set()], 88),
        "b": TaskNode("b", "Pear", [set()], 99),
        "g": TaskNode("g", "Goal", [{"a", "b"}, {"a"}], 100),
    }
    return SimpleNamespace(grid=GridSpec(2, 2, set(), (0, 0)),
                           task_graph=TaskGraph(nodes, "g"), required_nodes=set(nodes),
                           discovery_map={(1, 0): {"a"}, (0, 1): {"b"}, (1, 1): {"g"}},
                           hidden_budget=8, metadata={})


def example_record():
    return {"map_id": "fixture", "version": "v2", "episode_sha256": "fixture-hash",
            "generation": {"seed": 1, "exploration_profile": "low", "dag_profile": "easy"}}


def test_structural_geometry_and_and_or_semantics(monkeypatch):
    episode = example_episode()
    monkeypatch.setattr(maps, "episode_from_record", lambda record: episode)
    row = maps.analyze_map(example_record())
    assert (row["open_cells"], row["grid_edges"], row["cycle_rank"]) == (4, 4, 1)
    assert row["connected_components"] == 1
    assert row["dead_ends"] == row["junctions"] == 0
    assert row["task_nodes"] == 3 and row["primitive_tasks"] == 2
    assert row["dependency_edges"] == 2
    assert row["prerequisite_groups"] == 4
    assert row["and_groups"] == 1 and row["or_tasks"] == 1
    assert row["dag_depth"] == 1 and row["layer_widths"] == [2, 1]
    assert row["start_goal_distance"] == 2
    assert row["task_pair_distance_mean"] == pytest.approx(4 / 3)
    assert row["task_pair_distance_count"] == 3
    assert row["start_task_distance_mean"] == pytest.approx(4 / 3)
    assert row["unreachable_task_pairs"] == 0


def test_distances_report_disconnection_without_zero_imputation(monkeypatch):
    episode = example_episode()
    episode.grid = GridSpec(3, 2, {(1, 0), (1, 1)}, (0, 0))
    episode.discovery_map = {(0, 1): {"a"}, (2, 0): {"b"}, (2, 1): {"g"}}
    monkeypatch.setattr(maps, "episode_from_record", lambda record: episode)
    row = maps.analyze_map(example_record())
    assert row["connected_components"] == 2
    assert row["cycle_rank"] == 0
    assert row["start_goal_distance"] is None
    assert row["unreachable_tasks"] == 2
    assert row["task_pair_distance_count"] == 1
    assert row["unreachable_task_pairs"] == 2
    assert row["task_pair_distance_mean"] == 1


def test_budget_and_seed_metadata_do_not_change_structural_identity(monkeypatch):
    episode = example_episode()
    monkeypatch.setattr(maps, "episode_from_record", lambda record: episode)
    before = maps.analyze_map(example_record())
    episode.hidden_budget = 10_000
    episode.metadata = {"seed": 19, "witness": ["private"]}
    assert maps.analyze_map(example_record()) == before
    changed = deepcopy(episode)
    changed.grid.start = (1, 0)
    assert maps.fingerprints(episode)["gameplay_sha256"] != maps.fingerprints(changed)["gameplay_sha256"]
    assert maps.fingerprints(episode)["wall_layout_sha256"] == maps.fingerprints(changed)["wall_layout_sha256"]


def test_cycles_are_rejected_instead_of_using_stored_depth(monkeypatch):
    episode = example_episode()
    episode.task_graph.nodes["a"] = TaskNode("a", "Apple", [{"g"}], 0)
    monkeypatch.setattr(maps, "episode_from_record", lambda record: episode)
    with pytest.raises(ValueError, match="cycle"):
        maps.analyze_map(example_record())


def test_exact_dag_classes_ignore_names_but_preserve_alternatives_and_goal():
    graph = example_episode().task_graph
    registry = DAGRegistry()
    first = registry.add(graph)
    rename = {"a": "x", "b": "y", "g": "z"}
    renamed = TaskGraph({rename[key]: TaskNode(rename[key], "different" + key,
                         [{rename[parent] for parent in group} for group in node.prerequisites], 999)
                         for key, node in graph.nodes.items()}, "z")
    assert registry.add(renamed) == first
    flattened = deepcopy(graph)
    flattened.nodes["g"] = TaskNode("g", "Goal", [{"a", "b"}], 1)
    assert registry.add(flattened) != first
    changed_goal = deepcopy(graph)
    changed_goal.goal_id = "a"
    assert registry.add(changed_goal) != first
    assert len(registry) == 3


def test_duplicate_counts_and_profile_summary(monkeypatch):
    monkeypatch.setattr(maps, "episode_from_record", lambda record: example_episode())
    registry = DAGRegistry()
    rows = [maps.analyze_map({**example_record(), "map_id": name}, registry=registry) for name in ("one", "two")]
    counts = maps._diversity_counts(rows)
    assert counts["gameplay"]["distinct"] == 1
    assert counts["gameplay"]["duplicate_groups"] == [["one", "two"]]
    assert counts["dag_classes"]["duplicate_excess"] == 1
    summary = maps.summarize_profiles(rows)[0]
    assert summary["maps"] == 2
    assert summary["open_cells_mean"] == 4
    assert summary["open_cells_std"] == 0


def test_exact_matching_separates_graphs_with_identical_refinement_colors():
    from explo2_bench.analysis.diversity import _encode

    def bipartite_graph(parent_pairs):
        nodes = {f"p{i}": TaskNode(f"p{i}", f"P{i}", [set()], 0) for i in range(4)}
        nodes.update({f"c{i}": TaskNode(f"c{i}", f"C{i}", [{f"p{j}" for j in pair}], 1)
                      for i, pair in enumerate(parent_pairs)})
        nodes["g"] = TaskNode("g", "Goal", [{f"c{i}" for i in range(4)}], 2)
        return TaskGraph(nodes, "g")

    cycle = bipartite_graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    two_squares = bipartite_graph([(0, 1), (0, 1), (2, 3), (2, 3)])
    assert _encode(cycle).bucket_key == _encode(two_squares).bucket_key
    registry = DAGRegistry()
    assert registry.add(cycle) != registry.add(two_squares)


def test_cross_split_overlap_counts_components_without_calling_them_exact_duplicates():
    rows = [{"split": split, "gameplay_sha256": str(index), "wall_layout_sha256": "shared-wall",
             "task_coordinates_sha256": "shared-sites", "dag_class": 0}
            for index, split in enumerate(["train", "test", "test", ""])]
    report = maps.split_overlap(rows)
    pair = report["pairs"]["train/test"]
    assert report["unassigned_maps"] == 1
    assert pair["gameplay"]["shared_classes"] == 0
    assert pair["wall_layouts"] == {"shared_classes": 1, "left_maps": 1,
                                   "right_maps": 2, "cross_split_map_pairs": 2}
