# Benchmark definitions and release data

The fixed release consists of 250 (EXPLO)²-Bench v2 maps and 8,100 (EXPLO)²-Bench v3
maps. Both versions use the same record format and observation/action interface.
Their generation rules, profile sets, and sampling procedures are
distinct.

## Environment semantics

Each map specifies a rectangular grid, blocked cells, a starting cell, and one
cell per task. Cardinal moves cost one action. Coordinates are integer `(x, y)`;
up increases `y`. Visiting a cell reveals its task and the permitted dependency
information. Tasks activate automatically at their cell when their prerequisites
are satisfied. A task discovered too early may require a later revisit.

For each task, `prerequisites` is a list of alternative parent sets. Every parent
within a set is required (AND); satisfying any one set is enough (OR). `[[]]`
denotes a primitive task with no prerequisites. Success means activating the
named goal. The agent knows the goal's visible name, but does not initially know
the complete map or task graph.

The (EXPLO)²-Bench v3 construction makes every task part of a mandatory backbone
to the goal. Its saved successful route, called a feasibility witness, verifies
that the goal can be completed; it is not
an optimal route or a minimum-action certificate. (EXPLO)²-Bench v2 can have
alternatives that change which tasks need activation. Union-graph depth and node
count alone therefore do not determine the minimum work needed for success.

## Fixed collections

| Property | (EXPLO)²-Bench v2 | (EXPLO)²-Bench v3 |
|---|---|---|
| Exploration profiles | higher, high, medium, low, lower | x-lowest through x-highest; 9 profiles |
| DAG profiles | harder, hard, medium, easy, easier | x-easiest through x-hardest; 9 profiles |
| Profile pairs | 25 | 81 |
| Generation seeds | 0–9 in every pair | 100 selected seeds listed in `seeds.csv` for every pair |
| Maps | 250 | 8,100 |
| Saved default action budget | 3 × open cells | 2 × open cells |

The full profile order and exact expanded parameters are in
[`benchmarks/v2/settings.json`](../benchmarks/v2/settings.json) and
[`benchmarks/v3/settings.json`](../benchmarks/v3/settings.json). (EXPLO)²-Bench v3
used globally unique random seed candidates and selection constraints on
task-coordinate sets, wall layouts, and task-graph diversity. It is not an
unfiltered random sample. (EXPLO)²-Bench v2 and (EXPLO)²-Bench v3 do not have a
matched sampling protocol or paired maps merely because they share profile
names. Profile labels describe generator settings; increasing a nominal profile
does not guarantee greater difficulty for an agent. Selection details are
recorded in each version's `provenance.json`.

An **episode** is a complete generated environment specification. Its saved
default action budget is called its **native budget**. Evaluation copies an
episode and sets its runtime budget; it never alters the saved map or its
checksum. The evaluation multipliers are `0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10`.

## Common record format

The [JSON Schema](../benchmarks/record.schema.json) describes the shared shape,
coordinate and seed types, release markers, and version consistency. It allows
version-specific knob and metadata fields. Runtime validation additionally checks
hashes, graph semantics, connectivity, and generation identity.

Each version contains the same files:

```text
benchmarks/{v2,v3}/
  maps.jsonl.gz      # Complete records, one JSON object per decompressed line
  seeds.csv         # Map IDs, exact seeds, profiles, split memberships
  settings.json     # Exact generator parameters and ordered profiles
  manifest.json     # Counts, release metadata, checksums, canaries
  provenance.json   # Source revision, selection criteria, validation evidence
```

A record has these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Common record schema; currently `1` |
| `map_id` | Stable version/profile/seed identifier |
| `version` | `v2` or `v3` |
| `generation` | `seed`, `exploration_profile`, and `dag_profile` |
| `episode` | Complete environment specification, unchanged from generation |
| `episode_sha256` | SHA256 of the canonical episode JSON, including its native budget |
| `split` | `train`, `validation`, `test`, or `null` |
| `canary` | Stable project and release markers |

Both versions' `episode` objects share the same top-level keys:
`configuration_version`, `episode_id`, `exploration_level`,
`task_dag_difficulty`, `seed`, `knobs`, `grid`, `task_graph`, `discovery_map`,
`required_nodes`, `hidden_budget`, and `metadata`. Knob fields and metadata
contents are version-specific. The loader reconstructs each version losslessly.

Grid coordinates are two-element arrays; `discovery_map` keys are canonical
`"x,y"` strings. Graph node IDs and opaque visible names are distinct. Tasks
reference node IDs in serialized prerequisites. Every map includes the goal and
all task positions. This is evaluator data, not an agent observation.

Canonical episode hashing uses UTF-8 JSON with sorted keys, compact separators
`(',', ':')`, and `allow_nan=False`. Artifact hashes separately cover the exact
compressed files, seed table, settings, and provenance. Reproduction requires
the seed, settings, and generator revision together. The active source inventory
and checksums are in [`source_files.json`](../benchmarks/source_files.json).

```python
from explo2_bench import generate, load_benchmark
from explo2_bench.data import episode_from_record, validate_record

record = next(load_benchmark("v2"))
episode = episode_from_record(record)
recreated = generate("v2", "higher", "harder", seed=0)
validate_record(record, regenerate=True)
```

`load_benchmark` streams the full inventory by default. An explicit `root` may
point to a repository or its `benchmarks` directory; `EXPLO2_BENCH_DATA` supplies
the same location for installed clients. Wheels also include the release data
under the environment's `share/explo2_bench/benchmarks` directory. Nothing is
downloaded implicitly.

## Split memberships

All 250 (EXPLO)²-Bench v2 maps belong to the test split. (EXPLO)²-Bench v2 has no
training, validation, or unassigned maps.

(EXPLO)²-Bench v3 has 180 training, 60 validation, and 360 test maps. These are
disjoint map-ID subsets of the full 8,100-map collection.
The remaining 7,500 records have `split=null`; they are not automatically test maps.

```python
v2_test_records = load_benchmark("v2", split="test")
v3_test_records = load_benchmark("v3", split="test")
unassigned_records = load_benchmark("v3", split="unassigned")
```

Distinct IDs and seeds do not guarantee different geometry or dependency
structures. The [(EXPLO)²-Bench v3 diversity audit](../analysis/maps/v3/diversity.json)
reports overlap under explicit equivalence definitions. All memberships and maps
are public; they provide a reproducible comparison, not a secret held-out test.
For the (EXPLO)²-Bench v3 training, validation, and test subsets, the
audit finds no cross-split matches in gameplay records, wall layouts, or
task-coordinate sets.
Exact task-graph classes are shared across these subsets.
Training on designated training maps is permitted by the benchmark protocol.
Report the specific map IDs used for training, development, and evaluation;
results after tuning on the full inventory must not be described as held out.

## Dataset provenance

The generators and dataset inputs derive from the
[measurable-explore-exploit framework](https://github.com/jjj-madison/measurable-explore-exploit)
at revision `a7fa44dc138aba10a1a361302bab819188b08837`. Each version's provenance
file records its sources, selection procedure, and verification evidence.

All 8,100 (EXPLO)²-Bench v3 episodes match the canonical SHA256 hashes recorded
in the source dataset. Its 180 training, 60 validation, and 360 test assignments
come from the source split files; the other 7,500 maps are unassigned.
See [v3 provenance](../benchmarks/v3/provenance.json).

(EXPLO)²-Bench v2 contains all 25 profile combinations with seeds 0–9. The
included records and hashes define its 250 test maps. Source verification is
limited to one available example, which matches after restoring its native
action budget; identity with source copies of the other 249 maps is unverified.
See [v2 provenance](../benchmarks/v2/provenance.json) for the verification details.
