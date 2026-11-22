# Submit results from your own agent

Submit one JSON file containing your run configuration and individual problem
results through a pull request to this directory. The maintainers will derive
aggregate statistics from these records. You do not need to submit averages,
summary tables, or plots.

Start from [template.json](template.json) and name your copy
`<agent-name>-<run-id>.json`, using lowercase letters, numbers, and hyphens.
For example, `my-agent-v3-test-run-001.json` identifies one submission. Keep
`template.json` unchanged; it is an incomplete example, not a submitted run.

## Connect and evaluate your agent

Use [examples/lm_agent.py](../examples/lm_agent.py) as a starting point. Replace
the dummy callback with your own observation-to-action agent. The callback can
use an LM, a heuristic, or another policy. See the [agent interface
guide](../docs/lm-agents.md) for observations, action parsing, and scoring.

The current example runs only the first map, uses `alpha=2`, and prints its
scores. It does not iterate over a problem set or export submission files.
Preparing a submission requires your own evaluation loop and serialization
around the existing interface; there is currently no submission-export flag.

For each selected problem, budget multiplier, and replicate:

1. Create the environment from its benchmark record with the selected `alpha`.
2. Create a fresh agent or explicitly reset its episode state, including its
   conversation history. `env.reset()` resets the environment only; the example
   does not reset your callback for you.
3. Call `env.reset()` and pass only the returned public observations to your
   agent. While `env.done` is false, obtain an action and call `env.step(action)`.
4. After termination, call `score_episode(env)` and retain its complete returned
   dictionary as the result's `scores` object.
5. Save the result before proceeding so an interrupted evaluation retains the
   outcomes already collected.

Preserve memory within an episode. Keep full benchmark records, map IDs,
generation seeds, hidden budgets, and evaluator scores outside the agent's
input. Record agent metadata and scoring information in the submission file
after evaluation.

## Declare the problems and evaluation settings

Each submission contains one benchmark version and one fixed agent
configuration. Use a new run ID and file when you change the agent, model,
prompts, decoding, history policy, or evaluation configuration.

Record the exact selected map IDs in `run.selection.map_ids`. The `split` field
describes the selection filter: `train`, `validation`, `test`, `unassigned`, or
JSON `null` for no split filter. The map-ID list identifies the actual selection,
including any limited subset. A result's own `split` must match its benchmark
record, where unassigned maps have `split=null`.

| Version | Test maps | Complete inventory |
|---|---:|---:|
| `v2` | 250 | 250 |
| `v3` | 360 | 8,100 |

Loading the full v3 inventory also includes training, validation, and unassigned
maps. See the [split definitions](../docs/benchmark.md#split-memberships).
List any maps used to train or develop the agent in
`run.agent.training_map_ids` and `run.agent.development_map_ids`; use an empty
list only when no benchmark maps were used for that purpose.

This submission format uses explicit positive budget multipliers in
`run.evaluation.alphas`, with `action_budget = floor(alpha × open_cells)`.
Each multiplier must yield at least one action for every selected problem.
Record the actual integer allowance for each result. Replicates are independent
attempts with the same agent configuration, numbered from zero to
`run.evaluation.replicates - 1`. Describe how policy seeds are chosen, and record
each attempt's seed when supported. Use `null` when no seed is controlled.

Follow any problem selection, budget, and repetition requirements specified by
the maintainers for the comparison you are entering. This guide does not assign
a universal submission budget or repetition count. The example's `alpha=2` is a
demo setting; existing [baseline evaluations](../docs/baselines.md#fixed-configuration)
use ten multipliers. Compare results only under matching evaluation settings.

## Complete the submission file

The JSON object has three top-level fields: `schema_version`, `run`, and
`results`. Use submission `schema_version=1`; this is separate from the
benchmark record schema. The template contains one placeholder result showing
the record structure. Replace its placeholders and expand `results` to cover
every selected problem, multiplier, and replicate.

### Run metadata

| Field | What to record |
|---|---|
| `run.run_id`, `run.method` | A unique run identifier and a stable name for your agent. |
| `run.created_at_utc` | The run's creation time as an ISO 8601 UTC timestamp ending in `Z`. |
| `run.agent` | Agent description, source location when available, exact code revision, configuration, memory/reset policy, and retry policy. For LM agents, also provide the model/provider/version, exact prompt or prompt templates, and decoding settings. |
| `run.benchmark` | Benchmark version, installed package version, repository revision used for evaluation, and metric dependency provenance. |
| `run.selection` | The split filter and exact selected map IDs. |
| `run.evaluation` | Unique positive budget multipliers, a positive integer replicate count, and the policy-seed strategy. |
| `run.status` | `complete` when every selected case has a `completed` or `error` record; otherwise `incomplete`. This describes execution coverage, not agent success. |

Replace `run.benchmark.metric_dependency` with the provenance dictionary
returned by `explo2_bench.upstream.metric_dependency_provenance()` from the
evaluation environment. It identifies the installed metric sources and pinned
revision; see the [provenance helper](../src/explo2_bench/upstream.py).
Identify local agent or evaluator changes in the configuration and source
description so a revision alone does not conceal modifications.

For non-LM agents, `model`, `prompt`, and `decoding` may remain `null`.
For LM agents, use descriptive JSON objects or strings for these fields,
including the provider/model identifier and version, prompt contents and
construction, and decoding parameters. Do not leave placeholder text in a
submission.

### Problem-wise results

Within a run, `(map_id, alpha, replicate)` uniquely identifies a result. The
agent and benchmark version are inherited from `run`. For example, three
problems at one multiplier with two replicates require six result records,
including any errors or unattempted cases.

| Field | Meaning |
|---|---|
| `map_id`, `episode_sha256`, `split` | Copy these values from the selected benchmark record. The hash identifies the original episode, including its native budget; retain it when overriding the runtime budget. |
| `replicate`, `policy_seed` | The zero-based replicate index and actual policy seed, or `null` when unsupported. |
| `alpha`, `action_budget` | The selected multiplier and exact runtime allowance. The allowance may be `null` only when it was not established for an `error` or `not_run` case. |
| `status` | `completed`, `error`, or `not_run`, as described below. |
| `scores` | The entire unmodified `score_episode(env)` dictionary for a completed case; otherwise `null`. |
| `error` | Failure stage, exception type, and a short explanation for an execution error; otherwise `null`. |
| `usage` | Optional measurements for the individual attempt: agent wall time in seconds, input/output tokens, and cost in USD. Use `null` for unmeasured or unavailable values. |

Preserve every scorer field, including success, termination reason, action use,
task discovery/activation, coverage, invalid actions, error counts, eligible-step
denominators, regime counts, and error rates. Do not replace these with an
average or recompute the rates yourself. The [scoring documentation](../docs/lm-agents.md#score-a-completed-episode)
describes their meanings. The result's `action_budget` must match
`scores.action_budget` when scores are present.

Serialize Python `None` as JSON `null`, including undefined rates with zero
denominators. Keep booleans and numbers as JSON values, not strings. Do not use
`NaN` or infinity, round away the scorer's precision, or fill missing values
with zero.

If measured, `usage.agent_wall_seconds` covers agent setup/reset and action
selection, including provider waits and retries, but excludes environment
stepping and scoring. Token and cost measurements cover that same attempt,
including retries. These measurements are supplied by your harness, not by
`score_episode`.

### Unsuccessful episodes, errors, and incomplete runs

A completed, scored episode has `status="completed"` whether it reaches the
goal or exhausts its budget. Preserve the scorer's `success` and
`terminated_reason` values. Malformed actions and illegal moves consume actions
under the existing protocol; they do not justify discarding the episode.

If setup, the callback/provider, environment stepping, or scoring fails before
you obtain a completed score, set `status="error"` and `scores=null`. Put the
failure stage, exception type, and a short explanation in `error`, for example:

```json
{"stage": "callback", "type": "TimeoutError", "message": "Provider request timed out after the configured retries."}
```

The direct callback in `lm_agent.py` propagates exceptions, and
`score_episode` rejects unfinished environments. Your harness must record these
errors explicitly. If another runner handles a callback failure by terminating
and scoring the episode under the documented protocol, preserve that score and
its `callback_error` termination reason as a `completed` result.

Use `status="not_run"`, `scores=null`, and `error=null` for selected cases that
have not been attempted. Mark the overall run `incomplete` while any such cases
remain. Do not omit failed cases, select only successful replicates, or rerun
failures until they succeed. State retry behavior in the run metadata. Resume
only with the same agent, selection, and evaluation settings; preserve existing
outcomes and avoid duplicate result keys.

## Open a pull request

1. Fork the repository and create a branch for your submission.
2. Copy `submissions/template.json` to a new file in this directory, using the
   `<agent-name>-<run-id>.json` naming convention.
3. Replace the placeholder metadata and populate all problem-wise records.
   Check that the file parses as JSON, all selected cases are represented once,
   map IDs and hashes match the benchmark, and the run status reflects coverage.
4. Add your submission file to the pull request. Include the agent name, run ID,
   benchmark version, selection, budgets, and replicate count in its description.
   Explain any execution errors or incomplete cases.

Keep access credentials and private provider account details out of the public
file. Full model transcripts are not required. Retain any action traces and
local logs you use to inspect or reproduce the run; the scalar scores alone do
not reconstruct an episode.

Maintainers will use the individual records to derive aggregate statistics.
`template.json` is reserved for the template and must be excluded when reading
submitted runs.
