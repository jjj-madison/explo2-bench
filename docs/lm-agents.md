# Language-model agent interface

Use `make_env(record, alpha=...)` in the evaluator and send only observations to the agent. `observation_to_text` serializes an allowlist of public fields. No model service, provider SDK, or API key is required.

Call `reset()` before the first action. It returns the public observation dictionary. `step(action)` returns a `StepResult` with `observation`, `done`, `success`, and `info`; the latter contains only action validity and the termination reason. A goal at the start can terminate during reset, so check `env.done` before acting. The default uses the map's saved action budget, called its native budget: three times the open-cell count for v2 and twice that count for v3. An `alpha` override uses `floor(alpha × open_cells)` and rejects a result below one; alternatively pass an exact positive integer `action_budget`. These evaluator parameters remain hidden from the agent. Success on the final allowed action counts as success.

Observations contain the current coordinate, elapsed action count, legal cardinal moves, goal token, discovered/activated task tokens, and newly revealed prerequisites and successors. Groups represent alternatives of conjunctions: `[["A", "B"], ["C"]]` means `(A AND B) OR C`. Relations can mention undiscovered names without revealing locations. Agents retain earlier discoveries because relation payloads are incremental.

Coordinates use `up = (x, y+1)` and `right = (x+1, y)`. Return a mapping or JSON string with exactly one field: `{"move":"up"}`. Other directions are `down`, `left`, and `right`. Extra keys, prose, invalid directions, malformed JSON, and moves into walls consume a turn as invalid actions. No fallback movement or provider retry is implicit.

`LMCallableAgent(callback)` adapts a callback accepting a public observation dictionary to the internal policy protocol. The callback owns history and provider calls. Its `reset()` is called when available. Exceptions during an action callback become `callback_error`; setup/reset failures propagate to the caller. Under the core episode runner, an action callback failure terminates the episode and consumes one action, with the exception type recorded and its message omitted. Malformed callback responses instead become ordinary invalid actions. See the [runnable callback example](../examples/lm_agent.py), which uses the public reset/step loop and a dummy response without contacting an LM. External LM experiments should report model/version, prompt, decoding, context handling, retries, tokens, and monetary cost. The eight included baselines make no model calls.

Python object privacy is not an execution sandbox. Agents with repository access can inspect public maps and seeds. For controlled evaluation, isolate the agent process/filesystem and transmit only protocol messages. This cannot reverse prior public exposure. [Canary markers](../CANARY.md) identify benchmark material for exposure audits and are excluded from observations and prompts.

Full-map figures are privileged views for human analysis; do not give them to agents under the partial-observation protocol.

## Score a completed episode

Call the public evaluator helper after termination:

```python
from explo2_bench import score_episode

scores = score_episode(env)
```

`score_episode` requires a reset, completed `BenchmarkEnv`. It returns success, termination reason, action usage and allowance, goal activation step, discovery/activation counts and fractions, visited-cell coverage, invalid actions, regime counts, and exploration/exploitation/union errors. Regime counts group actions by the scorer's exploration-only, exploitation-only, and mixed classifications. Its eligible-step denominators and rates match the baseline runner; a zero denominator is `None`. A successful zero-action episode has goal activation step zero and undefined action-based error rates.

Per-action errors come from the pinned
[upstream metric implementation](https://github.com/jjj-madison/measurable-explore-exploit/blob/2bb260104d7272d3908f312f3b1f67e394fbaa9b/src/symbolic_environment/metrics.py).
Exploration eligibility excludes goal-pending actions; exploitation eligibility
includes them. See the [baseline measurement protocol](baselines.md#measurements)
for the denominators and aggregation used in reports.

This is evaluator-only reporting after termination. Scores include total open cells, task count, and the final allowance, so keep them outside agent prompts and observation histories. The result contains no map layout, task identities, generation seed, saved successful route, canary, or raw trajectory. Measure model latency, tokens, and cost separately in the callback or evaluation harness.
