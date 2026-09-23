# Architecture

## The core idea

A fast, non-autoregressive System-1 classifier (`laya`) makes a typed decision
in one forward pass (~33ms) and reports calibrated confidence. When it's
confident, that's the answer. When it isn't -- and on the stock English
checkpoint, on this project's causal-claim task, it almost never is (see
README's 38.2%/near-zero-confidence numbers) -- the loop escalates to
System 2: something slower and more capable, supplied by the host
application.

`gut-check` never blocks or rewrites output by itself (see
`core/gate/gate.py`'s docstring). It classifies and flags. What "escalate"
*means* is host-specific, and the two reference integrations realize it
differently on purpose:

## hermes-agent: escalation realized as a hook return, not a second call

hermes-agent's `pre_verify` hook fires once per turn, right before a
completion claim surfaces, and a callback can hand back
`{"action": "continue", "message": "..."}` to make the *same* agent loop
keep going with a hint, instead of finishing. For this host, System 2 is
already the same big LLM running the turn -- so `gut-check`'s hermes plugin
doesn't make a second LLM call itself. It calls the sidecar for a System-1
verdict, and on low confidence, returns a continue-directive carrying the
uncertainty hint. hermes' own loop re-prompts its own model with that hint.
See `adapters/hermes_plugin/__init__.py`.

This deliberately differs from `core/gate/escalation.py`'s generic
`system_two_call: Callable` contract, which assumes the host injects an
explicit second LLM call. hermes doesn't need that shape; its own hook
semantics already are the escalation mechanism.

## lauras-agents-kernel: escalation realized as an emitted decision event

Mission Control's dispatch loop calls the sidecar after each agent's
response (same insertion point as the existing `observe_locomotive_output`
calls), and the verdict becomes a new `KernelComponent::Laya` event through
the existing `KernelEvent`/`DecisionEvent` schema -- observed, not enforced,
following the same `Block -> FlagOnly` (never `Block -> Block`) precedent
`lauras-agents-kernel`'s own UIP gate already established. What happens with
a flagged event (whether anything calls a bigger model, re-runs the agent,
or just surfaces the flag to a human) is a decision for whoever consumes the
kernel event stream, not something `gut-check` decides on its own.

## Observation before enforcement

Both v1 integrations ship logging-only by default (hermes: `GUT_CHECK_ENFORCE`
unset; kernel: the event is always observation-only per the existing UIP
design). Turning on any behavior change is an explicit opt-in, after a
burn-in period of reviewing what the gate would have flagged.

## Deferred use cases (named, not built)

The same `Gate.classify()`/`.escalate()` shape generalizes past causal-claim
verification. `core/gate/questions.py` defines two further typed-question
presets as API-compatible stubs, neither trained nor wired into either
adapter in v1:

- `response_quality` -- "is this response actually the best one for the user
  right now, given context and session duration" -- a pre-output gate.
- `user_reaction_prediction` -- "will the user likely react negatively to
  this" -- no mined training-data source exists for this yet, unlike the
  other two, which both have real corpora today.

Building either is a matter of mining/labeling data and fine-tuning a
checkpoint for that question, not changing the core library.

## Recurring Stage C growth

`training/pipeline/nightly_stage_c.py` runs nightly at 03:00 via crontab
(see `crontab -l`), mining both corpora, deduping against already-labeled
claims, prioritizing scarce classes, and labeling a capped batch (200 items,
concurrency 6). Every batch still requires a manual stratified-sample
verification pass (`training/labeling/human_verify_sample.py sample` then
`score`) before being trusted for fine-tuning -- this job automates mining
and labeling, not the quality gate.

NIM's free tier is sometimes heavily overloaded (observed repeatedly during
development, e.g. 2026-09-22/23) -- a given night's batch may net 0 newly
labeled items if every teacher call times out. This is expected and
self-healing: failed-to-label candidates never enter
`combined_teacher_labeled.jsonl`, so the next night's dedup step naturally
requeues them without any special retry bookkeeping.
