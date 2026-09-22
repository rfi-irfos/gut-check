# gut-check

A System-1/System-2 dual-loop verification gate for AI agents.

## The story

We hand-verified 55 real agent session traces — claims like "all tests pass" or
"deployed" — against the nearest tool action and observation that actually
happened. A naive `is_error`-flag heuristic got it wrong on **47%** of a
stratified sample: not because agents lie often, but because "the nearest tool
call failed" and "the claim is false" are not the same fact, and treating them
as interchangeable produces false confidence in both directions.

We then ran [`laya`](https://github.com/rfi-irfos/laya) — a fast,
non-autoregressive "System 1" typed-decision model (33ms/call) — over the same
55 items, completely unmodified, no fine-tuning. Raw accuracy: **38.2%**,
worse than the heuristic. But its own calibrated confidence was **near-zero
on almost every item**. It didn't know the answer, and it said so, instead of
confidently hallucinating one.

That's the whole idea. `gut-check` doesn't ask you to trust a fast classifier's
raw accuracy. It asks the classifier to be honest about what it doesn't know,
and routes exactly those cases to something slower and more capable — your
big LLM, called only when the fast model says "I'm not sure." Two systems,
one loop: cheap and fast when the fast model is confident, slow and
deliberate when it isn't.

## Quickstart

```python
import laya
from gate import Gate, EscalationPolicy, causal_claim_verification

gate = Gate(
    router=laya.Router(preload=True),
    policy=EscalationPolicy(confidence_threshold=0.55, min_margin=0.15),
)

verdict = gate.classify(
    state={"claim": "All 12 tests pass. Committing now.",
           "context": "Bash exit code 1: 3 failed, 9 passed"},
    questions=causal_claim_verification,
)
print(verdict.choice, verdict.confidence)  # -> CONTRADICTED 0.71

if verdict.escalate:
    result = gate.escalate(verdict, system_two_call=your_llm_call)
```

`gate.escalate()` takes **your own** LLM call as an injected function.
`gut-check` never calls an LLM API on your behalf — it only decides *when*
you should.

## What's here

- `core/gate/` — the framework-agnostic classify/escalate library (this is
  the reusable part; the rest is training data, eval tooling, and adapters
  for two specific agent frameworks we run internally).
- `training/` — mining, teacher-labeling, and fine-tuning pipeline. Real
  session-derived training data is private and never committed here (see
  `.gitignore`); only the pipeline code and aggregate results are public.
- `adapters/` — integration points for `hermes-agent` (a plugin) and
  `lauras-agents-kernel` (an HTTP sidecar). Reference implementations of the
  pattern, not requirements — the core library works with any agent runtime
  that can inject its own System-2 call.

## Design principle: observation before enforcement

`gut-check` never silently blocks or rewrites your agent's output. It
classifies, and on low confidence it flags for escalation — what you do with
that flag (log it, call System 2, ignore it) is entirely up to the host
application. v1 integrations run in observation/logging-only mode before any
escalation call is allowed to affect real output.

## Status

Early / alpha. The causal-claim-verification use case (the one in the story
above) is the first vertical being built end-to-end, from stock-checkpoint
baseline through fine-tuning through live integration. `response_quality` and
`user_reaction_prediction` are named, API-compatible extension points with no
training data or wiring yet — see `core/gate/questions.py`.

## License

Apache-2.0, matching [`laya`](https://github.com/rfi-irfos/laya)'s own license.
