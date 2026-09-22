# lauras-agents-kernel sidecar

FastAPI wrapper around `laya.Agent` / `gate.Gate`, the HTTP boundary consumed
by `lauras-agents-kernel`'s Rust caller (`crates/lauras-agents-mission`) and
by the `gut-check` hermes-agent plugin -- one process, two consumers, instead
of embedding `laya`/torch a second time in either host.

```
pip install -e '.[sidecar]'
uvicorn adapters.lauras_kernel_sidecar.service:app --host 0.0.0.0 --port 8420
```

`GET /health` reports whether the model actually loaded. `POST /classify`
returns a gate verdict for one claim/context pair -- see `service.py`'s
docstring for the exact request/response shape.

## Rust caller contract

Mirrors `crates/lauras-agents-gate/src/lib.rs`'s existing TIS-engine call:
new env var `LAYA_GATE_URL` (analogous to `TIS_API_URL`); unset or a failed
call must fall back to a no-op "skip" verdict, never block the run. Insert
the call alongside the existing `observe_locomotive_output(&mut state,
run_id, &resp)` sites in `crates/lauras-agents-mission/src/lib.rs`.

## Status

Structurally verified locally (FastAPI TestClient against a mocked model, no
torch needed for that). Not yet load-tested against real concurrent traffic
from a 292-agent fan-out -- rate-limiting/sampling and batched inference
(laya's batched path: 7.2ms/question vs 33ms single) are real v1
requirements at that scale, not yet implemented, per the plan.
