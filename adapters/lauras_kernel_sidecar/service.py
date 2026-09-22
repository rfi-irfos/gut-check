"""FastAPI sidecar wrapping laya.Agent -- the shared inference boundary for
every non-Python-native host. lauras-agents-kernel's Rust caller talks to
this over HTTP (mirroring crates/lauras-agents-gate's TIS-engine pattern);
the hermes-agent plugin also talks to this over HTTP rather than embedding a
second copy of laya in hermes's own process (keeps torch/transformers out of
the host framework's own dependency tree).

Run:
    uvicorn service:app --host 0.0.0.0 --port 8420

Contract:
    POST /classify
    {"claim": "...", "context": "...", "confidence_threshold": 0.55, "min_margin": 0.15}
    ->
    {"choice": "CONTRADICTED", "confidence": 0.71, "probabilities": {...},
     "escalate": true, "reason": "confidence 0.71 below threshold 0.80"}

    GET /health -> {"status": "ok", "model_loaded": true}

Deliberately minimal: no batching queue, no auth, no rate limiting in v1 --
those are real requirements before this sits behind live 292-agent fan-out
traffic (see plan's rate-limiting note), not before a first working
integration. Falls back cleanly: if the model fails to load, /health reports
it and /classify returns 503 rather than crashing the process, so a Rust
caller's local-fallback-on-unreachable behavior (mirroring
lauras-agents-gate) has something well-defined to fall back from.
"""
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gate import Gate, EscalationPolicy, causal_claim_verification

app = FastAPI(title="gut-check sidecar")

_gate: Optional[Gate] = None
_load_error: Optional[str] = None


def get_gate() -> Gate:
    global _gate, _load_error
    if _gate is None and _load_error is None:
        try:
            from router_wrap import default_router  # noqa: E402  (needs core/ on path)
            _gate = Gate(router=default_router(preload=True))
        except Exception as e:
            _load_error = str(e)
    if _gate is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error}")
    return _gate


class ClassifyRequest(BaseModel):
    claim: str
    context: str
    confidence_threshold: float = 0.55
    min_margin: float = 0.15


class ClassifyResponse(BaseModel):
    choice: Optional[str] = None
    score: Optional[float] = None
    noul: Optional[float] = None
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    escalate: bool
    reason: str


@app.get("/health")
def health():
    return {"status": "ok" if _load_error is None else "degraded", "model_loaded": _gate is not None,
            "load_error": _load_error}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    gate = get_gate()
    gate.policy = EscalationPolicy(confidence_threshold=req.confidence_threshold, min_margin=req.min_margin)
    verdict = gate.classify(
        {"claim": req.claim, "context": req.context},
        causal_claim_verification,
    )
    return ClassifyResponse(
        choice=verdict.choice, score=verdict.score, noul=verdict.noul,
        confidence=verdict.confidence, probabilities=verdict.probabilities,
        escalate=verdict.escalate, reason=verdict.reason,
    )
