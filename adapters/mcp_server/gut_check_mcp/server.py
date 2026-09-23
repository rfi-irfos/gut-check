"""MCP server exposing core/gate as a single `verify_claim` tool, usable by
any MCP-capable agent host (Claude Code, Codex, Gemini CLI, hermes-agent via
its converted plugin -- see adapters/hermes_plugin). Mirrors
adapters/lauras_kernel_sidecar/service.py's Gate-instantiation pattern
(same lazy singleton, same imports) -- this is a second transport skin over
the same core Gate, not a second implementation of the classify logic.

Never calls a System-2 LLM itself: on low confidence it reports `escalate:
true` and a reason string; the calling host decides whether/how to escalate,
same contract as the FastAPI sidecar's /classify endpoint.
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "core"))

from mcp.server.fastmcp import FastMCP

from gate import Gate, EscalationPolicy, causal_claim_verification

app = FastMCP("gut-check")

_gate: Optional[Gate] = None
_load_error: Optional[str] = None


def get_gate() -> Gate:
    global _gate, _load_error
    if _gate is None and _load_error is None:
        try:
            from gate.router_wrap import default_router
            _gate = Gate(router=default_router(preload=True), policy=EscalationPolicy())
        except Exception as e:
            _load_error = str(e)
    if _load_error is not None:
        raise RuntimeError(f"gate failed to load: {_load_error}")
    return _gate


@app.tool()
def verify_claim(claim_text: str, context: str, confidence_threshold: float = 0.55) -> dict:
    """Classify a status/completion claim against nearby evidence. Returns a
    verdict with a confidence-based escalation recommendation -- this tool
    never verifies anything itself, it only signals when a claim looks
    uncertain enough to warrant a deeper check by the calling agent.

    `confidence_threshold` overrides EscalationPolicy's own default (0.55) per
    call, without disturbing the cached Gate/router singleton -- the policy is
    cheap to construct fresh each call, unlike the loaded model."""
    gate = get_gate()
    gate.policy = EscalationPolicy(confidence_threshold=confidence_threshold)
    verdict = gate.classify(
        state={"claim": claim_text, "context": context},
        questions=causal_claim_verification,
    )
    return {
        "choice": verdict.choice,
        "confidence": verdict.confidence,
        "escalate": verdict.escalate,
        "reason": verdict.reason,
    }


if __name__ == "__main__":
    app.run()
