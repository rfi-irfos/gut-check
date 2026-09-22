"""gut-check: a System-1/System-2 dual-loop verification gate for AI agents.

A fast, non-autoregressive typed-decision classifier (`laya`) makes cheap calls
in a single forward pass and honestly flags its own uncertainty. When it isn't
confident, the loop escalates to a slower, more deliberate System-2 caller
(a large LLM) supplied by the host application. `gut-check` never calls an LLM
API itself and never silently blocks or rewrites output on its own -- see
Gate.classify / Gate.escalate.
"""
from .escalation import EscalationContext, EscalationResult
from .gate import Gate, GateVerdict
from .policy import EscalationPolicy
from .questions import causal_claim_verification, response_quality, user_reaction_prediction

__version__ = "0.1.0"
__all__ = [
    "Gate",
    "GateVerdict",
    "EscalationPolicy",
    "EscalationContext",
    "EscalationResult",
    "causal_claim_verification",
    "response_quality",
    "user_reaction_prediction",
    "__version__",
]
