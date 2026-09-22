"""The core Gate: classify() wraps a laya call, escalate() hands off to System 2.

Philosophy (see docs/architecture.md): gut-check never blocks or rewrites
output by itself. It classifies and, on low confidence, flags for escalation;
the host decides what to do with that -- log it, call System 2, or ignore it.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .escalation import EscalationResult, SystemTwoCall, build_escalation_context
from .policy import EscalationPolicy


@dataclass
class GateVerdict:
    question_id: str
    type: str
    choice: Optional[str] = None
    score: Optional[float] = None
    noul: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    escalate: bool = False
    reason: str = ""
    raw: Optional[Dict] = None


def _to_verdict(qid: str, answer: Dict, policy: EscalationPolicy) -> GateVerdict:
    should_escalate = policy.should_escalate(answer)
    reason = "" if not should_escalate else (
        f"confidence {answer.get('confidence', 0.0):.2f} below threshold "
        f"{policy.confidence_threshold:.2f}"
    )
    return GateVerdict(
        question_id=qid,
        type=answer.get("type", ""),
        choice=answer.get("choice"),
        score=answer.get("score"),
        noul=answer.get("noul"),
        probabilities=answer.get("probabilities"),
        confidence=answer.get("confidence", 0.0),
        escalate=should_escalate,
        reason=reason,
        raw=answer,
    )


class Gate:
    """A System-1 classifier plus an escalation policy.

    `model` is anything exposing `.predict(state, questions)` or
    `.system_one(state, questions)` returning laya's `{"answers": {...}}` shape
    -- a real `laya.Router`/`laya.Agent`, or a test double for unit tests that
    don't need torch installed.
    """

    def __init__(self, router=None, policy: Optional[EscalationPolicy] = None):
        self.router = router
        self.policy = policy or EscalationPolicy()

    def classify(self, state: Any, questions: Dict[str, Dict]) -> Union[GateVerdict, Dict[str, GateVerdict]]:
        call = getattr(self.router, "predict", None) or getattr(self.router, "system_one", None)
        if call is None:
            raise TypeError(
                f"{type(self.router)!r} has no .predict or .system_one method; "
                f"pass a laya.Router, laya.Agent, or compatible object."
            )
        result = call(state, questions)
        answers = result["answers"]
        verdicts = {qid: _to_verdict(qid, ans, self.policy) for qid, ans in answers.items()}
        if len(verdicts) == 1:
            return next(iter(verdicts.values()))
        return verdicts

    def escalate(self, verdict: GateVerdict, system_two_call: SystemTwoCall, state: Any = None) -> EscalationResult:
        ctx = build_escalation_context(state, verdict.question_id, verdict.raw or {})
        output = system_two_call(ctx)
        return EscalationResult(
            final_answer=output,
            system_one_answer=verdict.raw or {},
            system_two_output=output,
            escalated=True,
        )
