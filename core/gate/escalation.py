"""The System-2 call contract.

`gut-check` never calls an LLM API itself -- the host application injects its
own call as `system_two_call`. `EscalationContext` bundles the System-1
verdict (choice/score/noul + full probability distribution + confidence) with
a natural-language uncertainty hint, so System 2 gets both a structured signal
to reason over and a plain-English summary, rather than being re-asked the
bare original question with no signal about *why* it was escalated.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class EscalationContext:
    state: Any
    """The original claim/state passed to Gate.classify."""

    question_id: str
    system_one_answer: Dict
    """The raw laya answer dict for the escalated question: type, choice/score/noul,
    probabilities, confidence, action."""

    hint: str
    """Natural-language uncertainty summary, e.g. 'scored UNVERIFIED at 0.12
    confidence -- treat this claim as unconfirmed and check directly.'"""


@dataclass
class EscalationResult:
    final_answer: Any
    system_one_answer: Dict
    system_two_output: Any
    escalated: bool = True


def build_hint(question_id: str, answer: Dict) -> str:
    conf = answer.get("confidence", 0.0)
    if answer.get("type") == "choice":
        choice = answer.get("choice", "?")
        return (
            f"a fast classifier scored '{question_id}' as {choice} at {conf:.2f} "
            f"confidence -- treat this as unconfirmed and verify directly rather than "
            f"trusting the label."
        )
    if answer.get("type") == "score":
        return (
            f"a fast classifier scored '{question_id}' at {answer.get('score')} "
            f"({conf:.2f} confidence) -- treat this as a weak prior, not a verdict."
        )
    return (
        f"a fast classifier scored '{question_id}' at {conf:.2f} confidence -- "
        f"treat this as unconfirmed."
    )


def build_escalation_context(state: Any, question_id: str, answer: Dict) -> EscalationContext:
    return EscalationContext(
        state=state,
        question_id=question_id,
        system_one_answer=answer,
        hint=build_hint(question_id, answer),
    )


SystemTwoCall = Callable[[EscalationContext], Any]
