"""Unit tests against a fake router -- no torch/laya model load needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from gate import EscalationPolicy, Gate, causal_claim_verification


class FakeRouter:
    """Returns a fixed laya-shaped answer regardless of input."""

    def __init__(self, answer):
        self._answer = answer

    def predict(self, state, questions):
        qid = next(iter(questions))
        return {"answers": {qid: self._answer}, "routing": {"model": "fake"}}


def _choice_answer(choice, confidence, probabilities):
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": probabilities,
        "confidence": confidence,
        "action": {"act_probability": 0.5},
    }


def test_high_confidence_does_not_escalate():
    router = FakeRouter(_choice_answer(
        "VERIFIED", 0.95,
        {"VERIFIED": 0.9, "CONTRADICTED": 0.05, "UNVERIFIED": 0.03, "SKIP": 0.02},
    ))
    gate = Gate(router=router, policy=EscalationPolicy(confidence_threshold=0.55, min_margin=0.15))
    verdict = gate.classify({"claim": "x", "context": "y"}, causal_claim_verification)
    assert verdict.choice == "VERIFIED"
    assert verdict.escalate is False


def test_low_confidence_escalates():
    router = FakeRouter(_choice_answer(
        "CONTRADICTED", 0.05,
        {"VERIFIED": 0.3, "CONTRADICTED": 0.3, "UNVERIFIED": 0.25, "SKIP": 0.15},
    ))
    gate = Gate(router=router, policy=EscalationPolicy(confidence_threshold=0.55, min_margin=0.15))
    verdict = gate.classify({"claim": "x", "context": "y"}, causal_claim_verification)
    assert verdict.escalate is True
    assert "confidence" in verdict.reason


def test_close_top2_margin_escalates_even_at_high_confidence():
    # confidence clears the threshold, but top-2 options are within min_margin
    router = FakeRouter(_choice_answer(
        "VERIFIED", 0.60,
        {"VERIFIED": 0.52, "CONTRADICTED": 0.48, "UNVERIFIED": 0.0, "SKIP": 0.0},
    ))
    gate = Gate(router=router, policy=EscalationPolicy(confidence_threshold=0.55, min_margin=0.15))
    verdict = gate.classify({"claim": "x", "context": "y"}, causal_claim_verification)
    assert verdict.escalate is True


def test_escalate_calls_injected_system_two_with_hint():
    router = FakeRouter(_choice_answer(
        "UNVERIFIED", 0.10,
        {"VERIFIED": 0.25, "CONTRADICTED": 0.25, "UNVERIFIED": 0.25, "SKIP": 0.25},
    ))
    gate = Gate(router=router, policy=EscalationPolicy())
    verdict = gate.classify({"claim": "x", "context": "y"}, causal_claim_verification)
    assert verdict.escalate is True

    calls = []

    def fake_system_two(ctx):
        calls.append(ctx)
        assert "confidence" in ctx.hint
        assert ctx.system_one_answer["choice"] == "UNVERIFIED"
        return "system-2 says VERIFIED"

    result = gate.escalate(verdict, system_two_call=fake_system_two)
    assert len(calls) == 1
    assert result.final_answer == "system-2 says VERIFIED"
    assert result.escalated is True


def test_gate_never_raises_on_no_escalation_path():
    """Escalation is opt-in: classify() alone never calls out to anything but the router."""
    router = FakeRouter(_choice_answer(
        "VERIFIED", 0.99,
        {"VERIFIED": 0.99, "CONTRADICTED": 0.003, "UNVERIFIED": 0.003, "SKIP": 0.004},
    ))
    gate = Gate(router=router)
    verdict = gate.classify({"claim": "x", "context": "y"}, causal_claim_verification)
    assert verdict.escalate is False
    # no escalate() call made -- nothing else should have happened
