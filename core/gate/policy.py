"""Escalation policy: decides, from a laya answer, whether to hand off to System 2.

Stateless by default (pure threshold on the model's own calibrated confidence).
Subclass for stateful policies (e.g. hysteresis across turns, per-question-type
thresholds) -- `laya`'s choice/score/noul question types calibrate confidence
differently, so a single global threshold is a starting point, not a law.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class EscalationPolicy:
    confidence_threshold: float = 0.55
    """Below this confidence, escalate to System 2."""

    min_margin: float = 0.15
    """Optional secondary gate: for `choice` answers, also escalate if the top-2
    option probabilities are within this margin of each other, even when the
    reported confidence alone clears the threshold -- catches confident-but-close
    calls that entropy-based confidence can under-flag."""

    def should_escalate(self, answer: Dict) -> bool:
        conf = answer.get("confidence", 0.0)
        if conf < self.confidence_threshold:
            return True
        probs = answer.get("probabilities")
        if answer.get("type") == "choice" and probs and len(probs) >= 2:
            top2 = sorted(probs.values(), reverse=True)[:2]
            if (top2[0] - top2[1]) < self.min_margin:
                return True
        return False
