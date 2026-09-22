"""Generic scoring: compare a Gate's classify() calls against ground-truth labels.

Deliberately takes candidates/ground_truth as plain data (list of {id, claim,
context, ...} + {id: label} dict) rather than hardcoding any corpus -- real
session-derived data is private and lives outside this repo (see .gitignore).

Usage:
    from gate import Gate
    from score import score_gate

    result = score_gate(gate, items, ground_truth, questions=causal_claim_verification)
    print(result.accuracy, result.escalation_rate)
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ScoreResult:
    n: int
    correct: int
    escalated: int
    confusion: Counter = field(default_factory=Counter)
    rows: List[Dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.n if self.n else 0.0


def score_gate(gate, items: List[Dict], ground_truth: Dict[str, str], questions: Dict,
                state_fn=None) -> ScoreResult:
    """`items` need `id` plus whatever `state_fn` needs to build a laya state
    (default: {"claim": item["claim"], "context": item["context"]})."""
    if state_fn is None:
        state_fn = lambda it: {"claim": it["claim"], "context": it["context"]}

    correct = escalated = 0
    confusion = Counter()
    rows = []
    for it in items:
        state = state_fn(it)
        verdict = gate.classify(state, questions)
        gt = ground_truth[str(it["id"])]
        pred = verdict.choice
        is_correct = pred == gt
        correct += is_correct
        escalated += verdict.escalate
        confusion[(gt, pred)] += 1
        rows.append({
            "id": it["id"], "ground_truth": gt, "pred": pred,
            "confidence": verdict.confidence, "escalate": verdict.escalate,
        })

    return ScoreResult(n=len(items), correct=correct, escalated=escalated,
                        confusion=confusion, rows=rows)


def print_report(result: ScoreResult, label: str = "gate") -> None:
    print(f"--- {label} ---")
    print(f"n:               {result.n}")
    print(f"accuracy:        {result.correct}/{result.n} = {result.accuracy:.1%}")
    print(f"escalation rate: {result.escalated}/{result.n} = {result.escalation_rate:.1%}")
    print("confusion (ground_truth -> pred):")
    for (gt, pred), cnt in sorted(result.confusion.items()):
        marker = "" if gt == pred else "  <-- miss"
        print(f"  {gt:12s} -> {pred!s:12s} : {cnt}{marker}")
