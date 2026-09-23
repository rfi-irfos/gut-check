"""Standalone, gold-set-scorable port of the heuristic decision logic that
already produces `label`/`label_source: heuristic_structural_v2` on mined
Stage A/B candidates (see training/mining/extract_claude_code_traces.py's
per-candidate loop). That version needs a structured event window
(nearest_action/nearest_observation objects); this version operates on the
plain (claim_text, context) pairs that every gold_eval.jsonl row carries
regardless of source (Claude-Code-trace or live-Hermes-harness), so the
baseline can be recomputed as the gold set grows past its original
Claude-Code-only 55 items.

Four decisions made porting the original branch logic to always emit a
prediction (the original silently excludes some candidates from mining
rather than labeling them -- scoring has no "exclude" option):
1. Negated claims -> UNVERIFIED (original: excluded, docstring says "route
   to human review", i.e. it isn't confident either way).
2. Policy-event observations -> UNVERIFIED (original: excluded, no reliable
   signal about the claim's actual effect).
3. No/empty observation -> UNVERIFIED (unchanged from original).
4. Non-claims (stated intentions, refusals) -> SKIP (original: excluded
   from mining; SKIP is Stage C's own label for exactly this case).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mining"))
from common_relevance_filter import (
    is_claim,
    is_negated,
    is_policy_event,
    has_lexical_overlap,
    score_pass_fail,
)


def predict_label(claim_text: str, context: str) -> tuple[str, str]:
    if not is_claim(claim_text):
        return "SKIP", "not_a_status_claim"
    if is_negated(claim_text):
        return "UNVERIFIED", "negated_claim_ambiguous_polarity"
    if not context:
        return "UNVERIFIED", "no_observation_in_window"
    if is_policy_event(context):
        return "UNVERIFIED", "policy_event_no_reliable_signal"
    if not has_lexical_overlap(claim_text, context):
        return "UNVERIFIED", "no_lexical_overlap_with_nearest_evidence"
    label, reason = score_pass_fail(context)
    if label is None:
        return "UNVERIFIED", "ambiguous_pass_and_fail_signals_both_or_neither"
    return label, reason


def score_against_gold(gold_path: str) -> dict:
    with open(gold_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    errors = []
    correct = 0
    for row in rows:
        pred, reason = predict_label(row["claim_text"], row.get("context", ""))
        if pred == row["label"]:
            correct += 1
        else:
            errors.append({
                "id": row["id"], "gold_label": row["label"],
                "predicted_label": pred, "reason": reason,
            })
    n = len(rows)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "errors": errors,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()
    result = score_against_gold(args.gold)
    print(f"n gold items:      {result['n']}")
    print(f"heuristic correct: {result['correct']}")
    print(f"heuristic accuracy: {result['accuracy']:.1%}")
    if result["errors"]:
        print(f"\n{len(result['errors'])} errors (id, gold_label, predicted_label, reason):")
        for e in result["errors"]:
            print(f"  {e['id']}: gold={e['gold_label']} pred={e['predicted_label']} ({e['reason']})")


if __name__ == "__main__":
    main()
