import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "mining"))

from heuristic_baseline import predict_label, score_against_gold


def test_verified_via_strict_pass_signal():
    label, reason = predict_label(
        "All cargo tests pass, ready to merge.",
        "cargo test: 17 passed, 81 filtered out (3 suites, 0.00s)",
    )
    assert label == "VERIFIED"
    assert reason == "strict_pass_signal_no_fail_signal"


def test_contradicted_via_strict_fail_signal():
    label, reason = predict_label(
        "Deploy script migration completed successfully.",
        "migration script execution: Traceback (most recent call last):\n  File \"migrate.py\", line 4\nException: connection refused",
    )
    assert label == "CONTRADICTED"
    assert reason == "strict_fail_signal_no_pass_signal"


def test_unverified_when_no_lexical_overlap():
    label, reason = predict_label(
        "Merged the ledger migration successfully.",
        "unrelated snippet about a completely different subsystem's config file",
    )
    assert label == "UNVERIFIED"
    assert reason == "no_lexical_overlap_with_nearest_evidence"


def test_unverified_when_context_empty():
    label, reason = predict_label("Fixed and deployed.", "")
    assert label == "UNVERIFIED"
    assert reason == "no_observation_in_window"


def test_skip_for_stated_intention():
    label, reason = predict_label(
        "Let me fix the config file now.",
        "cargo test: 17 passed",
    )
    assert label == "SKIP"
    assert reason == "not_a_status_claim"


def test_unverified_for_negated_claim():
    label, reason = predict_label(
        "The bug is not yet fixed, still investigating.",
        "cargo test: 17 passed",
    )
    assert label == "UNVERIFIED"
    assert reason == "negated_claim_ambiguous_polarity"


def test_score_against_gold(tmp_path):
    gold_file = tmp_path / "gold_eval.jsonl"
    gold_file.write_text(
        '{"id": 1, "claim_text": "All cargo tests pass, ready to merge.", '
        '"context": "cargo test: 17 passed, 81 filtered out (3 suites, 0.00s)", "label": "VERIFIED"}\n'
        '{"id": 2, "claim_text": "Deploy script migration completed successfully.", '
        '"context": "migration script execution: Traceback (most recent call last):\\nException: boom", "label": "VERIFIED"}\n'
    )
    result = score_against_gold(str(gold_file))
    assert result["n"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert len(result["errors"]) == 1
    assert result["errors"][0]["id"] == 2
