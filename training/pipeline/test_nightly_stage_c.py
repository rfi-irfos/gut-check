import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nightly_stage_c import dedupe_candidates, sort_by_scarcity


def test_dedupe_candidates_filters_already_seen():
    new = [
        {"claim_text": "a", "label": "VERIFIED"},
        {"claim_text": "b", "label": "UNVERIFIED"},
        {"claim_text": "c", "label": "CONTRADICTED"},
    ]
    result = dedupe_candidates(new, already_seen={"b"})
    assert [c["claim_text"] for c in result] == ["a", "c"]


def test_sort_by_scarcity_puts_contradicted_first():
    candidates = [
        {"claim_text": "u", "label": "UNVERIFIED"},
        {"claim_text": "c", "label": "CONTRADICTED"},
        {"claim_text": "v", "label": "VERIFIED"},
    ]
    result = sort_by_scarcity(candidates)
    assert [c["label"] for c in result] == ["CONTRADICTED", "VERIFIED", "UNVERIFIED"]
