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


def test_dedupe_candidates_filters_within_batch_duplicates():
    # "a" appears twice in the same mining batch (e.g. mined from both the
    # Claude-Code and Hermes corpora) -- only the first occurrence should
    # survive, even though neither is in already_seen.
    new = [
        {"claim_text": "a", "label": "VERIFIED"},
        {"claim_text": "b", "label": "UNVERIFIED"},
        {"claim_text": "a", "label": "CONTRADICTED"},
    ]
    result = dedupe_candidates(new, already_seen=set())
    assert [c["claim_text"] for c in result] == ["a", "b"]
    assert result[0]["label"] == "VERIFIED"  # first occurrence kept


def test_main_rejects_excessive_concurrency():
    import pytest
    from nightly_stage_c import main

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["nightly_stage_c.py", "--concurrency", "20"])
        with pytest.raises(SystemExit):
            main()
