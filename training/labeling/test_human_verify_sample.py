import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from human_verify_sample import score_reviewed_sheet


def test_score_reviewed_sheet_defaults_to_live_heuristic_baseline(tmp_path, monkeypatch, capsys):
    gold_file = tmp_path / "gold_eval.jsonl"
    gold_file.write_text(
        '{"id": 1, "claim_text": "Fixed and deployed.", "context": "cargo test: 17 passed", "label": "VERIFIED"}\n'
    )
    monkeypatch.setenv("GUT_CHECK_GOLD_EVAL_PATH", str(gold_file))

    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        "## 1. teacher_label: VERIFIED (source: claude_code, model: x)\n\n"
        "**claim:**\n> some claim\n\nitem_id: ``\n\nhuman_label: VERIFIED\nnote: \n\n---\n"
    )
    score_reviewed_sheet(str(sheet), known_heuristic_error_rate=None)
    out = capsys.readouterr().out
    assert "known heuristic error rate:" in out
    assert "computed live from" in out


def test_score_reviewed_sheet_floors_gate_against_majority_class_baseline(tmp_path, monkeypatch, capsys):
    # A gold set where the heuristic is worse than trivially guessing the
    # majority class ("VERIFIED"): the heuristic gets the one CONTRADICTED
    # item wrong too (predicts UNVERIFIED via no_lexical_overlap), so
    # heuristic accuracy (2/4 = 50%) is below majority-class accuracy
    # (3/4 = 75%). The gate must floor against the stronger (majority-class)
    # baseline, i.e. known_heuristic_error_rate should be 1 - 0.75 = 25%, not
    # 1 - 0.50 = 50%.
    gold_file = tmp_path / "gold_eval.jsonl"
    gold_file.write_text(
        '\n'.join([
            '{"id": 1, "claim_text": "Fixed and deployed.", "context": "cargo test: 17 passed", "label": "VERIFIED"}',
            '{"id": 2, "claim_text": "Fixed and shipped.", "context": "cargo test: 17 passed", "label": "VERIFIED"}',
            '{"id": 3, "claim_text": "Fixed and merged.", "context": "cargo test: 17 passed", "label": "VERIFIED"}',
            '{"id": 4, "claim_text": "totally unrelated words here", "context": "xyzzy plugh", "label": "CONTRADICTED"}',
        ])
    )
    monkeypatch.setenv("GUT_CHECK_GOLD_EVAL_PATH", str(gold_file))

    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        "## 1. teacher_label: VERIFIED (source: claude_code, model: x)\n\n"
        "**claim:**\n> some claim\n\nitem_id: ``\n\nhuman_label: VERIFIED\nnote: \n\n---\n"
    )
    score_reviewed_sheet(str(sheet), known_heuristic_error_rate=None)
    out = capsys.readouterr().out
    assert "majority-class accuracy: 75.0%" in out
    assert "known heuristic error rate: 25.0%" in out


def test_score_reviewed_sheet_missing_gold_file_does_not_raise(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUT_CHECK_GOLD_EVAL_PATH", str(tmp_path / "does_not_exist.jsonl"))

    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        "## 1. teacher_label: VERIFIED (source: claude_code, model: x)\n\n"
        "**claim:**\n> some claim\n\nitem_id: ``\n\nhuman_label: VERIFIED\nnote: \n\n---\n"
    )
    score_reviewed_sheet(str(sheet), known_heuristic_error_rate=None)
    out = capsys.readouterr().out
    assert "error: no gold eval set found at" in out


def test_score_reviewed_sheet_explicit_rate_not_overridden(tmp_path, capsys):
    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        "## 1. teacher_label: VERIFIED (source: claude_code, model: x)\n\n"
        "**claim:**\n> some claim\n\nitem_id: ``\n\nhuman_label: CONTRADICTED\nnote: \n\n---\n"
    )
    score_reviewed_sheet(str(sheet), known_heuristic_error_rate=0.47)
    out = capsys.readouterr().out
    assert "known heuristic error rate: 47.0% (explicitly passed)" in out
