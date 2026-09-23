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


def test_score_reviewed_sheet_explicit_rate_not_overridden(tmp_path, capsys):
    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        "## 1. teacher_label: VERIFIED (source: claude_code, model: x)\n\n"
        "**claim:**\n> some claim\n\nitem_id: ``\n\nhuman_label: CONTRADICTED\nnote: \n\n---\n"
    )
    score_reviewed_sheet(str(sheet), known_heuristic_error_rate=0.47)
    out = capsys.readouterr().out
    assert "known heuristic error rate: 47.0% (explicitly passed)" in out
