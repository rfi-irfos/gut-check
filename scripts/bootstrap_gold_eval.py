"""One-time (repeatable) ingestion of the original 55-item hand-verified spot
check into the gold_eval_set.py format. Reads two private, local files that
are NOT part of this repo:

  ~/Desktop/causal1_review_sheet.md      -- claim/context per item + filled-in correct_label
  ~/projects/laya-spikes/causal1_gate/items.json  -- the same items, machine-parsed

Writes training/eval/local_data/gold_eval.jsonl (gitignored).

Usage:
    python scripts/bootstrap_gold_eval.py
"""
import json
import re
from pathlib import Path

SHEET = Path.home() / "Desktop" / "causal1_review_sheet.md"
ITEMS_JSON = Path.home() / "projects" / "laya-spikes" / "causal1_gate" / "items.json"
GROUND_TRUTH_JSON = Path.home() / "projects" / "laya-spikes" / "causal1_gate" / "ground_truth.json"
OUT = Path(__file__).parent.parent / "training" / "eval" / "local_data" / "gold_eval.jsonl"

SESSION_RE = re.compile(r"session:\s*`([^`]+)`")


def main():
    items = json.loads(ITEMS_JSON.read_text())
    truth = json.loads(GROUND_TRUTH_JSON.read_text())
    sheet_text = SHEET.read_text()

    # pull the session ref per item number, same numbering as items.json's "id"
    session_refs = {}
    for i, block in enumerate(re.split(r"\n---\n", sheet_text)):
        m = re.search(r"^## (\d+)\.", block, re.MULTILINE)
        s = SESSION_RE.search(block)
        if m and s:
            session_refs[int(m.group(1))] = s.group(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for it in items:
            label = truth.get(str(it["id"]))
            if label is None:
                continue
            row = {
                "id": it["id"],
                "claim_text": it["claim"],
                "context": it["context"],
                "label": label,
                "source": "claude_code_spot_check",
                "session_ref": session_refs.get(it["id"], ""),
                "added_from": "causal1_review_sheet.md#55-item-seed",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} gold items to {OUT}")


if __name__ == "__main__":
    main()
