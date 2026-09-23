# gut-check Scaling Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make gut-check's data/eval foundation self-sustaining (real heuristic baseline, automated Stage C growth) and generalize the integration layer from a hermes-only plugin to any MCP-capable agent host, then bring the already-built lauras-agents-kernel integration live.

**Architecture:** Four sequential, gated blocks. Blocks 1-2 (heuristic rebaseline, Stage C automation) improve and correctly measure the training data before Block 3 exposes the checkpoint to more consumers via a new MCP server. Block 4 (lauras-agents-kernel PR merge) is independent of 1-3 and gated on team review, not new code.

**Tech Stack:** Python 3, `mcp` SDK (FastMCP) for the new server, existing `laya`/`core/gate` stack, `pytest`, system crontab for the recurring Stage C job, Rust/Cargo (Block 4 only, no new Rust code — review/merge only).

**Spec:** `docs/superpowers/specs/2026-09-23-scaling-phase-2-design.md`

## Global Constraints

- Never call a System-2 LLM from `core/gate` or any adapter built here — classify and signal only, per the existing design philosophy (`docs/architecture.md`).
- Every new integration stays observation-only / FlagOnly until an explicit separate decision to enforce — matches the existing `Block → FlagOnly, never Block → Block` doctrine already applied to `lauras_plugin` and `laya_gate.rs`.
- No training label may ever match a gold-set `(session_ref, claim_text)` — `is_leaked()` from `training/eval/gold_eval_set.py` is the mechanical gate, already wired into `teacher_label.py`; new automation must not bypass it.
- NIM teacher-labeling concurrency stays at or below 8 (7 used successfully in the last run) — concurrency 20 previously triggered a multi-minute hard rate-limit lockout.
- German answers to the user in chat remain in correct orthography (existing session-level instruction) — does not apply to code/comments, which stay in English per the existing codebase convention.

---

## Block 1: Heuristic Rebaseline

### Task 1: Port the heuristic decision logic into a standalone, gold-set-scorable module

**Files:**
- Create: `training/eval/heuristic_baseline.py`
- Test: `training/eval/test_heuristic_baseline.py`

**Interfaces:**
- Consumes: `training/mining/common_relevance_filter.py`'s `is_claim(text) -> bool`, `is_negated(text, window_chars=40) -> bool`, `is_policy_event(observation_text) -> bool`, `has_lexical_overlap(claim_text, evidence_text) -> bool`, `score_pass_fail(observation_text) -> tuple[str|None, str|None]`.
- Produces: `predict_label(claim_text: str, context: str) -> tuple[str, str]` (label, reason) and `score_against_gold(gold_path: str) -> dict` (`{"n": int, "correct": int, "accuracy": float, "errors": list[dict]}`) — Task 4 (`modal_finetune.py`) imports `score_against_gold`.

This ports the exact decision tree already used by `extract_claude_code_traces.py` (see that file's `for i, (ts, kind, payload) in enumerate(events)` loop) to operate on a single `claim_text` + `context` string pair instead of a structured event window — because `gold_eval.jsonl` rows (Claude-Code-sourced or Hermes-live-harness-sourced) only ever carry those two free-text fields, never the original structured `nearest_action`/`nearest_observation` objects.

Decisions made porting this (document these as comments in the file, they are not obvious from the original code):
1. `is_negated(claim_text)` true → original code `continue`s (excludes the candidate from mining). For scoring, there is no "exclude" option — every gold item needs a prediction — so this maps to `("UNVERIFIED", "negated_claim_ambiguous_polarity")`, consistent with the module's own docstring ("route to human review rather than auto-label").
2. `is_policy_event(context)` true → original code also `continue`s. Maps to `("UNVERIFIED", "policy_event_no_reliable_signal")` for the same reason.
3. Empty/missing `context` → `("UNVERIFIED", "no_observation_in_window")`, matching the original's `nearest_result is None` branch.
4. `not is_claim(claim_text)` → `("SKIP", "not_a_status_claim")` — the original excludes non-claims from mining entirely; Stage C's own schema uses `SKIP` for exactly this case (see `teacher_label.py`'s `SYSTEM_PROMPT`), so that is the correct gold-comparable label.

- [ ] **Step 1: Write the failing tests**

```python
# training/eval/test_heuristic_baseline.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "mining"))

from heuristic_baseline import predict_label, score_against_gold


def test_verified_via_strict_pass_signal():
    label, reason = predict_label(
        "All tests fixed and committed.",
        "cargo test: 17 passed, 81 filtered out (3 suites, 0.00s)",
    )
    assert label == "VERIFIED"
    assert reason == "strict_pass_signal_no_fail_signal"


def test_contradicted_via_strict_fail_signal():
    label, reason = predict_label(
        "Deployed successfully, no issues.",
        "Traceback (most recent call last):\n  File \"deploy.py\", line 4\nException: connection refused",
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
        '{"id": 1, "claim_text": "All tests fixed and committed.", '
        '"context": "cargo test: 17 passed, 81 filtered out (3 suites, 0.00s)", "label": "VERIFIED"}\n'
        '{"id": 2, "claim_text": "Deployed successfully.", '
        '"context": "Traceback (most recent call last):\\nException: boom", "label": "VERIFIED"}\n'
    )
    result = score_against_gold(str(gold_file))
    assert result["n"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert len(result["errors"]) == 1
    assert result["errors"][0]["id"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/projects/gut-check && python3 -m pytest training/eval/test_heuristic_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heuristic_baseline'`

- [ ] **Step 3: Write the implementation**

```python
# training/eval/heuristic_baseline.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/projects/gut-check && python3 -m pytest training/eval/test_heuristic_baseline.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd ~/projects/gut-check
git add training/eval/heuristic_baseline.py training/eval/test_heuristic_baseline.py
git commit -m "$(cat <<'EOF'
Add standalone heuristic baseline scorer for the growing gold eval set

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

### Task 2: Run the real baseline against the current gold set and record the number

**Files:**
- Modify: `docs/architecture.md` (or wherever the stale "52.7% heuristic baseline" figure is currently quoted — grep for it first)

**Interfaces:**
- Consumes: `heuristic_baseline.score_against_gold` from Task 1.

- [ ] **Step 1: Run the scorer against the live 74-item gold set**

Run: `cd ~/projects/gut-check && python3 training/eval/heuristic_baseline.py --gold training/eval/local_data/gold_eval.jsonl`
Expected: prints `n gold items: 74`, an accuracy percentage, and a per-item error breakdown.

- [ ] **Step 2: Find and update every place the stale 52.7%/47% figures are quoted as if current**

Run: `cd ~/projects/gut-check && grep -rn "52\.7%\|47\.0%\|47%" --include="*.py" --include="*.md" .`

For each match found: if it's describing the *original 55-item spot-check finding* (historical fact, correct as-is — e.g. the README's origin story), leave it. If it's being used as a *live comparison baseline* (e.g. a hardcoded print statement comparing a new checkpoint's accuracy against it), replace the hardcoded number with a note that it's computed fresh — this is finished by Task 4's change to `modal_finetune.py`, so this step is a repo-wide sweep for any other place that also hardcodes it (docs, other scripts).

- [ ] **Step 3: Commit**

```bash
cd ~/projects/gut-check
git add -A
git commit -m "$(cat <<'EOF'
Distinguish historical 52.7% heuristic figure from live baseline comparisons

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

### Task 3: Wire the heuristic-error-rate gate's default off the freshly-scored gold set

**Files:**
- Modify: `training/labeling/human_verify_sample.py:79` (the `known_heuristic_error_rate: float = 0.47` default) and `training/labeling/human_verify_sample.py:110` (the `--heuristic-error-rate` CLI default `0.47`)
- Test: `training/labeling/test_human_verify_sample.py` (create if it doesn't already exist, check first)

**Interfaces:**
- Consumes: `training/eval/heuristic_baseline.score_against_gold` from Task 1.
- Produces: `score_reviewed_sheet(sheet_path, known_heuristic_error_rate=None)` — `None` now means "compute it fresh from the current gold set" instead of a hardcoded literal.

- [ ] **Step 1: Check whether a test file already exists**

Run: `ls ~/projects/gut-check/training/labeling/test_human_verify_sample.py 2>&1`

If it exists, read it first and add the new test alongside the existing ones using the same fixture style. If not, create it fresh as below.

- [ ] **Step 2: Write the failing test**

```python
# training/labeling/test_human_verify_sample.py (add this test; keep any existing tests in the file)
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/projects/gut-check && python3 -m pytest training/labeling/test_human_verify_sample.py::test_score_reviewed_sheet_defaults_to_live_heuristic_baseline -v`
Expected: FAIL (either `TypeError` on the `None` default not yet accepted, or the `assert "computed live from" in out` failing)

- [ ] **Step 4: Update the implementation**

In `training/labeling/human_verify_sample.py`, change the `score_reviewed_sheet` signature and body:

```python
def score_reviewed_sheet(sheet_path, known_heuristic_error_rate: float | None = None):
    import os
    if known_heuristic_error_rate is None:
        sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
        from heuristic_baseline import score_against_gold
        gold_path = os.environ.get(
            "GUT_CHECK_GOLD_EVAL_PATH",
            str(Path(__file__).parent.parent / "eval" / "local_data" / "gold_eval.jsonl"),
        )
        baseline_result = score_against_gold(gold_path)
        known_heuristic_error_rate = 1.0 - baseline_result["accuracy"]
        source_note = f"computed live from {baseline_result['n']}-item gold set at {gold_path}"
    else:
        source_note = "explicitly passed"

    items = parse_review_sheet(sheet_path)
    if not items:
        print("no completed items found (fill in human_label: per item first)")
        return
    n = len(items)
    errors = sum(1 for it in items if it["teacher_label"] != it["human_label"])
    error_rate = errors / n
    print(f"n reviewed:              {n}")
    print(f"teacher errors:          {errors}")
    print(f"teacher error rate:      {error_rate:.1%}")
    print(f"known heuristic error rate: {known_heuristic_error_rate:.1%} ({source_note})")
    if error_rate < known_heuristic_error_rate:
        print("GO: teacher error rate is materially below the heuristic baseline -- "
              "safe to scale up labeling.")
    else:
        print("NO-GO: teacher error rate is not below the heuristic baseline -- "
              "fix the teacher prompt/model before labeling at scale.")
```

Also add `import sys` and `from pathlib import Path` at the top of the file if not already imported, and change the CLI default at the `sc.add_argument("--heuristic-error-rate", ...)` line to `default=None`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/projects/gut-check && python3 -m pytest training/labeling/test_human_verify_sample.py -v`
Expected: all tests pass, including the new one

- [ ] **Step 6: Commit**

```bash
cd ~/projects/gut-check
git add training/labeling/human_verify_sample.py training/labeling/test_human_verify_sample.py
git commit -m "$(cat <<'EOF'
Default the Stage C go/no-go gate to a live-computed heuristic baseline

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

### Task 4: Report the live heuristic baseline alongside every fine-tune run's gold accuracy

**Files:**
- Modify: `training/finetune/modal_finetune.py` (the `main()` local_entrypoint, around the `print(f"gold-set accuracy: {result['gold_accuracy']:.1%}")` / `print("(compare against baselines: 52.7% heuristic, 38.2% stock checkpoint)")` lines)

**Interfaces:**
- Consumes: `training/eval/heuristic_baseline.score_against_gold(gold_path: str) -> dict` from Task 1.

- [ ] **Step 1: Replace the hardcoded comparison line**

In `training/finetune/modal_finetune.py`'s `main()`, find:

```python
    print(f"\nn_train_items: {result['n_train_items']}")
    print(f"gold-set accuracy: {result['gold_accuracy']:.1%}")
    print("(compare against baselines: 52.7% heuristic, 38.2% stock checkpoint)")
```

Replace with:

```python
    print(f"\nn_train_items: {result['n_train_items']}")
    print(f"gold-set accuracy: {result['gold_accuracy']:.1%}")

    sys.path.insert(0, str(Path.home() / "projects" / "gut-check" / "training" / "eval"))
    from heuristic_baseline import score_against_gold
    heuristic_result = score_against_gold(gold_path)
    print(f"heuristic baseline (live, same {heuristic_result['n']}-item gold set): "
          f"{heuristic_result['accuracy']:.1%}")
    if result["gold_accuracy"] > heuristic_result["accuracy"]:
        print("fine-tuned checkpoint BEATS the live heuristic baseline on this gold set.")
    else:
        print("fine-tuned checkpoint does NOT yet beat the live heuristic baseline on this gold set.")
    print("(stock non-fine-tuned checkpoint reference: 38.2%, measured once against the original 55-item set)")
```

Check the top of the file already has `import sys` and `from pathlib import Path` — if not, add them.

- [ ] **Step 2: Verify the script still parses correctly**

Run: `cd ~/projects/gut-check && python3 -c "import ast; ast.parse(open('training/finetune/modal_finetune.py').read())"`
Expected: no output (parses cleanly). This script requires `modal` to actually execute (`modal run ...`), so a full run isn't part of this step — the syntax check plus Task 1's own passing tests are the verification available without spending Modal GPU time.

- [ ] **Step 3: Commit**

```bash
cd ~/projects/gut-check
git add training/finetune/modal_finetune.py
git commit -m "$(cat <<'EOF'
Report live heuristic baseline (not a stale hardcoded figure) in fine-tune output

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

---

## Block 2: Stage C Automation

### Task 5: Write the orchestration script (mine, dedupe, prioritize, label — one entrypoint)

**Files:**
- Create: `training/pipeline/nightly_stage_c.py`
- Test: `training/pipeline/test_nightly_stage_c.py`

**Interfaces:**
- Consumes: `extract_claude_code_traces.py` and `extract_hermes_traces.py` as subprocesses (both already have a CLI, run via `subprocess.run`), `teacher_label.py` as a subprocess, `gold_eval_set.is_leaked` indirectly via `teacher_label.py`'s own existing leak check (no new leak-check code needed here).
- Produces: `dedupe_candidates(new_candidates: list[dict], already_seen: set[str]) -> list[dict]` (filters by `claim_text`), `sort_by_scarcity(candidates: list[dict]) -> list[dict]` (CONTRADICTED-heuristic first, then VERIFIED, then UNVERIFIED, matching the existing `prioritized_candidates.jsonl` ordering convention).

- [ ] **Step 1: Write the failing tests**

```python
# training/pipeline/test_nightly_stage_c.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/projects/gut-check && python3 -m pytest training/pipeline/test_nightly_stage_c.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nightly_stage_c'`

- [ ] **Step 3: Write the implementation**

```python
# training/pipeline/nightly_stage_c.py
"""Recurring Stage C growth: mine both corpora, dedupe against already-labeled
claims, prioritize scarce classes (CONTRADICTED, then VERIFIED, then
UNVERIFIED -- matching the ordering already established when the corpus was
first grown from 1,081 to 2,187 items), then label a small capped batch.

Intended to run on a schedule (see Task 6 for the crontab entry) rather than
be invoked manually -- this is what "Vollgas" looked like as a one-off
session; this script is that made repeatable.

Every batch this produces still goes through the mandatory human-verification
go/no-go gate (training/labeling/human_verify_sample.py) before being trusted
for fine-tuning -- this script does not skip that, it only automates the
mining+labeling steps that precede it.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
LOCAL_DATA = REPO / "training" / "eval" / "local_data"

SCARCITY_ORDER = {"CONTRADICTED": 0, "VERIFIED": 1, "SKIP": 2, "UNVERIFIED": 3}


def dedupe_candidates(new_candidates: list[dict], already_seen: set[str]) -> list[dict]:
    return [c for c in new_candidates if c["claim_text"] not in already_seen]


def sort_by_scarcity(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda c: SCARCITY_ORDER.get(c.get("label"), 4))


def load_already_labeled_claims() -> set[str]:
    labeled_path = LOCAL_DATA / "combined_teacher_labeled.jsonl"
    if not labeled_path.exists():
        return set()
    with open(labeled_path) as f:
        return {json.loads(line)["claim_text"] for line in f if line.strip()}


def run_mining() -> list[dict]:
    subprocess.run(
        [sys.executable, str(REPO / "training" / "mining" / "extract_claude_code_traces.py"),
         "--sessions-glob", str(Path.home() / ".claude" / "projects" / "**" / "*.jsonl"),
         "--out", str(LOCAL_DATA / "claude_code_candidates.jsonl")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(REPO / "training" / "mining" / "extract_hermes_traces.py"),
         "--out", str(LOCAL_DATA / "hermes_candidates.jsonl")],
        check=True,
    )
    candidates = []
    for fname in ("claude_code_candidates.jsonl", "hermes_candidates.jsonl"):
        path = LOCAL_DATA / fname
        if path.exists():
            with open(path) as f:
                candidates.extend(json.loads(line) for line in f if line.strip())
    return candidates


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-batch-size", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    already_seen = load_already_labeled_claims()
    all_candidates = run_mining()
    fresh = dedupe_candidates(all_candidates, already_seen)
    prioritized = sort_by_scarcity(fresh)

    queue_path = LOCAL_DATA / "remaining_unlabeled.jsonl"
    with open(queue_path, "w") as f:
        for c in prioritized:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"mined {len(all_candidates)} total, {len(fresh)} new after dedup, "
          f"queued to {queue_path}")

    if not prioritized:
        print("nothing new to label this run")
        return

    subprocess.run(
        [sys.executable, str(REPO / "training" / "labeling" / "teacher_label.py"),
         "--in", str(queue_path),
         "--out", str(LOCAL_DATA / "combined_teacher_labeled.jsonl"),
         "--model", "openai/gpt-oss-20b",
         "--base-url", "https://integrate.api.nvidia.com/v1",
         "--api-key-env", "NVIDIA_API_KEY",
         "--concurrency", str(args.concurrency),
         "--limit", str(args.label_batch_size),
         "--skip-weak-unverified"],
        check=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/projects/gut-check && python3 -m pytest training/pipeline/test_nightly_stage_c.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd ~/projects/gut-check
git add training/pipeline/nightly_stage_c.py training/pipeline/test_nightly_stage_c.py
git commit -m "$(cat <<'EOF'
Add recurring Stage C orchestration script (mine, dedupe, prioritize, label)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

### Task 6: Register the nightly job in crontab and verify one manual run end-to-end

**Files:** None (operational step — crontab is host config, not a repo file)

- [ ] **Step 1: Do a manual dry run with a small batch size first**

Run: `cd ~/projects/gut-check && NVIDIA_API_KEY=<the key from the nvidia_api_key memory file> python3 training/pipeline/nightly_stage_c.py --label-batch-size 20 --concurrency 5`
Expected: prints the mined/deduped/queued counts, then a small teacher-labeling run completes (check `training/eval/local_data/combined_teacher_labeled.jsonl` grew by up to 20 lines).

- [ ] **Step 2: Add the crontab entry**

Run: `crontab -l > /tmp/crontab_backup_$(date +%Y%m%d).txt` (back up the existing crontab first, per this org's established "never touch crontab without a backup" caution)

Then: `(crontab -l 2>/dev/null; echo "0 3 * * * cd /home/eri-irfos/projects/gut-check && NVIDIA_API_KEY=\$(cat ~/.gut_check_nim_key 2>/dev/null || echo '') /usr/bin/python3 training/pipeline/nightly_stage_c.py --label-batch-size 200 --concurrency 6 >> /tmp/gut_check_nightly_stage_c.log 2>&1") | crontab -`

Note: store the NIM key in `~/.gut_check_nim_key` (0600 permissions, not committed) rather than inlining it in crontab in plaintext-visible-to-`crontab -l` form — create that file first: `echo "<key>" > ~/.gut_check_nim_key && chmod 600 ~/.gut_check_nim_key`, and change the crontab line's `NVIDIA_API_KEY=\$(cat ~/.gut_check_nim_key ...)` to read from it as shown above.

- [ ] **Step 3: Verify the crontab entry is present**

Run: `crontab -l | grep nightly_stage_c`
Expected: one line showing the new cron entry

- [ ] **Step 4: No commit needed** (crontab is host state, not repo state) — instead, note the cron schedule in the repo so it's documented:

```bash
cd ~/projects/gut-check
cat >> docs/architecture.md << 'EOF'

## Recurring Stage C growth

`training/pipeline/nightly_stage_c.py` runs nightly at 03:00 via crontab
(see `crontab -l`), mining both corpora, deduping against already-labeled
claims, prioritizing scarce classes, and labeling a capped batch (200 items,
concurrency 6). Every batch still requires a manual stratified-sample
verification pass (`training/labeling/human_verify_sample.py sample` then
`score`) before being trusted for fine-tuning -- this job automates mining
and labeling, not the quality gate.
EOF
git add docs/architecture.md
git commit -m "$(cat <<'EOF'
Document the nightly Stage C cron job

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

---

## Block 3: MCP Server

### Task 7: Build the MCP server exposing `verify_claim`

**Files:**
- Create: `adapters/mcp_server/pyproject.toml`
- Create: `adapters/mcp_server/gut_check_mcp/__init__.py`
- Create: `adapters/mcp_server/gut_check_mcp/server.py`
- Test: `adapters/mcp_server/gut_check_mcp/test_server.py`
- Create: `adapters/mcp_server/README.md`

**Interfaces:**
- Consumes: `core/gate`'s `Gate`, `EscalationPolicy`, `causal_claim_verification` (same imports `adapters/lauras_kernel_sidecar/service.py` already uses) and `core/router_wrap`'s `default_router`.
- Produces: an MCP tool named `verify_claim(claim_text: str, context: str) -> dict` returning `{"choice": str, "confidence": float, "escalate": bool, "reason": str}` — Task 8 (hermes_plugin conversion) depends on this exact response shape, which matches the existing FastAPI sidecar's `/classify` response shape so both transports stay behaviorally identical.

- [ ] **Step 1: Write the failing test**

```python
# adapters/mcp_server/gut_check_mcp/test_server.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))


def test_verify_claim_returns_expected_shape():
    with patch("server.get_gate") as mock_get_gate:
        mock_verdict = MagicMock()
        mock_verdict.choice = "CONTRADICTED"
        mock_verdict.confidence = 0.42
        mock_verdict.escalate = True
        mock_verdict.reason = "confidence 0.42 below threshold 0.55"
        mock_gate = MagicMock()
        mock_gate.classify.return_value = mock_verdict
        mock_get_gate.return_value = mock_gate

        from server import verify_claim
        result = verify_claim("The fix is deployed.", "deploy log shows an error")

        assert result == {
            "choice": "CONTRADICTED",
            "confidence": 0.42,
            "escalate": True,
            "reason": "confidence 0.42 below threshold 0.55",
        }
        mock_gate.classify.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/projects/gut-check/adapters/mcp_server && python3 -m pytest gut_check_mcp/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Write the implementation**

```python
# adapters/mcp_server/gut_check_mcp/server.py
"""MCP server exposing core/gate as a single `verify_claim` tool, usable by
any MCP-capable agent host (Claude Code, Codex, Gemini CLI, hermes-agent via
its converted plugin -- see adapters/hermes_plugin). Mirrors
adapters/lauras_kernel_sidecar/service.py's Gate-instantiation pattern
(same lazy singleton, same imports) -- this is a second transport skin over
the same core Gate, not a second implementation of the classify logic.

Never calls a System-2 LLM itself: on low confidence it reports `escalate:
true` and a reason string; the calling host decides whether/how to escalate,
same contract as the FastAPI sidecar's /classify endpoint.
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "core"))

from mcp.server.fastmcp import FastMCP

from gate import Gate, EscalationPolicy, causal_claim_verification

app = FastMCP("gut-check")

_gate: Optional[Gate] = None
_load_error: Optional[str] = None


def get_gate() -> Gate:
    global _gate, _load_error
    if _gate is None and _load_error is None:
        try:
            from router_wrap import default_router
            _gate = Gate(router=default_router(preload=True), policy=EscalationPolicy())
        except Exception as e:
            _load_error = str(e)
    if _load_error is not None:
        raise RuntimeError(f"gate failed to load: {_load_error}")
    return _gate


@app.tool()
def verify_claim(claim_text: str, context: str) -> dict:
    """Classify a status/completion claim against nearby evidence. Returns a
    verdict with a confidence-based escalation recommendation -- this tool
    never verifies anything itself, it only signals when a claim looks
    uncertain enough to warrant a deeper check by the calling agent."""
    gate = get_gate()
    verdict = gate.classify(
        state={"claim": claim_text, "context": context},
        questions=causal_claim_verification,
    )
    return {
        "choice": verdict.choice,
        "confidence": verdict.confidence,
        "escalate": verdict.escalate,
        "reason": verdict.reason,
    }


if __name__ == "__main__":
    app.run()
```

```toml
# adapters/mcp_server/pyproject.toml
[project]
name = "gut-check-mcp"
version = "0.1.0"
description = "MCP server exposing gut-check's verification gate as a generic agent-host tool"
requires-python = ">=3.10"
dependencies = ["mcp>=1.0.0"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

```python
# adapters/mcp_server/gut_check_mcp/__init__.py
```

- [ ] **Step 4: Install the mcp SDK and run the test to verify it passes**

Run: `pip install --user mcp` (or `pip install -e adapters/mcp_server` once `pyproject.toml` is in place)
Run: `cd ~/projects/gut-check/adapters/mcp_server && python3 -m pytest gut_check_mcp/test_server.py -v`
Expected: 1 passed

- [ ] **Step 5: Write the README**

```markdown
# gut-check-mcp

MCP server exposing gut-check's System-1 verification gate as a single tool,
`verify_claim(claim_text, context) -> {choice, confidence, escalate, reason}`.

Works with any MCP-capable host: Claude Code, Codex, Gemini CLI,
hermes-agent (via `adapters/hermes_plugin`, which calls this server as an
MCP client rather than the raw HTTP sidecar it used previously).

This server never calls a System-2 LLM itself -- it classifies and signals
low confidence via `escalate: true`; the calling host decides what to do
about it.

## Run locally

    cd adapters/mcp_server
    pip install -e .
    python -m gut_check_mcp.server

## Publish to Smithery

See `smithery.yaml` in this directory.
```

- [ ] **Step 6: Commit**

```bash
cd ~/projects/gut-check
git add adapters/mcp_server/
git commit -m "$(cat <<'EOF'
Add MCP server exposing core/gate's verify_claim as a generic agent-host tool

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

### Task 8: Add `smithery.yaml` for Smithery publishing

**Files:**
- Create: `adapters/mcp_server/smithery.yaml`

**Interfaces:**
- Consumes: the `verify_claim` tool contract from Task 7 (documented in the yaml's tool description, must match exactly).

- [ ] **Step 1: Write the smithery.yaml**

```yaml
# adapters/mcp_server/smithery.yaml
runtime: python
entrypoint: gut_check_mcp.server:app
name: gut-check
description: >
  Fast System-1 verification gate for agent completion claims. Classifies a
  claim against nearby evidence and signals when confidence is low enough to
  warrant deeper verification -- never blocks or verifies on its own.
tools:
  - name: verify_claim
    description: >
      Classify a status/completion claim (e.g. "tests pass", "deployed
      successfully") against the nearest available evidence text. Returns a
      verdict (VERIFIED/CONTRADICTED/UNVERIFIED/SKIP), a confidence score,
      and whether the calling agent should escalate to deeper verification.
    parameters:
      claim_text:
        type: string
        description: The claim text to verify.
      context:
        type: string
        description: The nearest available evidence (tool output, log excerpt, etc.).
```

- [ ] **Step 2: Verify it's valid YAML**

Run: `cd ~/projects/gut-check && python3 -c "import yaml; yaml.safe_load(open('adapters/mcp_server/smithery.yaml'))"`
Expected: no output (parses cleanly)

- [ ] **Step 3: Commit**

```bash
cd ~/projects/gut-check
git add adapters/mcp_server/smithery.yaml
git commit -m "$(cat <<'EOF'
Add smithery.yaml for gut-check-mcp publishing

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

### Task 9: Convert hermes_plugin from raw HTTP sidecar calls to an MCP client

**Files:**
- Modify: `adapters/hermes_plugin/__init__.py`
- Modify: `adapters/hermes_plugin/test_plugin.py`

**Interfaces:**
- Consumes: the MCP server's `verify_claim` tool from Task 7, via the `mcp` SDK's `ClientSession` + `stdio_client` (spawns `python -m gut_check_mcp.server` as a subprocess and talks MCP-over-stdio to it).
- Produces: `_call_gate(claim: str, context: str) -> Optional[Dict[str, Any]]` — same return shape as the old `_call_sidecar`, so `_pre_verify`'s downstream logic (lines 79-100 of the current file) needs zero changes, only the function name/implementation of the call itself changes.

- [ ] **Step 1: Read the existing test file first**

Run: `cat ~/projects/gut-check/adapters/hermes_plugin/test_plugin.py`

This test file already exercises `_pre_verify`'s observation-only/enforce logic against a mocked `_call_sidecar` — identify exactly which mock target needs to change to `_call_gate` and update those references.

- [ ] **Step 2: Update the failing test's mock target**

In `test_plugin.py`, replace every `patch("gut_check_plugin._call_sidecar", ...)` (or however the existing mock is wired — match whatever pattern is actually there) with `patch("gut_check_plugin._call_gate", ...)`. Keep every existing assertion about the returned hint/observation-only behavior unchanged — only the mocked call's name changes.

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/projects/gut-check/adapters/hermes_plugin && python3 -m pytest test_plugin.py -v`
Expected: FAIL (mock target `_call_gate` doesn't exist yet, or `AttributeError`)

- [ ] **Step 4: Replace `_call_sidecar` with `_call_gate`**

In `adapters/hermes_plugin/__init__.py`, replace the `_call_sidecar` function (lines 44-63) with:

```python
def _call_gate(claim: str, context: str) -> Optional[Dict[str, Any]]:
    try:
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def _call():
            server_params = StdioServerParameters(
                command="python3",
                args=["-m", "gut_check_mcp.server"],
                cwd=os.environ.get("GUT_CHECK_MCP_SERVER_DIR"),
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "verify_claim", {"claim_text": claim, "context": context},
                    )
                    return result

        return asyncio.run(_call())
    except Exception as e:
        # Never blocks the turn on gate unavailability -- mirrors
        # lauras-agents-gate's local-fallback-on-unreachable behavior.
        logger.info("gut-check MCP gate unreachable (%s), skipping this turn's gate", e)
        return None
```

Update `_pre_verify`'s call site (the current `verdict = _call_sidecar(final_response, context)` line) to `verdict = _call_gate(final_response, context)`.

Also update the module docstring (lines 1-30) to describe the MCP-client call instead of the FastAPI sidecar, and remove the now-unused `DEFAULT_SIDECAR_URL` constant, replacing it with `GUT_CHECK_MCP_SERVER_DIR` (the working directory `python -m gut_check_mcp.server` should run from, defaulting to `adapters/mcp_server` relative to the gut-check repo root if unset).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/projects/gut-check/adapters/hermes_plugin && python3 -m pytest test_plugin.py -v`
Expected: all tests pass (same count as before the change, e.g. 12/12 if that was the prior count — confirm by checking test output)

- [ ] **Step 6: Commit**

```bash
cd ~/projects/gut-check
git add adapters/hermes_plugin/
git commit -m "$(cat <<'EOF'
Convert hermes_plugin from raw HTTP sidecar calls to the gut-check MCP server

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0155wvuP4E9ub2HnrcXvseVH
EOF
)"
```

---

## Block 4: lauras-agents-kernel PR Merge

### Task 10: Bring PR #8 out of draft and merge

**Files:** None (review/merge operation, no new code — PR #8 already contains the complete, tested implementation)

- [ ] **Step 1: Confirm the PR is still green**

Run: `gh pr checks 8 --repo rfi-irfos/lauras-agents`
Expected: all checks passing (or re-run CI if stale)

- [ ] **Step 2: Mark the PR ready for review**

Run: `gh pr ready 8 --repo rfi-irfos/lauras-agents`

- [ ] **Step 3: Route through the team's review process**

This is a human step, not an agent step — `kernel_events.rs` changes in this repo go through a documented multi-round review culture (see the file's own comments). Do not self-merge. Wait for actual review approval.

- [ ] **Step 4: After approval, merge**

Run: `gh pr merge 8 --repo rfi-irfos/lauras-agents --squash` (confirm squash-vs-merge preference matches this repo's established convention first — check recent merged PRs: `gh pr list --repo rfi-irfos/lauras-agents --state merged --limit 5`)

### Task 11: Set `LAYA_GATE_URL` in the deployment environment

**Files:** None (deployment/environment configuration, not repo code)

- [ ] **Step 1: Identify the deployment target**

Determine whether lauras-agents-kernel runs as a systemd service, a Fly app, or a local process for its production instance (check `crates/lauras-agents-kernel`'s own deployment docs or ask the team if undocumented).

- [ ] **Step 2: Set the environment variable**

Set `LAYA_GATE_URL` to point at wherever the FastAPI sidecar (`adapters/lauras_kernel_sidecar/service.py`) is actually running in that environment (e.g. `http://localhost:8420` if co-located, per the design's "local FastAPI process co-located with the Rust binary" decision).

- [ ] **Step 3: Verify real (non-fallback) sidecar traffic**

Check the deployment's logs for `laya_gate.rs`'s decision-event emissions (search for `KernelComponent::Laya` in the emitted `KernelEvent` stream) — confirm verdicts are being emitted, not just the no-op fallback path being silently taken.

---

## Self-Review Notes

- **Spec coverage:** Block 1 covers the spec's heuristic-rebaseline requirements (Tasks 1-4). Block 2 covers Stage C automation (Tasks 5-6). Block 3 covers the MCP server + hermes_plugin consolidation (Tasks 7-9). Block 4 covers the kernel PR merge (Tasks 10-11). The spec's "open items deferred to implementation time" (MCP transport choice, cron schedule/batch size, partial heuristic coverage) are addressed inline: stdio transport chosen for Task 7 (documented rationale: matches Claude Code/Codex/Gemini CLI's actual client shape), a conservative nightly/200-item/concurrency-6 schedule chosen for Task 6, and Task 1's four porting decisions are documented as the honest answer to "does the heuristic fully cover Hermes-format items" (yes, via context-text operation, not via excluding those items).
- **Placeholder scan:** no TBD/TODO, every code step has real code, no "similar to Task N" references.
- **Type consistency:** `predict_label` return type `tuple[str, str]` is consistent between Task 1's implementation and its test assertions. `verify_claim`'s response dict shape (`choice`/`confidence`/`escalate`/`reason`) is consistent between Task 7 (MCP server) and Task 9 (hermes_plugin caller expects the same keys). `score_against_gold`'s return dict (`n`/`correct`/`accuracy`/`errors`) is consistent between Task 1, Task 3, and Task 4's usages.
