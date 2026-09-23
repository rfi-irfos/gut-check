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
    """Filters against already-labeled claims AND against duplicate
    claim_texts within this same batch of new_candidates, keeping only the
    first occurrence of each -- mining both corpora in one run can surface
    the same claim_text twice (e.g. it appears in both the Claude-Code and
    Hermes traces), and that shouldn't slip through as two "new" items."""
    seen_this_batch: set[str] = set()
    deduped = []
    for c in new_candidates:
        claim = c["claim_text"]
        if claim in already_seen or claim in seen_this_batch:
            continue
        seen_this_batch.add(claim)
        deduped.append(c)
    return deduped


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
         "--db", str(Path.home() / ".hermes" / "state.db"),
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

    # concurrency 20 previously triggered a multi-minute hard NIM rate-limit
    # lockout -- reject anything above the documented safe ceiling outright
    # rather than silently clamping it, so a bad crontab edit fails loudly.
    if args.concurrency > 8:
        ap.error(
            f"--concurrency {args.concurrency} exceeds the safe ceiling of 8 -- "
            f"concurrency 20 previously triggered a multi-minute NIM rate-limit lockout"
        )

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
