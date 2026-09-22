"""Stage C, step 2: stratified re-sample of teacher-labeled candidates for
human verification, in the same review-sheet format the original 55-item
spot check used -- so the same review workflow applies at any scale.

Also computes the go/no-go gate once a filled-in sheet is parsed back:
teacher error rate vs. the known 47% heuristic baseline. Fix the teacher
prompt/model before scaling up if this doesn't clear that bar -- see
score_reviewed_sheet().

Usage:
    python human_verify_sample.py sample --in teacher_labeled.jsonl \
        --out review_sheet.md --rate 0.10 --seed 42

    # ... a human fills in `human_label:` / `note:` per item in review_sheet.md ...

    python human_verify_sample.py score --sheet review_sheet.md
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict


def stratified_sample(rows, rate: float, seed: int, strata_keys=("teacher_label", "source")):
    by_stratum = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, "?") for k in strata_keys)
        by_stratum[key].append(r)

    rng = random.Random(seed)
    sample = []
    for key, group in by_stratum.items():
        rng.shuffle(group)
        n = max(1, round(len(group) * rate))
        sample.extend(group[:n])
    return sample


def write_review_sheet(sample, out_path):
    with open(out_path, "w") as f:
        f.write("# Teacher-label human verification sample\n\n")
        f.write(
            f"{len(sample)} items, stratified by (teacher_label, source). For each: read claim + "
            f"context + teacher_rationale, then fill in `human_label` (VERIFIED / CONTRADICTED / "
            f"UNVERIFIED / SKIP) and a one-line `note` if the teacher was wrong.\n\n---\n\n"
        )
        for i, r in enumerate(sample, 1):
            action = r.get("nearest_action") or {}
            obs = r.get("nearest_observation") or {}
            f.write(f"## {i}. teacher_label: {r.get('teacher_label')} "
                    f"(source: {r.get('source')}, model: {r.get('teacher_model', '?')})\n\n")
            f.write(f"**claim:**\n> {r['claim_text']}\n\n")
            f.write(f"**nearest_action:** {action.get('type', '(none)')}\n")
            f.write(f"**nearest_observation:**\n> {obs.get('excerpt', '(none)')}\n\n")
            f.write(f"**teacher_rationale:** {r.get('teacher_rationale', '')}\n\n")
            f.write(f"item_id: `{r.get('id') or r.get('session_id', '')}`\n\n")
            f.write("human_label: \nnote: \n\n---\n\n")


def parse_review_sheet(sheet_path):
    text = open(sheet_path).read()
    items = []
    for block in re.split(r"\n---\n", text):
        m_teacher = re.search(r"teacher_label:\s*(\w+)", block)
        m_human = re.search(r"human_label:\s*(\w+)", block)
        if m_teacher and m_human:
            items.append({"teacher_label": m_teacher.group(1), "human_label": m_human.group(1)})
    return items


def score_reviewed_sheet(sheet_path, known_heuristic_error_rate: float = 0.47):
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
    print(f"known heuristic error rate: {known_heuristic_error_rate:.1%}")
    if error_rate < known_heuristic_error_rate:
        print("GO: teacher error rate is materially below the heuristic baseline -- "
              "safe to scale up labeling.")
    else:
        print("NO-GO: teacher error rate is not below the heuristic baseline -- "
              "fix the teacher prompt/model before labeling at scale.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rate", type=float, default=0.10)
    s.add_argument("--seed", type=int, default=42)

    sc = sub.add_parser("score")
    sc.add_argument("--sheet", required=True)
    sc.add_argument("--heuristic-error-rate", type=float, default=0.47)

    args = ap.parse_args()
    if args.cmd == "sample":
        with open(args.inp) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        sample = stratified_sample(rows, args.rate, args.seed)
        write_review_sheet(sample, args.out)
        print(f"wrote {len(sample)} items (of {len(rows)}) to {args.out}")
    elif args.cmd == "score":
        score_reviewed_sheet(args.sheet, args.heuristic_error_rate)


if __name__ == "__main__":
    main()
