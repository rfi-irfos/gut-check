"""Stage C: label mined candidates with a large LLM acting as teacher.

Replaces the weak heuristic label (`label_source: heuristic_structural_v2`)
with a real judgment call from a capable model, plus a short rationale so the
human-verification step (human_verify_sample.py) can check reasoning, not
just the bare label.

Mechanical leak prevention: refuses to label any candidate whose
(session_ref, claim_text) matches a gold_eval_set entry -- the gold set is
eval-only and must never end up as training data by accident.

Model-agnostic: any OpenAI-chat-compatible endpoint (OpenRouter, a direct
provider, a local server) works via --base-url/--api-key-env/--model.

Usage:
    export OPENROUTER_API_KEY=...
    python teacher_label.py --in candidates.jsonl --out teacher_labeled.jsonl \
        --model anthropic/claude-sonnet-4.5 --base-url https://openrouter.ai/api/v1 \
        --limit 100
"""
import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from gold_eval_set import is_leaked

SYSTEM_PROMPT = (
    "You are labeling training data for a claim-verification classifier. An AI agent "
    "made a status/completion claim about its own work. You're shown the claim and the "
    "nearest tool action + observation. Decide: does the observation confirm the claim "
    "(VERIFIED), show it's false or the action didn't succeed (CONTRADICTED), fail to "
    "establish it either way -- missing, unrelated, or insufficient evidence "
    "(UNVERIFIED), or is the claim not really a checkable status/completion claim at "
    "all -- a stated intention, a refusal, a conceptual remark (SKIP)? "
    "Be skeptical of surface-level signals: an is_error flag on an unrelated or "
    "procedural tool call does NOT mean the claim is false, and a claim that itself "
    "asserts a negative (\"X does not exist yet\") is VERIFIED when the evidence "
    "confirms that negative, not CONTRADICTED. "
    "Respond with strict JSON: {\"label\": \"VERIFIED|CONTRADICTED|UNVERIFIED|SKIP\", "
    "\"rationale\": \"one sentence\"}."
)


def build_user_prompt(candidate: dict) -> str:
    action = candidate.get("nearest_action") or {}
    obs = candidate.get("nearest_observation") or {}
    return (
        f"CLAIM:\n{candidate['claim_text']}\n\n"
        f"NEAREST ACTION: {action.get('type', '(none)')}\n\n"
        f"NEAREST OBSERVATION:\n{obs.get('excerpt', '(none)')}"
    )


def call_teacher(base_url: str, api_key: str, model: str, user_prompt: str,
                  max_retries: int = 3) -> dict:
    import urllib.request

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if parsed.get("label") not in ("VERIFIED", "CONTRADICTED", "UNVERIFIED", "SKIP"):
                raise ValueError(f"bad label: {parsed.get('label')!r}")
            return parsed
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt+1}/{max_retries} after error: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of candidates labeled (pilot runs).")
    ap.add_argument("--concurrency", type=int, default=8,
                     help="Parallel teacher calls. Network-bound work against a flaky/rate-limited "
                          "endpoint benefits a lot from this -- raise it if the provider tolerates more.")
    ap.add_argument("--skip-weak-unverified", action="store_true",
                     help="Skip candidates the heuristic already called UNVERIFIED with "
                          "no_observation_in_window -- no evidence exists to relabel.")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"error: ${args.api_key_env} is not set", file=sys.stderr)
        sys.exit(1)

    with open(args.inp) as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        candidates = candidates[: args.limit]

    # Resume-safe: a flaky teacher API means a run can be killed mid-batch.
    # Append rather than truncate, and skip whatever's already labeled on disk
    # (keyed by claim_text -- candidates don't carry a stable cross-run id).
    already_done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["claim_text"])
        print(f"resuming: {len(already_done)} already labeled in {args.out}", file=sys.stderr)

    n_skipped_leak = n_skipped_no_evidence = n_skipped_done = 0
    pending = []
    for c in candidates:
        if c["claim_text"] in already_done:
            n_skipped_done += 1
            continue
        session_ref = c.get("session_file") or c.get("session_id", "")
        if is_leaked(session_ref, c["claim_text"]):
            n_skipped_leak += 1
            continue
        if args.skip_weak_unverified and c.get("label_reason") == "no_observation_in_window":
            n_skipped_no_evidence += 1
            continue
        pending.append(c)

    n_labeled = n_errors = 0
    agree_with_heuristic = 0
    write_lock = threading.Lock()
    out_f = open(args.out, "a")

    def label_one(c):
        try:
            return c, call_teacher(args.base_url, api_key, args.model, build_user_prompt(c)), None
        except Exception as e:
            return c, None, e

    # Network I/O bound (each call is a slow, often-503ing remote request) --
    # concurrency turns a rate of ~1 item/minute into something that finishes
    # in a sane amount of time against a flaky provider, instead of paying
    # the full retry/backoff cost of every failure serially.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(label_one, c) for c in pending]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            c, teacher, err = fut.result()
            if err is not None:
                print(f"  teacher call failed permanently: {err}", file=sys.stderr)
                n_errors += 1
                continue
            c["teacher_label"] = teacher["label"]
            c["teacher_rationale"] = teacher.get("rationale", "")
            c["teacher_model"] = args.model
            c["label_source"] = "teacher_llm"
            if teacher["label"] == c.get("label"):
                agree_with_heuristic += 1
            with write_lock:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                out_f.flush()  # a killed/timed-out run must not lose progress already on disk
            n_labeled += 1

            if (i + 1) % 20 == 0:
                print(f"...{i+1}/{len(pending)} processed, {n_labeled} labeled", file=sys.stderr)
    out_f.close()

    print(f"\nlabeled:          {n_labeled}")
    print(f"skipped (gold leak): {n_skipped_leak}")
    print(f"skipped (no evidence): {n_skipped_no_evidence}")
    print(f"errors:           {n_errors}")
    if n_labeled:
        print(f"teacher agrees with weak heuristic: {agree_with_heuristic}/{n_labeled} = "
              f"{agree_with_heuristic/n_labeled:.1%}  (a low number here doesn't mean the "
              f"teacher is wrong -- it may mean the heuristic was wrong, which is the "
              f"whole reason this step exists; run human_verify_sample.py to find out which)")


if __name__ == "__main__":
    main()
