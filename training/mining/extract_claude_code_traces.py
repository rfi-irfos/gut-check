"""Stage A: mine (claim, nearest_action, nearest_observation) candidates from
Claude Code session transcripts (~/.claude/projects/**/*.jsonl).

A cleaned, open-sourceable vendor of the private extract_session_traces_v2.py
(originally at ~/Desktop/kindom/execution/causal1/), factored to share its
claim/relevance/label logic with extract_hermes_traces.py via
common_relevance_filter.py. Paths are CLI args here rather than hardcoded,
since this script -- unlike the transcripts it reads -- is meant to be public.

Usage:
    python extract_claude_code_traces.py --sessions-glob '~/.claude/projects/**/*.jsonl' \
        --out candidates.jsonl
"""
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common_relevance_filter import (
    has_lexical_overlap,
    is_claim,
    is_negated,
    is_policy_event,
    score_pass_fail,
)

LOOKBACK_EVENTS = 15


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def iter_events(path):
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(d.get("timestamp", ""))
            if ts is None:
                continue
            t = d.get("type")
            if t == "assistant":
                for c in d.get("message", {}).get("content", []) or []:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "tool_use":
                        yield ts, "tool_use", {"name": c.get("name"), "input": c.get("input")}
                    elif c.get("type") == "text" and len(c.get("text", "")) > 20:
                        yield ts, "claim_candidate", {"text": c["text"]}
            elif t == "user":
                content = d.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            txt = tool_result_text(c.get("content"))
                            yield ts, "tool_result", {"excerpt": txt[:1500]}


def process_session(path):
    events = sorted(iter_events(path), key=lambda e: e[0])
    candidates = []
    for i, (ts, kind, payload) in enumerate(events):
        if kind != "claim_candidate":
            continue
        claim_text = payload["text"]
        if not is_claim(claim_text):
            continue
        if is_negated(claim_text):
            continue

        window = events[max(0, i - LOOKBACK_EVENTS):i]
        nearest_result = None
        nearest_action = None
        for wts, wkind, wpayload in reversed(window):
            if wkind == "tool_result" and nearest_result is None:
                nearest_result = (wts, wpayload)
            if wkind == "tool_use" and nearest_action is None:
                nearest_action = (wts, wpayload)
            if nearest_result and nearest_action:
                break

        if nearest_result is None:
            label, reason = "UNVERIFIED", "no_observation_in_window"
        else:
            excerpt = nearest_result[1]["excerpt"]
            if is_policy_event(excerpt):
                continue

            action_str = json.dumps(nearest_action[1].get("input")) if nearest_action else ""
            if not has_lexical_overlap(claim_text, excerpt + " " + action_str):
                label, reason = "UNVERIFIED", "no_lexical_overlap_with_nearest_evidence"
            else:
                label, reason = score_pass_fail(excerpt)
                if label is None:
                    label, reason = "UNVERIFIED", "ambiguous_pass_and_fail_signals_both_or_neither"

        candidates.append({
            "source": "claude_code",
            "claim_text": claim_text[:500],
            "claim_timestamp": ts.isoformat(),
            "nearest_action": {
                "type": nearest_action[1]["name"],
                "timestamp": nearest_action[0].isoformat(),
            } if nearest_action else None,
            "nearest_observation": {
                "excerpt": nearest_result[1]["excerpt"][:300],
                "timestamp": nearest_result[0].isoformat(),
            } if nearest_result else None,
            "label": label,
            "label_reason": reason,
            "label_source": "heuristic_structural_v2",
            "confidence": "weak",
            "session_file": Path(path).name,
        })
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-glob", required=True,
                     help="e.g. '~/.claude/projects/**/*.jsonl' (quote it so your shell doesn't expand **)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = glob.glob(str(Path(args.sessions_glob).expanduser()), recursive=True)
    print(f"found {len(files)} session files", file=sys.stderr)

    total = 0
    label_counts = {}
    reason_counts = {}
    with open(args.out, "w") as out:
        for n, path in enumerate(files):
            try:
                candidates = process_session(path)
            except Exception as e:
                print(f"skip {path}: {e}", file=sys.stderr)
                continue
            for c in candidates:
                out.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += 1
                label_counts[c["label"]] = label_counts.get(c["label"], 0) + 1
                reason_counts[c["label_reason"]] = reason_counts.get(c["label_reason"], 0) + 1
            if (n + 1) % 500 == 0:
                print(f"...{n+1}/{len(files)} sessions, {total} candidates so far", file=sys.stderr)

    print(f"wrote {total} weak candidates to {args.out}", file=sys.stderr)
    print(f"label distribution: {label_counts}", file=sys.stderr)
    print(f"reason distribution: {reason_counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
