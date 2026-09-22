"""Stage B: mine (claim, nearest_action, nearest_observation) candidates from
Hermes conversation history (~/.hermes/state.db, SQLite).

Primary training-data source per project decision: Hermes logs many different
LLM providers/models switching within and across sessions (sessions.model),
giving failure-signature diversity so the classifier doesn't overfit to one
model's writing style -- unlike the single-model Claude Code corpus.

Schema (confirmed against a real state.db):
  sessions(id, source, model, cwd, started_at, ...)
  messages(id, session_id, role, content, tool_call_id, tool_calls, tool_name,
           timestamp, effect_disposition, ...)
  roles seen: assistant, tool, user, session_meta.
  timestamp is a REAL unix epoch (seconds).

`effect_disposition` exists but is unpopulated in every real database checked
so far -- do not treat it as a label source, it's a schema hook for something
that was never wired up (possibly by this very project, later).

`source=cron` sessions (no live user in the loop) are carried through with
their source tag but NOT auto-included in training by default -- see
--include-cron. Project decision: hold them out as a distribution-shift eval
stratum unless explicitly requested.

Usage:
    python extract_hermes_traces.py --db ~/.hermes/state.db --out candidates.jsonl
"""
import argparse
import json
import sqlite3
import sys
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


def iter_session_events(rows):
    """rows: list of message rows for one session, already ordered by timestamp.
    Yields (ts, kind, payload) in the same shape extract_claude_code_traces.py uses,
    so both scripts can share the same claim/relevance logic."""
    for row in rows:
        role, content, tool_call_id, tool_calls_json, tool_name, ts = (
            row["role"], row["content"], row["tool_call_id"], row["tool_calls"],
            row["tool_name"], row["timestamp"],
        )
        if role == "assistant":
            if tool_calls_json:
                try:
                    calls = json.loads(tool_calls_json)
                except (json.JSONDecodeError, TypeError):
                    calls = []
                for c in calls:
                    fn = (c.get("function") or {})
                    yield ts, "tool_use", {"name": fn.get("name"), "input": fn.get("arguments")}
            if content and len(content) > 20:
                yield ts, "claim_candidate", {"text": content}
        elif role == "tool":
            yield ts, "tool_result", {"excerpt": (content or "")[:1500]}


def process_session(con, session_id: str, source: str, model: str):
    cur = con.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name, timestamp "
        "FROM messages WHERE session_id = ? AND active = 1 ORDER BY timestamp ASC",
        (session_id,),
    )
    rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    events = sorted(iter_session_events(rows), key=lambda e: e[0] or 0)

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

            action_str = str(nearest_action[1].get("input")) if nearest_action else ""
            if not has_lexical_overlap(claim_text, excerpt + " " + action_str):
                label, reason = "UNVERIFIED", "no_lexical_overlap_with_nearest_evidence"
            else:
                label, reason = score_pass_fail(excerpt)
                if label is None:
                    label, reason = "UNVERIFIED", "ambiguous_pass_and_fail_signals_both_or_neither"

        candidates.append({
            "source": "hermes",
            "hermes_source": source,
            "hermes_model": model,
            "claim_text": claim_text[:500],
            "claim_timestamp": ts,
            "nearest_action": {
                "type": nearest_action[1]["name"],
                "timestamp": nearest_action[0],
            } if nearest_action else None,
            "nearest_observation": {
                "excerpt": nearest_result[1]["excerpt"][:300],
                "timestamp": nearest_result[0],
            } if nearest_result else None,
            "label": label,
            "label_reason": reason,
            "label_source": "heuristic_structural_v2",
            "confidence": "weak",
            "session_id": session_id,
        })
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-cron", action="store_true",
                     help="Include source=cron sessions in the output (default: excluded, "
                          "held out as a distribution-shift eval stratum per project decision).")
    args = ap.parse_args()

    con = sqlite3.connect(str(Path(args.db).expanduser()))
    where = "" if args.include_cron else "WHERE source != 'cron'"
    sessions = con.execute(f"SELECT id, source, model FROM sessions {where}").fetchall()
    print(f"found {len(sessions)} sessions"
          f"{' (cron included)' if args.include_cron else ' (cron excluded, held out separately)'}",
          file=sys.stderr)

    total = 0
    label_counts = {}
    model_counts = {}
    with open(args.out, "w") as out:
        for n, (session_id, source, model) in enumerate(sessions):
            try:
                candidates = process_session(con, session_id, source, model)
            except Exception as e:
                print(f"skip session {session_id}: {e}", file=sys.stderr)
                continue
            for c in candidates:
                out.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += 1
                label_counts[c["label"]] = label_counts.get(c["label"], 0) + 1
                model_counts[model] = model_counts.get(model, 0) + 1
            if (n + 1) % 200 == 0:
                print(f"...{n+1}/{len(sessions)} sessions, {total} candidates so far", file=sys.stderr)

    print(f"wrote {total} weak candidates to {args.out}", file=sys.stderr)
    print(f"label distribution: {label_counts}", file=sys.stderr)
    print(f"distinct models represented: {len(model_counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
