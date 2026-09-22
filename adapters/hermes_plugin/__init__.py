"""gut-check hermes-agent plugin.

Registers on the `pre_verify` hook (hermes_cli/plugins.py's VALID_HOOKS) --
fires once per turn right before a completion claim surfaces, the same point
`agent/verification_stop.py`'s own evidence-nudge already runs at. Calls the
gut-check sidecar (a separate process, see adapters/lauras_kernel_sidecar) on
the turn's final_response text; on low confidence, this is where escalation
to "System 2" happens -- but concretely, for this host, System 2 IS the same
big LLM hermes is already running the turn with. So the escalation call here
isn't a second LLM invocation gut-check makes itself: it's a `pre_verify`
"continue" directive that hands the uncertainty hint back to hermes' own
verify loop, which re-prompts its own model with that hint. This is the
idiomatic fit for hermes' actual hook contract, not a generic external
system_two_call like core/gate/escalation.py's default shape assumes --
see docs/architecture.md "hermes-agent: escalation realized as hook return".

Ships observation-only by default (GUT_CHECK_ENFORCE unset): every verdict is
logged, but pre_verify never returns a "continue" directive, so nothing about
the agent's behavior changes yet. Set GUT_CHECK_ENFORCE=1 to actually nudge
the agent to keep verifying on low-confidence claims -- do this only after a
burn-in period of reviewing the logged verdicts, per the project's
observation-before-enforcement design principle.

Known gap (documented, not yet built): `context` is currently built from
`changed_paths` alone. A fuller integration would pull the actual proven
evidence hermes-agent already assembles in `agent/verification_evidence.py`
(the SQLite ledger of classified command results) instead of just the file
list -- left for a follow-up pass once the sidecar's real-world verdict
quality has been checked against that richer context.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("gut_check_plugin")

DEFAULT_SIDECAR_URL = os.environ.get("GUT_CHECK_SIDECAR_URL", "http://localhost:8420")
ENFORCE = os.environ.get("GUT_CHECK_ENFORCE", "").strip().lower() in ("1", "true", "yes")
CONFIDENCE_THRESHOLD = float(os.environ.get("GUT_CHECK_CONFIDENCE_THRESHOLD", "0.55"))


def _call_sidecar(claim: str, context: str) -> Optional[Dict[str, Any]]:
    try:
        import urllib.request
        import json

        body = json.dumps({
            "claim": claim, "context": context,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }).encode()
        req = urllib.request.Request(
            f"{DEFAULT_SIDECAR_URL}/classify", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        # Never blocks the turn on sidecar unavailability -- mirrors
        # lauras-agents-gate's local-fallback-on-unreachable behavior.
        logger.info("gut-check sidecar unreachable (%s), skipping this turn's gate", e)
        return None


def _build_context(changed_paths: List[str]) -> str:
    if not changed_paths:
        return "(no changed files reported for this turn)"
    return "changed files this turn: " + ", ".join(changed_paths[:20])


def _pre_verify(*, session_id: str = "", platform: str = "", model: str = "",
                 coding: bool = False, attempt: int = 0, final_response: str = "",
                 changed_paths: Optional[List[str]] = None, **_ignored) -> Optional[Dict[str, str]]:
    if not final_response or len(final_response) < 20:
        return None

    context = _build_context(changed_paths or [])
    verdict = _call_sidecar(final_response, context)
    if verdict is None:
        return None

    logger.info(
        "gut-check verdict: session=%s model=%s choice=%s confidence=%.2f escalate=%s reason=%s",
        session_id, model, verdict.get("choice"), verdict.get("confidence", 0.0),
        verdict.get("escalate"), verdict.get("reason"),
    )

    if not verdict.get("escalate"):
        return None
    if not ENFORCE:
        return None  # observation-only: logged above, but doesn't change agent behavior yet

    hint = (
        f"A fast verification gate scored this response's completion claim as "
        f"{verdict.get('choice')} with only {verdict.get('confidence', 0.0):.2f} confidence "
        f"({verdict.get('reason')}). Before finishing, double-check the claim against real "
        f"evidence rather than trusting the summary you just wrote."
    )
    return {"action": "continue", "message": hint}


def register(ctx) -> None:
    ctx.register_hook("pre_verify", _pre_verify)
    mode = "ENFORCE" if ENFORCE else "observation-only"
    logger.info("gut-check plugin registered (sidecar=%s, mode=%s)", DEFAULT_SIDECAR_URL, mode)
