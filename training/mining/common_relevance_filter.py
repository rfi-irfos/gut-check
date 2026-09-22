"""Shared claim-detection, relevance, and label-signal logic used by every
corpus-specific extractor (extract_claude_code_traces.py, extract_hermes_traces.py).

Factored out so both extractors share one implementation instead of drifting
into two copies -- the same discipline lauras-agents-kernel already applies
to its own shared event-mapping code.

Originally developed as extract_session_traces_v2.py, revised against a
55-item human spot-check that found the v1 heuristic wrong on ~47% of a
stratified sample. Four concrete error classes came out of that review, and
this module addresses each one directly:

1. is_error is not a reliable failure signal for procedural/policy tool
   events (ExitPlanMode, permission-classifier blocks). Those carry no
   evidence about whether the CLAIM's effect happened, and get excluded
   entirely rather than auto-labeled.

2. Loose "error word anywhere in the output" matching fires even when the
   same output also carries an explicit positive verdict. POSITIVE and
   NEGATIVE signal are scored independently from the observation text, and a
   label only commits when exactly one fires; both-or-neither goes to
   UNVERIFIED instead of guessing.

3. Negated claims ("existiert noch nicht", "ist nicht behoben") get flagged
   and routed to human review rather than auto-labeled with the polarity
   mechanically flipped, which is fragile.

4. The single biggest error source was topical irrelevance: the nearest
   tool_result in the event stream was frequently about something else
   entirely. A cheap lexical relevance filter (shared significant token
   between claim and the nearest action input / observation excerpt) is a
   precondition for keeping a candidate as VERIFIED/CONTRADICTED at all.

Also filtered out at generation time: refusals, stated intentions
("I'll now...", "let me..."), and conceptual/reflective text that isn't a
status claim about a completed effect at all.
"""
import re

CLAIM_SIGNAL = re.compile(
    r"\b(fixed|deployed|is live|now live|completed|resolved|committed and pushed|"
    r"successfully|merged|tests? pass(es|ed)?|behoben|erledigt|gefixt|fertig|"
    r"funktioniert (jetzt|wieder)|erfolgreich|gemerged|verifiziert|bestätigt)\b",
    re.IGNORECASE,
)

# class 3: negation near the claim signal -- polarity is ambiguous, route to
# human instead of guessing
NEGATION = re.compile(
    r"\b(nicht|kein|keine|noch nicht|not yet|doesn'?t|isn'?t|nie|never|without|"
    r"existiert (noch )?nicht)\b",
    re.IGNORECASE,
)

# excluded from claim consideration entirely: not status claims about a
# completed effect
NON_CLAIM = re.compile(
    r"^\s*(ich werde|let me|i'?ll |i will |i'?m going to|kurz und klar[:,]? das mache ich nicht|"
    r"won'?t|das ist eine (klare|wichtige)|starte (jetzt|damit)|"
    r"bevor ich|before i)\b",
    re.IGNORECASE,
)

# class 1: tool/system events whose content is procedural, not evidence about
# the claim's effect
POLICY_EVENT = re.compile(
    r"(permission for this action was denied by the claude code auto mode classifier|"
    r"user has approved your plan)",
    re.IGNORECASE,
)

# class 2: score positive/negative independently instead of one loose regex
STRICT_FAIL = re.compile(
    r"(Traceback \(most recent call last\)|Exit code [1-9]|^FAILED\b|AttributeError:|"
    r"ImportError:|TypeError:|ValueError:|Exception:|refusing to |BLOCKED:|"
    r"550 5\.\d\.\d|Recipient address rejected|could not be found)",
    re.MULTILINE,
)
STRICT_PASS = re.compile(
    r"(\ball pass\b|\d+ passed\b|\bOK\b$|\bPASS\b|0 failed|correctly rejected|"
    r"successfully|Finished `(test|dev)` profile|Complete job)",
    re.MULTILINE | re.IGNORECASE,
)

STOPWORDS = set(
    "the a an is are was were und der die das ein eine ist sind war waren "
    "jetzt now then dann bevor before let me ich du wir du er sie es this "
    "that these those mit von zu für auf aus im in am an on at for with of to".split()
)


def significant_tokens(text: str) -> set:
    words = re.findall(r"[a-zA-ZäöüÄÖÜß_./-]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def is_claim(text: str) -> bool:
    """A status/completion claim, not a stated intention/refusal/conceptual remark."""
    if not CLAIM_SIGNAL.search(text):
        return False
    if NON_CLAIM.search(text.strip()):
        return False
    return True


def is_negated(text: str, window_chars: int = 40) -> bool:
    """True if a negation word appears just before the claim-signal match --
    polarity is ambiguous, caller should route to human review rather than
    auto-label."""
    m = CLAIM_SIGNAL.search(text)
    if not m:
        return False
    window_before = text[max(0, m.start() - window_chars):m.start()]
    return bool(NEGATION.search(window_before))


def is_policy_event(observation_text: str) -> bool:
    return bool(POLICY_EVENT.search(observation_text))


def has_lexical_overlap(claim_text: str, evidence_text: str) -> bool:
    return bool(significant_tokens(claim_text) & significant_tokens(evidence_text))


def score_pass_fail(observation_text: str):
    """Returns (label, reason) or (None, None) if ambiguous (both or neither
    signal present) -- caller should fall back to UNVERIFIED on None."""
    has_fail = bool(STRICT_FAIL.search(observation_text))
    has_pass = bool(STRICT_PASS.search(observation_text))
    if has_fail and not has_pass:
        return "CONTRADICTED", "strict_fail_signal_no_pass_signal"
    if has_pass and not has_fail:
        return "VERIFIED", "strict_pass_signal_no_fail_signal"
    return None, None
