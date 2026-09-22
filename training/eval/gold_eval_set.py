"""The versioned gold evaluation set: hand-verified (claim, context, label)
triples that NEVER get trained on. Seeded from the 55-item CAUSAL-1 spot
check; grows over time as Stage C's human-verification samples get added
(after confirming no training overlap).

The gold data itself is private (real session excerpts) and lives outside
this repo, like every other real-session artifact -- see .gitignore. This
module is the public loader/schema/leak-check code; point GOLD_PATH at your
own local file.
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

DEFAULT_GOLD_PATH = Path(__file__).parent / "local_data" / "gold_eval.jsonl"


def load_gold(path: Path = DEFAULT_GOLD_PATH) -> List[Dict]:
    """Each row: {id, claim_text, context, label, source, added_from}."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_keys(path: Path = DEFAULT_GOLD_PATH) -> Set[Tuple[str, str]]:
    """(session_ref, claim_text) pairs -- the leak-check key. A training
    labeler should refuse to emit output for any candidate matching one of
    these, mechanically, not just by policy."""
    return {(row.get("session_ref", ""), row["claim_text"]) for row in load_gold(path)}


def is_leaked(session_ref: str, claim_text: str, path: Path = DEFAULT_GOLD_PATH) -> bool:
    return (session_ref, claim_text) in gold_keys(path)


def ground_truth_dict(path: Path = DEFAULT_GOLD_PATH) -> Dict[str, str]:
    return {str(row["id"]): row["label"] for row in load_gold(path)}
