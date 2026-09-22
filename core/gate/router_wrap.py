"""Thin wrapper around laya.Router for checkpoint selection.

Kept separate from gate.py so Gate can be constructed with any object exposing
a `.predict(state, questions) -> dict` / `.system_one(state, questions) -> dict`
method -- a real laya.Router, a laya.Agent, or a test double.
"""
from typing import Optional


def default_router(preload: bool = True):
    """Lazily imports laya so `gut-check` can be unit-tested without torch installed."""
    import laya

    return laya.Router(preload=preload)


def load_agent(model_id_or_path: str = "convaiinnovations/laya", subfolder: Optional[str] = None):
    import laya

    return laya.load(model_id_or_path, subfolder=subfolder)
