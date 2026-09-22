"""Unit tests against a mocked sidecar call and a fake PluginContext -- no
real hermes-agent install or running sidecar needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import hermes_plugin as plugin  # noqa: E402  (the package under adapters/hermes_plugin/__init__.py)


def test_no_escalation_below_confidence_threshold_stays_silent(monkeypatch):
    monkeypatch.setattr(plugin, "_call_sidecar", lambda claim, ctx: {
        "choice": "VERIFIED", "confidence": 0.9, "escalate": False, "reason": "",
    })
    result = plugin._pre_verify(final_response="all tests pass, fixed the bug for real")
    assert result is None


def test_escalation_observed_but_silent_by_default(monkeypatch):
    monkeypatch.setattr(plugin, "_call_sidecar", lambda claim, ctx: {
        "choice": "CONTRADICTED", "confidence": 0.1, "escalate": True, "reason": "low confidence",
    })
    monkeypatch.setattr(plugin, "ENFORCE", False)
    result = plugin._pre_verify(final_response="all tests pass, fixed the bug for real")
    assert result is None  # observation-only: logged, but no behavior change


def test_escalation_returns_continue_directive_when_enforced(monkeypatch):
    monkeypatch.setattr(plugin, "_call_sidecar", lambda claim, ctx: {
        "choice": "CONTRADICTED", "confidence": 0.1, "escalate": True, "reason": "low confidence",
    })
    monkeypatch.setattr(plugin, "ENFORCE", True)
    result = plugin._pre_verify(final_response="all tests pass, fixed the bug for real")
    assert result is not None
    assert result["action"] == "continue"
    assert "CONTRADICTED" in result["message"]


def test_sidecar_unreachable_never_blocks_turn(monkeypatch):
    monkeypatch.setattr(plugin, "_call_sidecar", lambda claim, ctx: None)
    result = plugin._pre_verify(final_response="all tests pass, fixed the bug for real")
    assert result is None


def test_register_wires_pre_verify_hook():
    calls = []

    class FakeCtx:
        def register_hook(self, name, cb):
            calls.append((name, cb))

    plugin.register(FakeCtx())
    assert calls[0][0] == "pre_verify"
    assert calls[0][1] is plugin._pre_verify
