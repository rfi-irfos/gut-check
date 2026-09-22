"""Structural test with a mocked model -- no torch/laya model load needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

import service
from gate import Gate


class FakeRouter:
    def predict(self, state, questions):
        qid = next(iter(questions))
        return {"answers": {qid: {
            "type": "choice", "choice": "CONTRADICTED",
            "probabilities": {"VERIFIED": 0.1, "CONTRADICTED": 0.6, "UNVERIFIED": 0.2, "SKIP": 0.1},
            "confidence": 0.35, "action": {"act_probability": 0.5},
        }}}


def test_health_and_classify_with_mocked_gate():
    from fastapi.testclient import TestClient

    service._gate = Gate(router=FakeRouter())
    client = TestClient(service.app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

    r = client.post("/classify", json={"claim": "fixed the bug", "context": "tests: 2 failed"})
    assert r.status_code == 200
    body = r.json()
    assert body["choice"] == "CONTRADICTED"
    assert body["escalate"] is True  # confidence 0.35 < default 0.55 threshold


def test_health_reports_unloaded_when_model_never_set():
    service._gate = None
    service._load_error = "simulated load failure"
    from fastapi.testclient import TestClient
    client = TestClient(service.app)

    r = client.get("/health")
    assert r.json()["model_loaded"] is False

    r = client.post("/classify", json={"claim": "x", "context": "y"})
    assert r.status_code == 503
