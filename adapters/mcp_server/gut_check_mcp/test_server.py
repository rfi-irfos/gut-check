import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))


def test_verify_claim_returns_expected_shape():
    with patch("server.get_gate") as mock_get_gate:
        mock_verdict = MagicMock()
        mock_verdict.choice = "CONTRADICTED"
        mock_verdict.confidence = 0.42
        mock_verdict.escalate = True
        mock_verdict.reason = "confidence 0.42 below threshold 0.55"
        mock_gate = MagicMock()
        mock_gate.classify.return_value = mock_verdict
        mock_get_gate.return_value = mock_gate

        from server import verify_claim
        result = verify_claim("The fix is deployed.", "deploy log shows an error")

        assert result == {
            "choice": "CONTRADICTED",
            "confidence": 0.42,
            "escalate": True,
            "reason": "confidence 0.42 below threshold 0.55",
        }
        mock_gate.classify.assert_called_once()
