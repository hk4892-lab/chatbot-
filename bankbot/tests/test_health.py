import os

os.environ.setdefault("USE_SLM", "0")

from bankbot.app.api import healthz


def test_health_endpoint() -> None:
    payload = healthz()
    assert payload["ok"] is True
    assert isinstance(payload["rag_index"], bool)
    assert isinstance(payload["slm"], bool)
    assert isinstance(payload["embed_model"], str)
