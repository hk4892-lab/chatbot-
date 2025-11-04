import os

os.environ.setdefault("USE_SLM", "0")

from bankbot.app.api import ChatRequest, chat


def test_chat_returns_answer() -> None:
    request = ChatRequest(
        messages=[{"role": "user", "content": "What is the fixed deposit interest rate right now?"}]
    )
    response = chat(request)
    assert response.reply
    assert response.source in {"rag", "tool", "clarify", "slm_rag", "escalate"}
    assert response.lang in {"en", "ta", "hi"}
