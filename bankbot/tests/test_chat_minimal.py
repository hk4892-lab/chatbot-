import os
import sys
from pathlib import Path

os.environ.setdefault("USE_SLM", "0")

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bankbot.app.api import ChatMessage, ChatRequest, chat  # noqa: E402


def test_chat_basic_query():
    request = ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are BankBot."),
            ChatMessage(role="user", content="What are your branch hours?"),
        ]
    )
    response = chat(request)
    assert response.reply
    assert response.lang in {"en", "hi", "ta"}
    assert response.source in {"rag", "slm", "tool", "clarify", "escalate"}
