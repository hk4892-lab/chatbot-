"""Streamlit UI for the banking chatbot."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from bankbot.app.core.engine import ChatEngine
from bankbot.app.core.render import format_answer


st.set_page_config(page_title="BankBot", page_icon="🏦", layout="wide")
st.title("🏦 BankBot Multilingual Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # type: ignore[assignment]


def get_engine() -> ChatEngine:
    if "engine" not in st.session_state:
        data_path = Path(__file__).resolve().parents[1] / "app" / "data"
        log_path = Path(__file__).resolve().parents[1] / "logs" / "audit.log"
        st.session_state.engine = ChatEngine(data_path=data_path, log_path=log_path)
    return st.session_state.engine  # type: ignore[return-value]

st.sidebar.header("Controls")
selected_language = st.sidebar.selectbox("Language preference", ["Auto", "English", "Hindi", "Tamil"], index=0)
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.18, 0.01)
use_st = st.sidebar.checkbox("Use sentence-transformers if available", value=False)
quick_action = st.sidebar.selectbox("Quick actions", ["None", "EMI calculator", "Block card", "Check UPI limit"], index=0)
if st.sidebar.button("Apply quick action") and quick_action != "None":
    if quick_action == "EMI calculator":
        st.session_state.chat_history.append({
            "speaker": "user",
            "message": "calculate emi P=500000 r=9.5 n=60",
        })
    elif quick_action == "Block card":
        st.session_state.chat_history.append({
            "speaker": "user",
            "message": "please block my card",
        })
    else:
        st.session_state.chat_history.append({
            "speaker": "user",
            "message": "upi limit kitna hai",
        })

st.info("This demo avoids storing your personal data. Don’t share sensitive info.")

chat_container = st.container()
with chat_container:
    for item in st.session_state.chat_history:
        if item["speaker"] == "user":
            st.markdown(f"**You:** {item['message']}")
        else:
            st.markdown(f"**Bot ({item['route']} · {item['confidence']:.2f}):** {item['message']}")
            if item.get("citations"):
                st.caption("Citations: " + ", ".join(f"{c['title']} ({c['id']})" for c in item["citations"]))
            if item.get("tool_result"):
                st.json(item["tool_result"])
            if item.get("clarifying_question"):
                st.caption(f"Follow-up: {item['clarifying_question']}")

user_message = st.text_input("Ask a question", key="user_input")
col1, col2 = st.columns([3, 1])
with col1:
    send_clicked = st.button("Send")
with col2:
    escalate_clicked = st.button("Escalate to human")

if send_clicked and user_message:
    st.session_state.chat_history.append({"speaker": "user", "message": user_message})
    engine = get_engine()
    lang_map = {
        "English": "EN",
        "Hindi": "HI",
        "Tamil": "TA",
    }
    lang_value = lang_map.get(selected_language) if selected_language != "Auto" else None
    result = engine.run(
        user_message,
        threshold=threshold,
        lang=lang_value,
        use_sentence_transformers=use_st,
    )
    decision = result.decision
    answer = decision.answer or ""
    if answer:
        answer = format_answer(answer, decision.citations)
    st.session_state.chat_history.append(
        {
            "speaker": "bot",
            "message": answer,
            "route": decision.route,
            "confidence": decision.confidence,
            "citations": decision.citations,
            "tool_result": decision.tool_result,
            "clarifying_question": decision.clarifying_question,
        }
    )

if escalate_clicked:
    st.session_state.chat_history.append(
        {
            "speaker": "bot",
            "message": "We've noted your request. A human specialist will reach out shortly.",
            "route": "escalate",
            "confidence": 1.0,
            "citations": [],
        }
    )
