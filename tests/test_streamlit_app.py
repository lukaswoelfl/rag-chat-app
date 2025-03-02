import pytest
import streamlit as st

from streamlit_app import clear_chat, initialize_session


def test_initialize_session(monkeypatch):
    st.session_state.clear()
    # Monkey-patch init_rag_system to return a dummy chain.
    monkeypatch.setattr("streamlit_app.init_rag_system", lambda x: "dummy_chain")
    initialize_session()
    assert st.session_state["qa_chain"] == "dummy_chain"
    assert "chat_history" in st.session_state


def test_clear_chat(monkeypatch):
    st.session_state["chat_history"] = [("User", "Test message")]
    monkeypatch.setattr(st, "experimental_rerun", lambda: None)
    clear_chat()
    assert st.session_state["chat_history"] == []
