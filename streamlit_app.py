"""
Streamlit Chat Application for interacting with the RAG system.

This app loads all PDF files from a folder, builds a vector store, and allows
users to chat with the model via a modern chat UI using st.chat_message.
"""

import streamlit as st

from rag_utils import init_rag_system

# Directory containing your PDF documents.
PDF_DIR = "data/"


def initialize_session() -> None:
    """
    Initialize session state variables for the RAG system and chat history.
    """
    if "qa_chain" not in st.session_state:
        try:
            st.session_state["qa_chain"] = init_rag_system(PDF_DIR)
        except Exception as exc:
            st.error(f"Error initializing RAG system: {exc}")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def clear_chat() -> None:
    """
    Clear the chat history.
    """
    st.session_state["chat_history"] = []
    st.rerun()


def display_chat() -> None:
    """
    Display chat messages using st.chat_message for a modern UI.
    """
    for speaker, message in st.session_state["chat_history"]:
        if speaker.lower() == "user":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)


def main() -> None:
    st.title("RAG Chat App")
    st.write("Ask questions about the content of the PDF documents.")

    initialize_session()

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question")
        col1, col2 = st.columns(2)
        submit_button = col1.form_submit_button(label="Send")
        clear_chat_history = col2.form_submit_button("Clear Chat", type="tertiary")

    if clear_chat_history:
        clear_chat()

    if submit_button and user_input:
        qa_chain = st.session_state.get("qa_chain")
        if qa_chain is not None:
            st.session_state["chat_history"].append(("User", user_input))
            try:
                answer = qa_chain.invoke(user_input)
                st.session_state["chat_history"].append(("Bot", answer["result"]))
            except Exception as exc:
                st.error(f"Error during processing: {exc}")
        else:
            st.error("The RAG system is not properly initialized.")

    display_chat()


if __name__ == "__main__":
    main()
