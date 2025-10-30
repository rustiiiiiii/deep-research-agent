"""
Streamlit entry point for the Trip Research Agent.

Provides a chat-style interface on top of `ResearchAgent`, preserving the
conversation history per browser session and surfacing the most recent tool
activity in sidebar expanders.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from agents.research_agent import ResearchAgent

# Ensure environment variables from .env are loaded before instantiating the agent.
load_dotenv()

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _get_agent() -> ResearchAgent:
    """Create a singleton ResearchAgent per Streamlit process."""
    return ResearchAgent()


def _init_session_state() -> None:
    """Initialize keys stored in st.session_state."""
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "metadata" not in st.session_state:
        st.session_state.metadata = {
            "research_questions": [],
            "action_history": [],
        }


def _render_sidebar() -> None:
    """Render sidebar controls and metadata viewers."""
    with st.sidebar:
        st.header("Session Controls")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.metadata = {
                "research_questions": [],
                "action_history": [],
            }
            st.experimental_rerun()

        st.divider()
        st.header("Latest Run Details")
        questions = st.session_state.metadata.get("research_questions", [])
        with st.expander("Research questions", expanded=False):
            if questions:
                for idx, question in enumerate(questions, start=1):
                    st.markdown(f"{idx}. {question}")
            else:
                st.caption("No questions generated yet.")

        actions = st.session_state.metadata.get("action_history", [])
        with st.expander("Tool calls", expanded=False):
            if actions:
                for action in actions:
                    st.markdown(f"**{action.get('type', 'tool')}** — {action.get('query', '')}")
                    results = action.get("results") or []
                    for result in results:
                        title = result.get("title", "Result")
                        summary = result.get("summary", result.get("content", ""))
                        url = result.get("url")
                        line = f"- {title}: {summary}"
                        if url:
                            line += f"  \n  [{url}]({url})"
                        st.markdown(line)
                    st.markdown("---")
            else:
                st.caption("No tool calls recorded yet.")


def main() -> None:
    st.set_page_config(
        page_title=" Research Agent",
        layout="wide",
    )

    st.title("Research Agent")
    st.caption(
        "Ask travel questions and the agent will research live sources before drafting a report."
    )

    _init_session_state()
    _render_sidebar()

    try:
        agent = _get_agent()
    except Exception as exc:  # pragma: no cover - defensive guard for UI
        error_message = (
            "Failed to initialize the research agent. "
            "Verify API keys in your environment and restart the app.\n\n"
            f"Details: {exc}"
        )
        LOGGER.exception("Streamlit failed to initialize ResearchAgent: %s", exc)
        st.error(error_message)
        return

    # Replay the chat history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Where can I help you plan your next trip?")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Researching your question..."):
            try:
                response = agent.invoke(prompt)
            except Exception as exc:  # pragma: no cover - surfaced to UI
                LOGGER.exception("Agent invocation failed: %s", exc)
                response = f"An error occurred while researching your question:\n\n{exc}"
        placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.metadata["research_questions"] = agent.last_research_questions
    st.session_state.metadata["action_history"] = agent.action_history


if __name__ == "__main__":
    main()
