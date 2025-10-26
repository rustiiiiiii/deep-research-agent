"""IntentAgent: lightweight gatekeeper that checks question clarity."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Iterable, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseChatMessage
from autogen_core.models import ChatCompletionClient, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

logger = logging.getLogger(__name__)


class IntentAgent:
    """Determines whether the user's query is ready for research or needs clarification."""

    def __init__(
        self,
        *,
        openai_model_name: str = "gpt-5-nano",
        temperature: Optional[float] = None,
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        system_message = (
            "You evaluate user questions before they are sent to the research agent. "
            "If the question is clear and specific, respond with 'READY:' followed by the cleaned question. "
            "If the question is ambiguous, missing key details, or potentially disallowed, respond with "
            "'CLARIFY:' followed by a short clarification request.")

        self._model_client = self._build_openai_client(
            openai_model_name=openai_model_name,
            temperature=temperature,
        )
        self._assistant = AssistantAgent(
            name="intent_checker",
            model_client=self._model_client,
            system_message=system_message,
            description="Checks question clarity and requests clarifications when needed.",
            tools=[],
            max_tool_iterations=1,
        )

    def analyze(self, question: str) -> Dict[str, str]:
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        logger.info("IntentAgent evaluating question: %s", question)
        result = self._run_async(self._assistant.run(task=question))
        final = self._extract_text(result.messages)
        logger.info("IntentAgent raw output: %s", final)
        if final.upper().startswith("READY:"):
            return {"status": "ready", "question": final.split(":", 1)[1].strip()}
        if final.upper().startswith("CLARIFY:"):
            return {"status": "clarify", "message": final.split(":", 1)[1].strip()}
        # Fallback: treat as clarify so user sees the message
        return {"status": "clarify", "message": final.strip()}

    def _extract_text(self, messages: Iterable[Any]) -> str:
        last = self._last_chat_message(messages)
        return last.to_text().strip()

    @staticmethod
    def _last_chat_message(messages: Iterable[Any]) -> BaseChatMessage:
        for message in reversed(list(messages)):
            if isinstance(message, BaseChatMessage):
                return message
        raise RuntimeError("IntentAgent did not produce a chat response.")

    @staticmethod
    def _build_openai_client(*, openai_model_name: str, temperature: Optional[float]) -> ChatCompletionClient:
        model_info: ModelInfo = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "openai",
        }
        client_kwargs = {
            "model": openai_model_name,
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            "include_name_in_message": False,
            "model_info": model_info,
        }
        if temperature is not None:
            client_kwargs["temperature"] = temperature
        return OpenAIChatCompletionClient(**client_kwargs)

    @staticmethod
    def _run_async(coro: asyncio.Future[str] | asyncio.coroutines.Coroutine[Any, Any, str]) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            if "event loop is running" in str(exc):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(coro)
            raise
