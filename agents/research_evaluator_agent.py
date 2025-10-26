"""ResearchEvaluatorAgent: critiques and rates research answers."""

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


class ResearchEvaluatorAgent:
    """Grades research answers for completeness, accuracy, and clarity."""

    def __init__(
        self,
        *,
        openai_model_name: str = "gpt-5-nano",
        temperature: Optional[float] = None,
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        system_message = (
            "You are a meticulous research evaluator. Given a user's question and the assistant's answer, "
            "judge whether the response is accurate, complete, well-cited, and actionable. "
            "Return a JSON block with fields: rating (1-5), summary, strengths, weaknesses, recommendations."
        )

        self._model_client = self._build_openai_client(
            openai_model_name=openai_model_name,
            temperature=temperature,
        )
        self._assistant = AssistantAgent(
            name="research_evaluator",
            model_client=self._model_client,
            system_message=system_message,
            description="Evaluates research answers for quality.",
            tools=[],
            max_tool_iterations=1,
        )

    def evaluate(self, question: str, answer: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        if not answer or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")

        prompt = (
            f"USER QUESTION:\n{question}\n\nASSISTANT ANSWER:\n{answer}\n\n"
            "Provide the JSON evaluation as specified."
        )
        logger.info("ResearchEvaluatorAgent scoring response for question: %s", question)
        result = self._run_async(self._assistant.run(task=prompt))
        evaluation = self._extract_text(result.messages)
        logger.info("ResearchEvaluatorAgent output: %s", evaluation)
        return evaluation

    def _extract_text(self, messages: Iterable[Any]) -> str:
        last = self._last_chat_message(messages)
        return last.to_text().strip()

    @staticmethod
    def _last_chat_message(messages: Iterable[Any]) -> BaseChatMessage:
        for message in reversed(list(messages)):
            if isinstance(message, BaseChatMessage):
                return message
        raise RuntimeError("ResearchEvaluatorAgent did not produce a chat response.")

    @staticmethod
    def _build_openai_client(*, openai_model_name: str, temperature: Optional[float]) -> ChatCompletionClient:
        model_info: ModelInfo = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "openai",
        }
        kwargs: Dict[str, Any] = {
            "model": openai_model_name,
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            "include_name_in_message": False,
            "model_info": model_info,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return OpenAIChatCompletionClient(**kwargs)

    @staticmethod
    def _run_async(coro: asyncio.Future[str] | asyncio.coroutines.Coroutine[Any, Any, str]) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            if "event loop is running" in str(exc):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(coro)
            raise
