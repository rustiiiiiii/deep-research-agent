"""
Trip Research Agent (Tavily + Groq)

- Built on Microsoft AutoGen (agentchat/core/ext stack).
- Groq LLM accessed via the OpenAI-compatible Groq endpoint.
- Tavily search tool exposed as an AutoGen function tool.

Required env:
  - GROQ_API_KEY
  - TAVILY_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Iterable, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseChatMessage
from autogen_core.models import ChatCompletionClient, ModelInfo
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tavily import TavilyClient

logger = logging.getLogger(__name__)


class ResearchAgent:
    """AutoGen agent for travel research using Groq + Tavily."""

    def __init__(
        self,
        *,
        temperature: float = 0.0,
        groq_model_name: str = "openai/gpt-oss-120b",
        verbose: bool = False,
        tavily_max_results: int = 5,
        tavily_include_raw_content: bool = False,
        tavily_kwargs: Optional[dict] = None,
        max_tool_iterations: int = 6,
    ) -> None:
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY is not set.")
        if not os.getenv("TAVILY_API_KEY"):
            raise EnvironmentError("TAVILY_API_KEY is not set.")

        self._verbose = verbose
        self._max_tool_iterations = max(1, max_tool_iterations)

        tavily_options = {
            "max_results": tavily_max_results,
            "include_raw_content": tavily_include_raw_content,
        }
        if tavily_kwargs:
            tavily_options.update(tavily_kwargs)
        self._tavily_options = tavily_options
        self._tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        self._model_client = self._build_groq_client(
            groq_model_name=groq_model_name,
            temperature=temperature,
        )

        system_message = (
            "You are a travel research assistant. "
            "Use the `web_search` tool to gather fresh information when helpful. "
            "Cite sources when available. "
            "Respond with your final synthesized answer prefixed by 'FINAL ANSWER:' "
            "and avoid adding additional turns after the final answer."
        )

        logger.info("Initializing AutoGen assistant with Groq model '%s'", groq_model_name)
        web_search_tool = FunctionTool(
            func=self._web_search,
            name="web_search",
            description=(
                "Search the web for up-to-date travel info (flights, hotels, attractions, "
                "practical tips, local regulations, etc.). Input should be a natural-language query."
            ),
        )

        self._assistant = AssistantAgent(
            name="research_assistant",
            model_client=self._model_client,
            system_message=system_message,
            description="Travel research specialist using Groq + Tavily.",
            tools=[web_search_tool],
            max_tool_iterations=self._max_tool_iterations,
        )

    def run(self, question: str) -> str:
        """Back-compat: delegate to .invoke()."""
        return self.invoke(question)

    def invoke(self, question: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        logger.info("Received query: %s", question)
        self._reset_conversation()

        try:
            answer = self._run_async(self._invoke_async(question))
            logger.info("Query processed successfully.")
            return answer
        except Exception:
            logger.exception("Error while invoking the research agent.")
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _invoke_async(self, question: str) -> str:
        result = await self._assistant.run(task=question)
        final_message = self._last_chat_message(result.messages)
        content = final_message.to_text().strip()
        if content.startswith("FINAL ANSWER:"):
            return content.split("FINAL ANSWER:", 1)[1].strip()
        return content

    def _run_async(self, coro: asyncio.Future[str] | asyncio.coroutines.Coroutine[Any, Any, str]) -> str:
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:  # pragma: no cover - defensive path
            if "event loop is running" in str(exc):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(coro)
            raise

    @staticmethod
    def _last_chat_message(messages: Iterable[Any]) -> BaseChatMessage:
        for message in reversed(list(messages)):
            if isinstance(message, BaseChatMessage):
                return message
        raise RuntimeError("Assistant did not produce a chat response.")

    def _reset_conversation(self) -> None:
        reset = getattr(self._assistant, "reset", None)
        if callable(reset):
            reset()

    def _web_search(self, query: str) -> Dict[str, Any]:
        logger.info("Executing Tavily search for: %s", query)
        response = self._tavily_client.search(query=query, **self._tavily_options)
        logger.info("Tavily response: %s", response)
        return response

    @staticmethod
    def _build_groq_client(
        *,
        groq_model_name: str,
        temperature: float,
    ) -> ChatCompletionClient:
        model_info: ModelInfo = {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
            "family": "groq",
        }
        return OpenAIChatCompletionClient(
            model=groq_model_name,
            api_key=os.environ["GROQ_API_KEY"],
            base_url=os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1"),
            temperature=temperature,
            include_name_in_message=False,
            model_info=model_info,
        )
