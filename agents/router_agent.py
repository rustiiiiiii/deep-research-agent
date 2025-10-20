"""
RouterAgent: dispatcher that exposes Hotels/Flights/Places domain tools
and Tavily web search via Microsoft AutoGen with Groq LLM.
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

from ..domain.flights import search_flights
from ..domain.hotels import search_hotels
from ..domain.locations import search_places

logger = logging.getLogger(__name__)


class RouterAgent:
    def __init__(
        self,
        *,
        groq_model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
        verbose: bool = False,
        tavily_max_results: int = 5,
        tavily_include_raw_content: bool = False,
        tavily_kwargs: Optional[dict] = None,
        max_tool_iterations: int = 8,
    ):
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY not set in environment.")
        if not os.getenv("TAVILY_API_KEY"):
            raise EnvironmentError("TAVILY_API_KEY not set in environment.")

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
            "You are a travel planning router. "
            "Use the specialized tools (`hotels.search`, `flights.search`, `places.search`, `web.search`) "
            "to gather structured information, then synthesize a helpful answer. "
            "Finish by summarizing the recommendations prefixed with 'FINAL ANSWER:' "
            "and avoid continuing the conversation afterwards."
        )

        tools = [
            FunctionTool(
                func=self._hotels_search,
                name="hotels.search",
                description=(
                    "Look up hotels. Provide destination, check-in/out dates (YYYY-MM-DD), "
                    "guest count, and optional budget/stars filters."
                ),
            ),
            FunctionTool(
                func=self._flights_search,
                name="flights.search",
                description=(
                    "Find flight options. Requires origin, destination (IATA codes), "
                    "depart date (YYYY-MM-DD), optional return_date, passenger count, and cabin."
                ),
            ),
            FunctionTool(
                func=self._places_search,
                name="places.search",
                description=(
                    "Recommend attractions or places. Needs destination and optional filters "
                    "like themes, open_now, with_kids."
                ),
            ),
            FunctionTool(
                func=self._web_search,
                name="web.search",
                description="General-purpose web search for supplemental information.",
            ),
        ]

        self._assistant = AssistantAgent(
            name="router_assistant",
            model_client=self._model_client,
            system_message=system_message,
            description="Router that decides which travel tool to call.",
            tools=tools,
            max_tool_iterations=self._max_tool_iterations,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, query: str) -> str:
        return self.invoke(query)

    def invoke(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        logger.info("RouterAgent processing query: %s", query)
        self._reset_conversation()
        try:
            answer = self._run_async(self._invoke_async(query))
            logger.info("RouterAgent delivered response successfully.")
            return answer
        except Exception:
            logger.exception("RouterAgent failed to process the query.")
            raise

    # ------------------------------------------------------------------
    # Internal execution helpers
    # ------------------------------------------------------------------
    async def _invoke_async(self, query: str) -> str:
        result = await self._assistant.run(task=query)
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
        raise RuntimeError("Router assistant produced no chat response.")

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    def _hotels_search(
        self,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int,
        budget_min: Optional[int] = None,
        budget_max: Optional[int] = None,
        stars: Optional[list[int]] = None,
    ) -> Dict[str, Any]:
        logger.info("RouterAgent running hotels.search for %s", destination)
        results = search_hotels(
            destination=destination,
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            budget_min=budget_min,
            budget_max=budget_max,
            stars=stars,
        )
        return {"results": results}

    def _flights_search(
        self,
        origin: str,
        destination: str,
        depart: str,
        pax: int,
        cabin: str = "economy",
        return_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("RouterAgent running flights.search %s -> %s", origin, destination)
        options = search_flights(
            origin=origin,
            destination=destination,
            depart=depart,
            return_=return_date,
            pax=pax,
            cabin=cabin,
        )
        return {"options": options}

    def _places_search(
        self,
        destination: str,
        themes: Optional[list[str]] = None,
        open_now: Optional[bool] = None,
        with_kids: Optional[bool] = None,
    ) -> Dict[str, Any]:
        logger.info("RouterAgent running places.search for %s", destination)
        results = search_places(
            destination=destination,
            themes=themes,
            open_now=open_now,
            with_kids=with_kids,
        )
        return {"results": results}

    def _web_search(self, query: str) -> Dict[str, Any]:
        logger.info("RouterAgent running web.search for %s", query)
        return self._tavily_client.search(query=query, **self._tavily_options)

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------
    def _reset_conversation(self) -> None:
        reset = getattr(self._assistant, "reset", None)
        if callable(reset):
            reset()

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
