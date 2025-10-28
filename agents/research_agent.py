"""
Trip Research Agent (Tavily + OpenAI)

- Built on Microsoft AutoGen (agentchat/core/ext stack).
- OpenAI GPT-5 Nano (or any compatible model) accessed via the OpenAI API.
- Tavily search tool exposed as an AutoGen function tool.

Required env:
  - OPENAI_API_KEY
  - TAVILY_MCP_BASE_URL
  - TAVILY_MCP_API_KEY (optional, falls back to TAVILY_API_KEY)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.models import ChatCompletionClient, ModelInfo
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import ValidationError

from .mcp_client import MCPServerConfig, MCPToolClient

logger = logging.getLogger(__name__)


class ResearchAgent:
    """AutoGen agent for travel research using OpenAI + Tavily."""

    def __init__(
        self,
        *,
        temperature: Optional[float] = None,
        openai_model_name: str = "gpt-5-nano",
        verbose: bool = False,
        tavily_max_results: int = 5,
        tavily_include_raw_content: bool = False,
        tavily_kwargs: Optional[dict] = None,
        tavily_mcp_base_url: Optional[str] = None,
        tavily_mcp_api_key: Optional[str] = None,
        tavily_mcp_tool_name: str = "tavily.search",
        max_tool_iterations: int = 6,
        group_chat_max_turns: int = 8,
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        default_local_url = "http://127.0.0.1:6112/mcp"
        tavily_mcp_base_url = tavily_mcp_base_url or os.getenv("TAVILY_MCP_BASE_URL", default_local_url)
        if not tavily_mcp_base_url:
            raise EnvironmentError("TAVILY_MCP_BASE_URL is not set.")
        tavily_mcp_api_key = tavily_mcp_api_key or os.getenv("TAVILY_MCP_API_KEY") or os.getenv("TAVILY_API_KEY")

        self._verbose = verbose
        self._max_tool_iterations = max(1, max_tool_iterations)
        self._group_chat_max_turns = max(2, group_chat_max_turns)

        tavily_options = {
            "max_results": tavily_max_results,
            "include_raw_content": tavily_include_raw_content,
        }
        if tavily_kwargs:
            tavily_options.update(tavily_kwargs)
        self._tavily_options = tavily_options
        self._tavily_tool = MCPToolClient(
            config=MCPServerConfig(
                base_url=tavily_mcp_base_url,
                api_key=tavily_mcp_api_key,
                tool_name=tavily_mcp_tool_name,
            )
        )

        current_dt_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        current_day = datetime.utcnow().strftime("%A")

        self._model_client = self._build_openai_client(
            openai_model_name=openai_model_name,
            temperature=temperature,
        )

        self._group_search_tool = FunctionTool(
            func=self._run_tavily_group_search,
            name=tavily_mcp_tool_name,
            description=(
                "Call the Tavily search service. Provide a focused query string and you will "
                "receive a concise list of high-quality web results summarised for you."
            ),
        )

        self._question_agent = AssistantAgent(
            name="question_breakdown",
            model_client=self._model_client,
            system_message=(
                "You analyze the user's research request and break it into 4-5 concise, targeted sub-questions. "
                "Each sub-question should be specific enough to research independently while collectively addressing the original request. "
                "Respond ONLY with a JSON array of strings."
            ),
            description="Decomposes the primary request into focused research questions.",
            tools=[],
            max_tool_iterations=1,
        )

        self._search_agent = AssistantAgent(
            name="search_query_generator",
            model_client=self._model_client,
            system_message=(
                "You craft focused web search queries. Given a user question, list 3-4 short, precise search queries that "
                "will retrieve trustworthy, fresh information. Respond ONLY with a JSON array of strings."
            ),
            description="Generates targeted search queries with no tool usage.",
            tools=[],
            max_tool_iterations=1,
        )

        system_message = (
            f"You are a meticulous research analyst. The current UTC datetime is {current_dt_iso} and today is {current_day}. "
            "You will be given the user question and compiled research notes. Synthesize them into a concise, well-structured report "
            "that: (1) directly answers the user, (2) highlights 2-3 key takeaways, (3) includes actionable guidance when applicable, "
            "and (4) cites every factual statement with `[source name]`. Close with the prefix 'FINAL ANSWER:' and nothing after it."
        )

        logger.info("Initializing report agent with OpenAI model '%s'", openai_model_name)
        self._report_agent = AssistantAgent(
            name="research_reporter",
            model_client=self._model_client,
            system_message=system_message,
            description="Synthesizes gathered evidence into a final report.",
            tools=[],
            max_tool_iterations=1,
        )

        planner_system_message = (
            "You are a research planner collaborating with the `research_writer` in a Microsoft AutoGen group chat. "
            "Always acknowledge the history of actions already taken before deciding what to do next. "
            "When a user question arrives, break it into concrete search tasks by proposing a JSON array of concise, "
            "targeted queries. Do not call any tools yourself—the host application will execute the searches. "
            "Once you have listed the necessary queries, send a summary message that begins with 'HANDOFF TO WRITER:' "
            "and outline how the collected results should be synthesized. Never write the user's final answer yourself."
        )
        self._planner_agent = AssistantAgent(
            name="research_planner",
            model_client=self._model_client,
            system_message=planner_system_message,
            description="Plans searches by emitting query suggestions only.",
            tools=[self._group_search_tool],
            max_tool_iterations=self._max_tool_iterations,
        )

        writer_system_message = (
            f"You are a research writer collaborating with `research_planner`. The current UTC datetime is {current_dt_iso} and today is {current_day}. "
            "Review the entire conversation history, including the planner's notes and tool outputs. Wait until the planner "
            "sends a message starting with 'HANDOFF TO WRITER:' before you reply. Craft a concise, well-structured report that "
            "(1) directly answers the user, (2) highlights 2-3 key takeaways, (3) includes actionable guidance when applicable, "
            "and (4) cites every factual statement with `[source name]`. Close with the prefix 'FINAL ANSWER:' and nothing after it."
        )
        self._writer_agent = AssistantAgent(
            name="research_writer",
            model_client=self._model_client,
            system_message=writer_system_message,
            description="Writes the final response after the planner hands off.",
            tools=[],
            max_tool_iterations=1,
        )

        self._last_group_transcript: List[Dict[str, Any]] = []
        self._action_history: List[Dict[str, Any]] = []
        self._last_research_questions: List[str] = []

    def run(self, question: str) -> str:
        """Back-compat: delegate to .invoke()."""
        return self.invoke(question)

    def invoke(self, question: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        logger.info("Received query: %s", question)
        self._last_research_questions = self._generate_research_questions(question)
        logger.info("Breakdown agent produced %d research questions.", len(self._last_research_questions))
        self._last_group_transcript = []
        self._action_history = []
        queries = self._generate_search_queries(question)
        logger.info("Search agent produced %d queries; executing searches host-side.", len(queries))
        evidence = self._collect_evidence(queries)
        answer = self._generate_report(question, evidence)
        try:
            logger.info("Query processed successfully.")
            self._persist_response(question=question, response=answer)
            return answer
        except Exception:
            logger.exception("Error while invoking the research agent.")
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_collaborative_group_chat(self, question: str) -> str:
        """Coordinate the planner and writer agents through a group chat run."""

        logger.info("Starting group chat workflow for question: %s", question)
        self._action_history = []
        team = self._create_group_chat()
        result = self._run_async(team.run(task=question))
        transcript: List[Dict[str, Any]] = []
        for message in result.messages:
            dump_method = getattr(message, "dump", None)
            if callable(dump_method):
                try:
                    transcript.append(dump_method())
                except ValidationError:
                    transcript.append({"type": type(message).__name__, "repr": repr(message)})
            else:
                transcript.append({"type": type(message).__name__, "repr": repr(message)})
        self._last_group_transcript = transcript
        final_text = self._extract_text(result.messages, preferred_source=self._writer_agent.name)
        return final_text

    def _create_group_chat(self) -> SelectorGroupChat:
        """Initialise a fresh group chat team for the current run."""

        self._reset_agent(self._planner_agent)
        self._reset_agent(self._writer_agent)
        termination = TextMentionTermination("FINAL ANSWER:")
        team = SelectorGroupChat(
            participants=[self._planner_agent, self._writer_agent],
            model_client=self._model_client,
            selector_func=self._planner_writer_selector,
            termination_condition=termination,
            max_turns=self._group_chat_max_turns,
            allow_repeated_speaker=True,
        )
        return team

    def _planner_writer_selector(self, history: Sequence[Any]) -> str:
        """Choose the next speaker for the planner/writer collaboration."""

        for item in reversed(history):
            if isinstance(item, BaseChatMessage):
                last_message = item
                break
        else:
            return self._planner_agent.name

        source = getattr(last_message, "source", "")
        if source == self._planner_agent.name:
            content = last_message.to_text().strip()
            if content.upper().startswith("HANDOFF TO WRITER:"):
                return self._writer_agent.name
            return self._planner_agent.name
        if source == self._writer_agent.name:
            content = last_message.to_text()
            if "FINAL ANSWER:" in content:
                return self._writer_agent.name
            return self._planner_agent.name
        return self._planner_agent.name

    def _generate_research_questions(self, question: str) -> List[str]:
        self._reset_agent(self._question_agent)
        prompt = (
            "Break the user's request into 4-5 concise research questions that collectively cover the topic."
            f"\n\nUser request: {question}\n"
            "Return only a JSON array of strings."
        )
        result = self._run_async(self._question_agent.run(task=prompt))
        text = self._extract_text(result.messages, preferred_source=self._question_agent.name)
        logger.info("Question breakdown output: %s", text)
        questions = self._parse_queries(text)
        if len(questions) > 5:
            questions = questions[:5]
        if len(questions) < 1 and question.strip():
            questions = [question.strip()]
        logger.info("Using research questions: %s", questions)
        return questions

    def _generate_search_queries(self, question: str) -> List[str]:
        self._reset_agent(self._search_agent)
        prompt_parts = [
            "Generate search queries for the following user question.",
            f"Question: {question}",
        ]
        if self._last_research_questions:
            joined = "\n".join(f"- {item}" for item in self._last_research_questions)
            prompt_parts.append("Focus on addressing these sub-questions:\n" + joined)
        prompt_parts.append("Return only a JSON array of strings.")
        prompt = "\n\n".join(prompt_parts)
        result = self._run_async(self._search_agent.run(task=prompt))
        text = self._extract_text(result.messages, preferred_source=self._search_agent.name)
        logger.info("Query planner output: %s", text)
        queries = self._parse_queries(text)
        if not queries:
            queries = [question.strip()]
        logger.info("Using search queries: %s", queries)
        return queries

    def _collect_evidence(self, queries: List[str]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for query in queries:
            logger.info("Executing Tavily search for: %s", query)
            payload = {"query": query, **self._tavily_options}
            response = self._tavily_tool.call_tool(**payload)
            simplified = []
            entries = []
            if isinstance(response, dict):
                entries = response.get("results") or response.get("data") or []
                if not isinstance(entries, list):
                    entries = [response]
            elif isinstance(response, list):
                entries = response
            else:
                entries = [response]

            for entry in entries:
                if not isinstance(entry, dict):
                    simplified.append({"title": str(entry), "content": str(entry)})
                    continue
                title = entry.get("title") or entry.get("source") or entry.get("url") or "Result"
                content = entry.get("content") or entry.get("snippet") or entry.get("summary") or ""
                simplified.append({"title": title, "content": content})

            evidence.append({"query": query, "results": simplified})
        return evidence

    def _generate_report(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        self._reset_agent(self._report_agent)
        research_notes = self._format_evidence(evidence)
        prompt = (
            f"User question:\n{question}\n\nResearch notes:\n{research_notes}\n\n"
            "Follow the system instructions to produce the final report."
        )
        result = self._run_async(self._report_agent.run(task=prompt))
        text = self._extract_text(result.messages, preferred_source=self._report_agent.name)
        logger.info("Final response: %s", text)
        if text.startswith("FINAL ANSWER:"):
            return text.split("FINAL ANSWER:", 1)[1].strip()
        return text


    def _persist_response(self, *, question: str, response: str) -> None:
        if not self._verbose:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_question = re.sub(r"[^a-zA-Z0-9-_ ]", "", question).strip()[:50]
        filename = f"{timestamp}_{safe_question or 'query'}.txt"
        output_dir = Path("responses")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        content = f"Question:\n{question}\n\nResponse:\n{response}\n"
        file_path.write_text(content, encoding="utf-8")
        logger.info("Saved response to %s", file_path)

    def _run_tavily_group_search(self, query: str) -> str:
        """Wrapper used by the group chat planner to call the Tavily MCP tool."""

        logger.info("Group planner invoking Tavily search for: %s", query)
        payload = {"query": query, **self._tavily_options}
        raw = self._tavily_tool.call_tool(**payload)
        results = raw.get("results", []) if isinstance(raw, dict) else []
        if not isinstance(results, list):
            results = [results]
        simplified: List[Dict[str, Any]] = []
        lines: List[str] = []
        for item in results[: self._tavily_options.get("max_results", 5)]:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or "Result"
                summary = item.get("content") or item.get("snippet") or item.get("description") or ""
                url = item.get("url") or item.get("link") or item.get("source", "")
            else:
                title = str(item)
                summary = ""
                url = ""
            simplified.append({"title": title, "summary": summary, "url": url})
            citation = f" ({url})" if url else ""
            lines.append(f"- {title}: {summary}{citation}".strip())

        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        self._action_history.append(
            {
                "type": "tavily.search",
                "query": query,
                "timestamp": timestamp,
                "results": simplified,
            }
        )

        if not lines:
            lines.append("No results found.")
        summary_text = "Search results for '{query}':\n" + "\n".join(lines)
        return summary_text.format(query=query)

    def _run_async(self, coro: asyncio.Future[str] | asyncio.coroutines.Coroutine[Any, Any, str]) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:  # pragma: no cover - defensive path
            if "event loop is running" in str(exc):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(coro)
            raise

    @staticmethod
    def _last_chat_message(messages: Iterable[Any], preferred_source: Optional[str] = None) -> BaseChatMessage:
        candidate: Optional[BaseChatMessage] = None
        for message in reversed(list(messages)):
            if not isinstance(message, BaseChatMessage):
                continue
            if preferred_source and getattr(message, "source", None) == preferred_source:
                return message
            if candidate is None:
                candidate = message
        if candidate:
            return candidate
        raise RuntimeError("Assistant did not produce a chat response.")

    def _extract_text(self, messages: Iterable[Any], preferred_source: Optional[str]) -> str:
        final_message = self._last_chat_message(messages, preferred_source=preferred_source)
        text_method = getattr(final_message, "to_text", None)
        if callable(text_method):
            try:
                return text_method()
            except Exception:  # pragma: no cover - defensive fallback
                logger.debug("Failed to extract text via to_text().", exc_info=True)
        content = getattr(final_message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(final_message)

    def _parse_queries(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON queries; falling back to plain text.")
        queries: List[str] = []
        for line in text.splitlines():
            line = line.strip().lstrip("-•1234567890. ").strip()
            if line:
                queries.append(line)
        return queries

    def _format_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for item in evidence:
            query = item.get("query")
            entries = item.get("results", [])
            bullet_lines = []
            for entry in entries[: self._tavily_options.get("max_results", 5)]:
                title = entry.get("title") if isinstance(entry, dict) else str(entry)
                summary = entry.get("content") if isinstance(entry, dict) else str(entry)
                bullet_lines.append(f"- {title}: {summary}")
            chunks.append(f"Query: {query}\n" + "\n".join(bullet_lines))
        return "\n\n".join(chunks)

    @property
    def last_group_transcript(self) -> List[Dict[str, Any]]:
        """Return the most recent group chat transcript in serializable form."""

        return list(self._last_group_transcript)

    @property
    def action_history(self) -> List[Dict[str, Any]]:
        """Return the list of tool invocations performed during the last run."""

        return list(self._action_history)

    @property
    def last_research_questions(self) -> List[str]:
        """Return the most recent set of generated research questions."""

        return list(self._last_research_questions)

    def _reset_agent(self, agent: AssistantAgent) -> None:
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()

    @staticmethod
    def _build_openai_client(
        *,
        openai_model_name: str,
        temperature: Optional[float],
    ) -> ChatCompletionClient:
        model_info: ModelInfo = {
            "vision": False,
            "function_calling": True,
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
