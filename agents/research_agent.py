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
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.models import ChatCompletionClient, ModelInfo
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from research_state_manager import ResearchQuestionState
from pydantic import ValidationError

from .mcp_client import MCPServerConfig, MCPToolClient
from .research_prompts import (
    EVALUATOR_PROMPT,
    PLANNER_PROMPT,
    QUESTION_BREAKDOWN_PROMPT,
    SEARCH_QUERY_PROMPT,
    report_prompt,
    writer_prompt,
)
from research_state_manager import ResearchQuestionState

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
        self._termination_phrase = "Report complete."

        tavily_options = {
            "max_results": tavily_max_results,
            "include_raw_content": tavily_include_raw_content,
        }
        if tavily_kwargs:
            tavily_options.update(tavily_kwargs)
        self._tavily_options = tavily_options
        self._tavily_tool_name = tavily_mcp_tool_name
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

        self._question_system_message = QUESTION_BREAKDOWN_PROMPT
        self._question_agent = AssistantAgent(
            name="question_breakdown",
            model_client=self._model_client,
            system_message=self._question_system_message,
            description="Decomposes the primary request into focused research questions.",
            tools=[],
            max_tool_iterations=1,
        )

        self._search_system_message = SEARCH_QUERY_PROMPT
        self._search_agent = AssistantAgent(
            name="search_query_generator",
            model_client=self._model_client,
            system_message=self._search_system_message,
            description="Generates targeted search queries with no tool usage.",
            tools=[],
            max_tool_iterations=1,
        )

        self._report_system_message = report_prompt(current_dt_iso=current_dt_iso, current_day=current_day)
        logger.info("Initializing report agent with OpenAI model '%s'", openai_model_name)
        self._report_agent = AssistantAgent(
            name="research_reporter",
            model_client=self._model_client,
            system_message=self._report_system_message,
            description="Synthesizes gathered evidence into a final report.",
            tools=[],
            max_tool_iterations=1,
        )

        self._planner_system_message = PLANNER_PROMPT
        self._planner_agent = AssistantAgent(
            name="research_planner",
            model_client=self._model_client,
            system_message=self._planner_system_message,
            description="Plans searches by emitting query suggestions only.",
            tools=[],
            max_tool_iterations=1,
        )

        self._writer_system_message = writer_prompt(current_dt_iso=current_dt_iso, current_day=current_day)
        self._writer_agent = AssistantAgent(
            name="research_writer",
            model_client=self._model_client,
            system_message=self._writer_system_message,
            description="Writes the final response after the planner hands off.",
            tools=[],
            max_tool_iterations=1,
        )

        self._evaluator_system_message = EVALUATOR_PROMPT
        self._evaluator_agent = AssistantAgent(
            name="research_evaluator",
            model_client=self._model_client,
            system_message=self._evaluator_system_message,
            description="Evaluates retrieved evidence before synthesis.",
            tools=[],
            max_tool_iterations=1,
        )

        self._last_group_transcript: List[Dict[str, Any]] = []
        self._action_history: List[Dict[str, Any]] = []
        self._last_research_questions: List[str] = []
        self._research_states: List[ResearchQuestionState] = []
        self._latest_evidence: List[Dict[str, Any]] = []
        self._pending_queries: Deque[str] = deque()
        self._current_query: Optional[str] = None
        self._log_dir = Path("logs")
        self._current_log: Optional[Dict[str, Any]] = None
        self._current_log_path: Optional[Path] = None

    def run(self, question: str) -> str:
        """Back-compat: delegate to .invoke()."""
        return self.invoke(question)


    def process_single_research_question(self, research_question: ResearchQuestionState) -> None:
        """Placeholder for single research question processing hook."""
        _ = research_question

    def invoke(self, question: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        original_question = question
        self._start_interaction_log(original_question)
        logger.info("Received query: %s", original_question)

        # generate the research questions for the user's question
        self._last_research_questions = self._generate_research_questions(original_question)

        logger.info("Breakdown agent produced %d research questions.", len(self._last_research_questions))

        # create a state object for every research question
        self._research_states = []
        for idx, research_question in enumerate(self._last_research_questions):
            state = ResearchQuestionState(research_question=research_question)
            self._research_states.append(state)
            self._log_agent_context(
                "research_question_state",
                {
                    "index": idx,
                    "state": self._make_serializable(asdict(state)),
                },
            )

            # process this research task 
            self.process_single_research_question(state)
            

        self._last_group_transcript = []
        self._action_history = []
        self._latest_evidence = []
        queries = self._generate_search_queries(original_question)
        logger.info("Search agent produced %d queries; executing searches host-side.", len(queries))
        self._pending_queries = deque(queries)
        self._log_agent_context(
            "query_queue",
            {
                "pending": list(self._pending_queries),
            },
        )
        if not self._pending_queries:
            raise RuntimeError("No research queries were generated for this question.")
        current_query = self._pending_queries.popleft()
        self._current_query = current_query
        self._log_agent_context(
            "query_dispatch",
            {
                "selected_query": current_query,
                "remaining": list(self._pending_queries),
            },
        )
        evidence = self._collect_single_query_evidence(current_query)
        answer: Optional[str] = None
        try:
            answer = self._generate_report(original_question, evidence)
            logger.info("Query processed successfully.")
            self._persist_response(question=original_question, response=answer)
            self._finalize_interaction_log(final_response=answer)
            return answer
        except Exception as exc:
            logger.exception("Error while invoking the research agent.")
            self._finalize_interaction_log(final_response=answer, error=str(exc))
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
        self._log_agent_context(
            "group_chat",
            {
                "task": question,
                "participants": [self._planner_agent.name, self._writer_agent.name],
                "planner_system_message": self._planner_system_message,
                "writer_system_message": self._writer_system_message,
                "termination_phrase": self._termination_phrase,
                "transcript": transcript,
            },
        )
        return final_text

    def _create_group_chat(self) -> SelectorGroupChat:
        """Initialise a fresh group chat team for the current run."""

        self._reset_agent(self._planner_agent)
        self._reset_agent(self._writer_agent)
        termination = TextMentionTermination(self._termination_phrase)
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
            if content.strip().lower().endswith(self._termination_phrase.lower()):
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
        self._log_agent_context(
            self._question_agent.name,
            {
                "system_message": self._question_system_message,
                "prompt": prompt,
                "raw_output": text,
                "parsed_questions": questions,
            },
        )
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
        self._log_agent_context(
            self._search_agent.name,
            {
                "system_message": self._search_system_message,
                "prompt": prompt,
                "raw_output": text,
                "parsed_queries": queries,
            },
        )
        return queries

    def _collect_single_query_evidence(self, query: str) -> List[Dict[str, Any]]:
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

        evidence = [{"query": query, "results": simplified}]
        self._log_agent_context(
            self._tavily_tool_name,
            {
                "query": query,
                "payload": self._make_serializable(payload),
                "raw_response": self._make_serializable(response),
                "simplified_results": simplified,
            },
        )
        self._latest_evidence = evidence
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
        cleaned = text.strip()
        if cleaned.lower().endswith(self._termination_phrase.lower()):
            cleaned = cleaned[: -(len(self._termination_phrase))].rstrip(" .")
        self._log_agent_context(
            self._report_agent.name,
            {
                "system_message": self._report_system_message,
                "prompt": prompt,
                "raw_output": text,
                "final_output": cleaned,
            },
        )
        return cleaned


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
        self._log_agent_context(
            f"{self._tavily_tool_name} (group)",
            {
                "query": query,
                "payload": self._make_serializable(payload),
                "raw_response": self._make_serializable(raw),
                "summarised_results": simplified,
            },
        )
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
        to_text = getattr(final_message, "to_text", None)
        if callable(to_text):
            return to_text().strip()
        return str(final_message).strip()

    def _parse_queries(self, text: str) -> List[str]:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON array; falling back to plaintext parsing.")

        queries: List[str] = []
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-•1234567890. ").strip()
            if cleaned:
                queries.append(cleaned)
        if not queries and text.strip():
            queries.append(text.strip())
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

    @property
    def research_states(self) -> List[ResearchQuestionState]:
        """Return dataclass snapshots for the generated research questions."""

        return list(self._research_states)

    @property
    def pending_queries(self) -> List[str]:
        """Return the queue of assistant-generated queries awaiting processing."""

        return list(self._pending_queries)

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

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _start_interaction_log(self, question: str) -> None:
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem issues
            logger.warning("Unable to create log directory %s: %s", self._log_dir, exc)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_name = f"research_agent_{timestamp}_{uuid4().hex[:8]}.json"
        self._current_log_path = self._log_dir / file_name
        self._current_log = {
            "timestamp": timestamp,
            "question": question,
            "steps": [],
        }

    def _log_agent_context(self, agent_name: str, context: Dict[str, Any]) -> None:
        if not self._current_log:
            return
        entry = {
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "context": self._make_serializable(context),
        }
        self._current_log.setdefault("steps", []).append(entry)

    def _finalize_interaction_log(self, *, final_response: Optional[str], error: Optional[str] = None) -> None:
        if not self._current_log or not self._current_log_path:
            return
        if final_response is not None:
            self._current_log["final_response"] = final_response
        if error:
            self._current_log["error"] = error
        self._current_log["research_questions"] = self._last_research_questions
        self._current_log["research_states"] = self._make_serializable(
            [asdict(state) for state in self._research_states]
        )
        self._current_log["processed_query"] = self._current_query
        self._current_log["remaining_queries"] = list(self._pending_queries)
        self._current_log["action_history"] = self._make_serializable(self._action_history)
        self._current_log["evidence_summary"] = self._make_serializable(self._latest_evidence)
        try:
            serialized = json.dumps(self._current_log, indent=2, ensure_ascii=False)
            self._current_log_path.write_text(serialized, encoding="utf-8")
            logger.info("Wrote interaction log to %s", self._current_log_path)
        except OSError as exc:  # pragma: no cover - filesystem issues
            logger.warning("Failed to write interaction log %s: %s", self._current_log_path, exc)
        finally:
            self._current_log = None
            self._current_log_path = None
            self._current_query = None

    def _make_serializable(self, data: Any) -> Any:
        try:
            json.dumps(data)
            return data
        except TypeError:
            if isinstance(data, dict):
                return {str(key): self._make_serializable(value) for key, value in data.items()}
            if isinstance(data, list):
                return [self._make_serializable(item) for item in data]
            if isinstance(data, (set, tuple)):
                return [self._make_serializable(item) for item in data]
            return repr(data)
