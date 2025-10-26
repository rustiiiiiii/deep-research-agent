"""
Utility helpers for interacting with Model Context Protocol (MCP) servers over JSON-RPC.

Some MCP deployments (e.g. Tavily) expose their functionality via HTTP JSON-RPC
instead of the canonical WebSocket transport.  This module implements a tiny
JSON-RPC client that can list tools and invoke them on demand.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import requests

logger = logging.getLogger(__name__)


class MCPRPCError(RuntimeError):
    """Raised when the MCP server responds with a JSON-RPC error."""

    def __init__(self, *, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"MCP RPC error {code}: {message}")
        self.code = code
        self.data = data


@dataclass(slots=True)
class MCPServerConfig:
    """Connection details for an MCP server exposed via JSON-RPC over HTTP."""

    base_url: str
    api_key: Optional[str] = None
    tool_name: Optional[str] = None  # Optional default tool to call.


class MCPToolClient:
    """
    Minimal JSON-RPC client for an MCP server.

    The client calls `list_tools` once at startup to cache the advertised
    tools, and later issues `call_tool` requests whenever a tool invocation is
    needed.
    """

    def __init__(self, *, config: MCPServerConfig, request_timeout: int = 60) -> None:
        self._config = config
        self._session = requests.Session()
        default_headers = {
            "Accept": "application/json, application/*+json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "TripResearchAgent/0.1",
        }
        if config.api_key:
            default_headers["Authorization"] = f"Bearer {config.api_key}"
        self._session.headers.update(default_headers)

        self._base_url = config.base_url.rstrip("/")
        self._timeout = request_timeout
        self._tools: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialising MCPToolClient for %s", self._base_url)
        self._load_catalogue()

    def _load_catalogue(self) -> None:
        logger.info("Fetching MCP tool catalogue via JSON-RPC list_tools")
        result = self._json_rpc("list_tools", params={})
        tools = result.get("tools", []) if isinstance(result, dict) else result or []
        self._tools = {tool["name"]: tool for tool in tools}
        if self._tools:
            logger.info("Discovered MCP tools: %s", ", ".join(sorted(self._tools)))
        else:
            logger.warning("No tools discovered from MCP server at %s", self._base_url)

    def call_tool(self, *, tool_name: Optional[str] = None, **arguments: Any) -> Dict[str, Any]:
        target = tool_name or self._config.tool_name
        if not target:
            raise ValueError("Tool name must be provided when no default is configured.")
        if target not in self._tools:
            logger.warning("Tool '%s' not present in cached catalogue; invoking anyway.", target)

        params = {"name": target, "arguments": arguments}
        logger.info("Invoking MCP tool '%s' via JSON-RPC call_tool", target)
        logger.debug("Invocation params: %s", params)
        result = self._json_rpc("call_tool", params=params)
        logger.debug("Invocation result: %s", result)
        return result if isinstance(result, dict) else {"result": result}

    def _json_rpc(self, method: str, params: Optional[Dict[str, Any]]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
        logger.debug("JSON-RPC request payload: %s", payload)

        try:
            response = self._session.post(self._base_url, json=payload, timeout=self._timeout)
        except requests.exceptions.RequestException as exc:
            logger.exception("JSON-RPC request failed: %s", exc)
            raise

        logger.info("JSON-RPC response status: %s", response.status_code)
        logger.info("JSON-RPC raw response text: %s", response.text)
        if response.status_code >= 400:
            logger.error("JSON-RPC HTTP error: %s", response.text)
            response.raise_for_status()
        data = response.json()
        logger.debug("JSON-RPC response payload: %s", data)

        if "error" in data and data["error"] is not None:
            error = data["error"]
            raise MCPRPCError(
                code=error.get("code", -32000),
                message=error.get("message", "Unknown error"),
                data=error.get("data"),
            )

        return data.get("result")

    def described_tools(self) -> Iterable[Dict[str, Any]]:
        """Returns the cached tool catalogue."""
        return self._tools.values()
