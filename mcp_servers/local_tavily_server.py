"""
Local JSON-RPC MCP server that proxies requests to the Tavily Python client.

This server is intended for development usage when a remote MCP endpoint is not
available.  It exposes two JSON-RPC 2.0 methods over HTTP:

  - `list_tools` returns the Tavily search tool metadata.
  - `call_tool` executes `tavily.search` with the provided arguments.

Run it with:

    python -m mcp_servers.local_tavily_server

Make sure `TAVILY_API_KEY` is set in your environment.  Configure the research
agent to point at `http://127.0.0.1:6112/mcp` (or whichever host/port you use).
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple, Type

from dotenv import load_dotenv
from tavily import TavilyClient

logger = logging.getLogger("local_mcp_server")

TOOL_DEFINITION: Dict[str, Any] = {
    "name": "tavily.search",
    "description": "General-purpose Tavily web search.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query", "minLength": 1},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
            "include_raw_content": {
                "type": "boolean",
                "default": False,
                "description": "Include the raw page content in the response.",
            },
        },
        "required": ["query"],
    },
}


def _json_rpc_error(rpc_id: Any, code: int, message: str, data: Any | None = None) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {
            "code": code,
            "message": message,
            "data": data,
        },
    }


def _json_rpc_result(rpc_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def make_handler(tavily_client: TavilyClient) -> Type[BaseHTTPRequestHandler]:
    class LocalMCPHandler(BaseHTTPRequestHandler):
        server_version = "LocalMCP/0.1"
        rpc_path = "/mcp"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            logger.info("%s - - %s", self.address_string(), format % args)

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != self.rpc_path:
                self.send_error(404, "Not Found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            try:
                request = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                logger.error("Invalid JSON payload: %s", exc)
                self._send_json(_json_rpc_error(None, -32700, "Invalid JSON"))
                return

            response = self._handle_rpc(request)
            self._send_json(response)

        def _handle_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
            rpc_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}

            if method == "list_tools":
                logger.debug("Handling list_tools request")
                return _json_rpc_result(rpc_id, {"tools": [TOOL_DEFINITION]})

            if method == "call_tool":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if name != TOOL_DEFINITION["name"]:
                    return _json_rpc_error(rpc_id, -32601, f"Unknown tool '{name}'")

                query = arguments.get("query")
                if not query or not isinstance(query, str):
                    return _json_rpc_error(rpc_id, -32602, "Argument 'query' is required.")

                try:
                    result = tavily_client.search(**arguments)
                    return _json_rpc_result(rpc_id, result)
                except Exception as exc:  # pragma: no cover - passthrough
                    logger.exception("Error while executing Tavily search.")
                    return _json_rpc_error(rpc_id, -32001, str(exc))

            return _json_rpc_error(rpc_id, -32601, f"Unknown method '{method}'")

    return LocalMCPHandler


def run_server(host: str, port: int, tavily_client: TavilyClient) -> None:
    handler_cls = make_handler(tavily_client)
    server = ThreadingHTTPServer((host, port), handler_cls)
    logger.info("Local MCP server listening on http://%s:%d/mcp", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        logger.info("Shutting down local MCP server.")
    finally:
        server.server_close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY must be set to run the local MCP server.")

    host = os.getenv("LOCAL_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("LOCAL_MCP_PORT", "6112"))
    tavily_client = TavilyClient(api_key=api_key)

    run_server(host, port, tavily_client)


if __name__ == "__main__":
    main()
