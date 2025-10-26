# Trip Research Agent

This project contains a minimal Microsoft AutoGen‑based agent designed to perform
deep research to help users plan their trips.  The agent can search,
retrieve, and synthesize information from the web, generating concise
answers with citations.  It is organized as a small Python package that
you can extend as you build more sophisticated planning capabilities.

## Features

- Uses [Microsoft AutoGen](https://microsoft.github.io/autogen/) to orchestrate language models and tools.
- Connects to a Tavily MCP server over HTTP JSON-RPC to fetch up‑to‑date travel information.
- Illustrates how to build a custom AutoGen assistant that can call multiple tools (search plus in‑house domain stubs).
- Designed for interactive use via the command line or integration into a chat interface.

## Getting Started

### Prerequisites

- Python 3.8 or higher.
- API keys for your chosen language model (e.g. OpenAI GPT-5 Nano via `OPENAI_API_KEY`) and credentials for the Tavily MCP server (set `TAVILY_MCP_BASE_URL` to the JSON-RPC endpoint, e.g. `https://mcp.tavily.com/mcp`, and optionally `TAVILY_MCP_API_KEY`).
- Optional: VS Code with the Python extension for a smoother development experience.

### Installation

1. Clone or extract this repository and change into its root directory.

   ```bash
   cd trip_research_agent
   ```

2. Install the dependencies into a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy the `.env.example` to `.env` and update the placeholders with your own API keys and configuration values:

   ```bash
   cp .env.example .env
   # Edit .env in your favourite editor and fill in the API keys.
   ```

### Running the Agent

The package exposes a simple CLI entry point.  Once your environment
variables are set and the Tavily MCP JSON-RPC endpoint is reachable you can run:

```bash
python main.py
```

You’ll be prompted for a travel‑related question.  The agent will
perform a web search through the Tavily MCP tool and return a concise answer.

#### Local MCP server

If you do not have access to a hosted MCP endpoint you can run the bundled
Tavily MCP shim locally:

```bash
TAVILY_API_KEY=your_key python -m mcp_servers.local_tavily_server
```

By default it listens on `http://127.0.0.1:6112/mcp`.  Point
`TAVILY_MCP_BASE_URL` at that URL (this is already the default in `.env`) and
the agent will route all `web_search` tool calls through the local server.

### Project Structure

| Path                        | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `main.py`                  | Entrypoint for running the agent from the command line.      |
| `requirements.txt`         | Python dependencies.                                         |
| `.env.example`             | Template for environment variables.                          |
| `agents/`                  | Package containing agent classes and helper functions.       |
| `agents/research_agent.py` | Implements the core research agent using AutoGen.            |

### Extending

This project is intended as a starting point.  Some ideas for extension include:

- Adding specialized tools (e.g. flight search APIs, hotel availability, weather forecasts).
- Integrating memory so that the agent can remember previous interactions.
- Building a web or chat interface on top of the underlying agent.
- Implementing caching to avoid repeated identical searches.

Contributions are welcome—feel free to fork and create a pull request.
