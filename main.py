"""
Command line interface for the Trip Research Agent.

Loads API keys from environment variables (via `.env`), creates a ResearchAgent,
and enters an interactive loop to answer travel-related queries.
"""

import os
import sys
import logging
from dotenv import load_dotenv


from agents.research_agent import ResearchAgent

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the command line loop for the research agent."""
    logger.info("Loading environment variables from .env file...")
    load_dotenv()

    try:
        agent = ResearchAgent()
    except Exception as exc:
        logger.exception("Failed to initialize the research agent: %s", exc)
        return

    print(
        "\nWelcome to the Trip Research Agent!\n"
        "Type your travel question and press Enter.  Type 'quit' to exit.\n"
    )

    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            logger.info("EOF received; exiting.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            logger.info("User requested exit.")
            break

        try:
            logger.info("Processing query: %s", query)
            response = agent.invoke(query)
            logger.info(f"\n{response}\n")
            logger.info("Response delivered successfully.")
        except Exception as exc:
            logger.exception("Error while processing query: %s", exc)
            print(f"An error occurred: {exc}\n")

    logger.info("Session ended. Goodbye!")
    print("Goodbye!")


if __name__ == "__main__":
    main()
