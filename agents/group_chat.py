from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from autogen_agentchat.teams import BaseGroupChat  # noqa: F401 - placeholder for future integration


@dataclass(slots=True)
class GroupChatMessage:
    timestamp: str
    agent: str
    message: str


class ActionGroupChat:
    """Lightweight group chat log so agents can describe their actions."""

    def __init__(self, participants: List[str]) -> None:
        self.participants = participants[:]
        self.messages: List[GroupChatMessage] = []

    def post(self, agent: str, message: str) -> None:
        if agent not in self.participants:
            self.participants.append(agent)
        entry = GroupChatMessage(
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            agent=agent,
            message=message,
        )
        self.messages.append(entry)

    def transcript(self) -> str:
        return "\n".join(f"[{m.timestamp}] {m.agent}: {m.message}" for m in self.messages)
