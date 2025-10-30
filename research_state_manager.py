"""
State models supporting the research workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4


@dataclass(slots=True)
class ResearchQuestionState:
    """Represents a single assistant-generated research question and its progress."""

    research_question: str
    search_queries: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    id: str = field(default_factory=lambda: uuid4().hex)
