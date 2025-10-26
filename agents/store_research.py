"""Data model for storing research runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class StoreResearch:
    """Lightweight container for persisting research results."""

    id: str
    research_question: str
    research_report: str
    notes: Optional[str] = None
