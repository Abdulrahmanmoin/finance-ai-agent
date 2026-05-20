"""LangGraph state container shared across all nodes."""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from finance_agent.schemas import (
    FilingsAnswer,
    FinalReport,
    MarketDataAnalysis,
    NewsAnalysis,
)


class AgentState(BaseModel):
    """Shared state passed through the LangGraph workflow.

    Each parallel specialist writes to a distinct field, so concurrent updates
    never collide. ``messages`` and ``errors`` use commutative reducers so
    parallel branches can safely contribute.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: str
    question: str

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    market_data: MarketDataAnalysis | None = None
    news_analysis: NewsAnalysis | None = None
    filings_answer: FilingsAnswer | None = None

    final_report: FinalReport | None = None

    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
