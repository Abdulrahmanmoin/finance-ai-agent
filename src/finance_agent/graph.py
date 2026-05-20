"""LangGraph wiring: parallel fan-out of three specialists into a final aggregator."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from finance_agent.agents import (
    filings_node,
    market_data_node,
    news_node,
    report_node,
)
from finance_agent.state import AgentState


def _dispatch(_state: AgentState) -> dict[str, Any]:
    """No-op fan-out hub: lets the three specialists branch from a common parent."""
    return {}


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compile the multi-agent finance-research graph.

    Topology::

        START → dispatch ─┬→ market_data ─┐
                          ├→ news ────────┼→ report → END
                          └→ filings_qa ──┘

    Sibling nodes execute concurrently; ``report`` waits for all three to land.
    """
    g: StateGraph = StateGraph(AgentState)

    g.add_node("dispatch", _dispatch)
    g.add_node("market_data", market_data_node)
    g.add_node("news", news_node)
    g.add_node("filings_qa", filings_node)
    g.add_node("report", report_node)

    g.add_edge(START, "dispatch")
    g.add_edge("dispatch", "market_data")
    g.add_edge("dispatch", "news")
    g.add_edge("dispatch", "filings_qa")
    g.add_edge("market_data", "report")
    g.add_edge("news", "report")
    g.add_edge("filings_qa", "report")
    g.add_edge("report", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
