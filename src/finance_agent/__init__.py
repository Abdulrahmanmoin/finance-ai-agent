"""Finance Agent: multi-agent equity research powered by LangGraph + Google Gemini."""

from __future__ import annotations

__version__ = "0.2.0"

from finance_agent.graph import build_graph
from finance_agent.state import AgentState

__all__ = ["AgentState", "__version__", "build_graph"]
