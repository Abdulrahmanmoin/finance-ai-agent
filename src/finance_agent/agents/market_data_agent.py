"""Market-data specialist: ReAct loop over yfinance tools with structured output."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from finance_agent.llm import get_chat_llm
from finance_agent.prompts import MARKET_DATA_SYSTEM_PROMPT
from finance_agent.schemas import MarketDataAnalysis
from finance_agent.state import AgentState
from finance_agent.tools.market_data import get_financial_metrics, get_stock_prices

logger = logging.getLogger(__name__)


def _build_agent() -> Any:
    return create_agent(
        model=get_chat_llm(temperature=0.0),
        tools=[get_stock_prices, get_financial_metrics],
        system_prompt=MARKET_DATA_SYSTEM_PROMPT,
        response_format=MarketDataAnalysis,
    )


def market_data_node(state: AgentState) -> dict[str, Any]:
    """Run the ReAct loop and write a ``MarketDataAnalysis`` into the state."""
    agent = _build_agent()
    user_msg = (
        f"Ticker: {state.ticker}\n"
        f"User question: {state.question}\n\n"
        "Call the tools, then produce the MarketDataAnalysis."
    )
    try:
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=MARKET_DATA_SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
            }
        )
    except Exception as exc:
        logger.exception("market_data_agent failed")
        return {"errors": [f"market_data_agent: {exc}"]}

    structured: MarketDataAnalysis | None = result.get("structured_response")
    if structured is None:
        return {"errors": ["market_data_agent: missing structured_response"]}
    return {"market_data": structured}
