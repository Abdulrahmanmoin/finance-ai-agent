"""Final report aggregator node."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from finance_agent.llm import get_chat_llm
from finance_agent.prompts import REPORT_SYSTEM_PROMPT
from finance_agent.schemas import (
    FilingsAnswer,
    FinalReport,
    MarketDataAnalysis,
    NewsAnalysis,
)
from finance_agent.state import AgentState

logger = logging.getLogger(__name__)


_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REPORT_SYSTEM_PROMPT),
        (
            "human",
            "Ticker: {ticker}\nUser question: {question}\n\n"
            "Market data analysis:\n{market_data}\n\n"
            "News sentiment analysis:\n{news_analysis}\n\n"
            "Filings (10-K) answer:\n{filings_answer}\n\n"
            "Known degradations (may be empty):\n{errors}",
        ),
    ]
)


def _safe_dump(value: MarketDataAnalysis | NewsAnalysis | FilingsAnswer | None, label: str) -> str:
    if value is None:
        return f"<{label} unavailable>"
    return value.model_dump_json(indent=2)


def report_node(state: AgentState) -> dict[str, Any]:
    """Synthesize all three specialist outputs into a ``FinalReport``."""
    chain = _PROMPT | get_chat_llm(temperature=0.0).with_structured_output(FinalReport)
    try:
        report = chain.invoke(
            {
                "ticker": state.ticker,
                "question": state.question,
                "market_data": _safe_dump(state.market_data, "market_data"),
                "news_analysis": _safe_dump(state.news_analysis, "news_analysis"),
                "filings_answer": _safe_dump(state.filings_answer, "filings_answer"),
                "errors": "\n".join(state.errors) or "<none>",
            }
        )
    except Exception as exc:
        logger.exception("report aggregator failed")
        return {
            "final_report": FinalReport(
                ticker=state.ticker,
                recommendation="HOLD",
                confidence="LOW",
                rationale=f"Report synthesis failed: {exc}",
                key_risks=[],
                citations=[],
            ),
            "errors": [f"report_node: {exc}"],
        }
    return {"final_report": report}
