"""Tests for `AgentState` schema + reducers."""

from __future__ import annotations

from finance_agent.schemas import FinalReport
from finance_agent.state import AgentState


def test_agent_state_defaults() -> None:
    s = AgentState(ticker="NVDA", question="Buy?")
    assert s.market_data is None
    assert s.news_analysis is None
    assert s.filings_answer is None
    assert s.final_report is None
    assert s.errors == []
    assert s.messages == []


def test_agent_state_accepts_final_report() -> None:
    s = AgentState(
        ticker="NVDA",
        question="Buy?",
        final_report=FinalReport(
            ticker="NVDA",
            recommendation="BUY",
            confidence="HIGH",
            rationale="Because.",
            key_risks=[],
            citations=[],
        ),
    )
    assert s.final_report is not None
    assert s.final_report.recommendation == "BUY"
