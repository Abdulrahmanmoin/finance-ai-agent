"""End-to-end graph smoke test using stubbed specialist nodes."""

from __future__ import annotations

from typing import Any

import pytest

from finance_agent import graph as graph_module
from finance_agent.schemas import (
    FilingsAnswer,
    FinalReport,
    MarketDataAnalysis,
    NewsAnalysis,
)
from finance_agent.state import AgentState


def _fake_market_data(_state: AgentState) -> dict[str, Any]:
    return {
        "market_data": MarketDataAnalysis(
            stock="NVDA",
            price_analysis="Uptrend.",
            technical_analysis="RSI 65, MACD positive.",
            financial_analysis="Forward P/E 38, margins strong.",
            summary="Constructive.",
        )
    }


def _fake_news(_state: AgentState) -> dict[str, Any]:
    return {
        "news_analysis": NewsAnalysis(
            article_sentiments={"AI boom continues": "Positive"},
            overall_summary="Generally upbeat.",
            recommendation="Consider investing.",
        )
    }


def _fake_filings(_state: AgentState) -> dict[str, Any]:
    return {
        "filings_answer": FilingsAnswer(
            answer="Strong segment growth disclosed.",
            citations=[],
        )
    }


def _fake_report(_state: AgentState) -> dict[str, Any]:
    return {
        "final_report": FinalReport(
            ticker="NVDA",
            recommendation="BUY",
            confidence="MEDIUM",
            rationale="Strong technicals, supportive news, healthy filing.",
            key_risks=["Concentration in datacenter", "Geopolitical exposure"],
            citations=["RSI 65", "AI boom"],
        )
    }


@pytest.fixture
def stubbed_graph(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(graph_module, "market_data_node", _fake_market_data)
    monkeypatch.setattr(graph_module, "news_node", _fake_news)
    monkeypatch.setattr(graph_module, "filings_node", _fake_filings)
    monkeypatch.setattr(graph_module, "report_node", _fake_report)
    return graph_module.build_graph()


def test_graph_runs_end_to_end(stubbed_graph: Any) -> None:
    result = stubbed_graph.invoke(
        {"ticker": "NVDA", "question": "Is NVDA a buy?"},
        config={"configurable": {"thread_id": "smoke"}},
    )
    final = result["final_report"]
    assert final.recommendation == "BUY"
    assert final.ticker == "NVDA"


def test_graph_topology_has_expected_nodes_and_edges(stubbed_graph: Any) -> None:
    g = stubbed_graph.get_graph()
    node_ids = set(g.nodes.keys())
    for expected in {"dispatch", "market_data", "news", "filings_qa", "report"}:
        assert expected in node_ids

    edges = {(e.source, e.target) for e in g.edges}
    assert ("dispatch", "market_data") in edges
    assert ("dispatch", "news") in edges
    assert ("dispatch", "filings_qa") in edges
    assert ("market_data", "report") in edges
    assert ("news", "report") in edges
    assert ("filings_qa", "report") in edges
