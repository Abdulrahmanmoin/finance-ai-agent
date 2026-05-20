"""Pydantic output schemas for every structured-output call in the graph."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Recommendation = Literal["BUY", "HOLD", "SELL"]
Confidence = Literal["LOW", "MEDIUM", "HIGH"]
Sentiment = Literal["Positive", "Negative", "Neutral"]


class MarketDataAnalysis(BaseModel):
    """Output of the market-data ReAct agent."""

    stock: str = Field(..., description="Ticker symbol that was analyzed.")
    price_analysis: str = Field(
        ..., description="Trend direction, support/resistance, momentum read."
    )
    technical_analysis: str = Field(
        ..., description="Interpretation of RSI, Stochastic, MACD, VWAP."
    )
    financial_analysis: str = Field(
        ..., description="Interpretation of P/E, P/B, debt-to-equity, profit margins."
    )
    summary: str = Field(..., description="One-paragraph synthesis with buy/hold/sell lean.")


class NewsAnalysis(BaseModel):
    """Output of the news sentiment agent."""

    article_sentiments: dict[str, Sentiment] = Field(
        default_factory=dict,
        description="Map of article title -> Positive / Negative / Neutral.",
    )
    overall_summary: str = Field(..., description="Two-three sentence summary of news mood.")
    recommendation: str = Field(
        ...,
        description="Investment lean based on news alone (e.g. invest / wait / avoid).",
    )


class FilingsCitation(BaseModel):
    """A single supporting passage from the indexed filing."""

    snippet: str = Field(..., description="Verbatim excerpt from the filing.")
    source_section: str | None = Field(
        default=None, description="Best guess at the section/header of origin."
    )


class FilingsAnswer(BaseModel):
    """Output of the 10-K RAG agent."""

    answer: str = Field(..., description="Answer derived only from the indexed filing.")
    citations: list[FilingsCitation] = Field(
        default_factory=list, description="Verbatim passages backing the answer."
    )


class FinalReport(BaseModel):
    """Synthesized recommendation produced by the report aggregator."""

    ticker: str
    recommendation: Recommendation
    confidence: Confidence
    rationale: str = Field(
        ...,
        description="Why this recommendation, citing market / news / filings signals.",
    )
    key_risks: list[str] = Field(
        default_factory=list, description="Concrete risks the user should track."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Quoted snippets across all sources backing the rationale.",
    )
