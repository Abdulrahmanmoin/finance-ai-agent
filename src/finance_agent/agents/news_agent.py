"""News-sentiment specialist node."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from finance_agent.llm import get_chat_llm
from finance_agent.prompts import NEWS_SYSTEM_PROMPT
from finance_agent.schemas import NewsAnalysis
from finance_agent.state import AgentState
from finance_agent.tools.news import NewsBundle, fetch_news_for_ticker

logger = logging.getLogger(__name__)


_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NEWS_SYSTEM_PROMPT),
        (
            "human",
            "Stock: {ticker}\nUser question: {question}\n\nNews articles (JSON):\n{articles}",
        ),
    ]
)


def _bundle_to_prompt_payload(bundle: NewsBundle) -> str:
    return json.dumps(
        [article.to_prompt_dict() for article in bundle.articles],
        indent=2,
        ensure_ascii=False,
    )


def news_node(state: AgentState) -> dict[str, Any]:
    """Fetch news + extract bodies + run structured-output LLM sentiment pass."""
    bundle = fetch_news_for_ticker(state.ticker)

    if not bundle.articles:
        analysis = NewsAnalysis(
            article_sentiments={},
            overall_summary="No news articles could be retrieved for this ticker.",
            recommendation="N/A — news pipeline returned no usable articles.",
        )
        return {
            "news_analysis": analysis,
            "errors": bundle.errors,
        }

    chain = _PROMPT | get_chat_llm(temperature=0.0).with_structured_output(NewsAnalysis)
    try:
        analysis = chain.invoke(
            {
                "ticker": state.ticker,
                "question": state.question,
                "articles": _bundle_to_prompt_payload(bundle),
            }
        )
    except Exception as exc:
        logger.exception("news_node LLM call failed")
        return {
            "news_analysis": NewsAnalysis(
                article_sentiments={},
                overall_summary="News sentiment analysis failed.",
                recommendation="N/A — LLM error during news analysis.",
            ),
            "errors": [*bundle.errors, f"news_node: {exc}"],
        }

    return {"news_analysis": analysis, "errors": bundle.errors}
