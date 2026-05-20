"""10-K filings RAG specialist node."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from finance_agent.llm import get_chat_llm
from finance_agent.prompts import FILINGS_SYSTEM_PROMPT
from finance_agent.schemas import FilingsAnswer
from finance_agent.state import AgentState
from finance_agent.tools.filings import FilingsRetriever

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", FILINGS_SYSTEM_PROMPT),
        (
            "human",
            "Question about the {ticker} 10-K: {question}\n\nRetrieved excerpts:\n{excerpts}",
        ),
    ]
)


def _format_excerpts(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "filing")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def filings_node(
    state: AgentState,
    *,
    retriever: FilingsRetriever | None = None,
) -> dict[str, Any]:
    """Retrieve top-k 10-K chunks and synthesize a structured ``FilingsAnswer``."""
    try:
        active = retriever or FilingsRetriever()
        docs = active.query(state.question)
    except Exception as exc:
        logger.exception("filings retrieval failed")
        return {
            "filings_answer": FilingsAnswer(
                answer="The 10-K index could not be queried.",
                citations=[],
            ),
            "errors": [f"filings_node retrieval: {exc}"],
        }

    if not docs:
        return {
            "filings_answer": FilingsAnswer(
                answer="No relevant passages were found in the indexed filing.",
                citations=[],
            )
        }

    chain = _PROMPT | get_chat_llm(temperature=0.0).with_structured_output(FilingsAnswer)
    try:
        answer = chain.invoke(
            {
                "ticker": state.ticker,
                "question": state.question,
                "excerpts": _format_excerpts(docs),
            }
        )
    except Exception as exc:
        logger.exception("filings_node LLM call failed")
        return {
            "filings_answer": FilingsAnswer(
                answer="LLM error during filings synthesis.",
                citations=[],
            ),
            "errors": [f"filings_node llm: {exc}"],
        }

    return {"filings_answer": answer}
