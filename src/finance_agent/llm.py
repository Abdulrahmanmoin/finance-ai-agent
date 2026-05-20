"""Single chokepoint for Google Gemini chat + embedding clients."""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from finance_agent.config import get_settings


@lru_cache(maxsize=4)
def get_chat_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return a cached Gemini chat model.

    Model name is sourced from ``GEMINI_MODEL`` — never hardcoded.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=temperature,
        google_api_key=settings.google_api_key,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return cached Gemini embeddings.

    Embedding model name is sourced from ``GEMINI_EMBEDDING_MODEL``.
    """
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )
