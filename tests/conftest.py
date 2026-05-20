"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Provide deterministic env vars so ``Settings()`` constructs without a real ``.env``."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / ".faiss_index"))
    monkeypatch.setenv("FILINGS_TEXT_PATH", str(tmp_path / "filings.txt"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    # Reset the lru_cache in config so each test gets fresh settings.
    from finance_agent.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    """Synthetic ~6mo daily OHLCV frame used by market-data tests."""
    idx = pd.date_range("2024-01-02", periods=180, freq="B")
    base = pd.Series(range(180), index=idx, dtype="float64") + 100.0
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.5,
            "Low": base - 1.5,
            "Close": base + 0.5,
            "Adj Close": base + 0.5,
            "Volume": pd.Series([1_000_000 + i * 1000 for i in range(180)], index=idx),
        }
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive guard: fail loudly if any test reaches the real internet."""

    def _raise(*_a: object, **_kw: object) -> None:
        raise RuntimeError("network access blocked in tests")

    # Block requests.get / .post at the module level. Tests mock specific clients.
    import requests

    monkeypatch.setattr(requests, "get", _raise)
    monkeypatch.setattr(requests, "post", _raise)
    monkeypatch.setattr(requests, "request", _raise)

    # Block any accidental httpx use (langchain-google-genai uses it under the hood).
    os.environ.setdefault("NO_NETWORK", "1")
