"""Unit tests for `finance_agent.tools.filings.FilingsRetriever`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from finance_agent.config import get_settings
from finance_agent.tools.filings import FilingsRetriever


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hash embedder for tests (no network)."""

    DIM = 64

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in text.lower().split():
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:2], "little")
            vec[h % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def sample_filing(tmp_path: Path) -> Path:
    text = (
        "Section 1. Business overview.\n\n"
        "Acme Corp manufactures industrial widgets across three plants in Ohio.\n\n"
        "Section 2. Risk factors.\n\n"
        "Key risks include supply chain disruption, currency volatility, and cybersecurity threats.\n\n"
        "Section 3. Financial highlights.\n\n"
        "Revenue grew 14% year over year, driven by enterprise demand.\n"
    )
    path = tmp_path / "fake_10k.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_build_then_query_round_trip(
    monkeypatch: pytest.MonkeyPatch, sample_filing: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("FILINGS_TEXT_PATH", str(sample_filing))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "idx"))
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    get_settings.cache_clear()

    retriever = FilingsRetriever(embeddings=HashEmbeddings())
    retriever.build()

    docs = retriever.query("What risks does the company face?", k=2)
    assert docs, "retriever should return chunks"
    combined = " ".join(d.page_content.lower() for d in docs)
    assert "risk" in combined or "cybersecurity" in combined


def test_load_existing_index_skips_rebuild(
    monkeypatch: pytest.MonkeyPatch, sample_filing: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("FILINGS_TEXT_PATH", str(sample_filing))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "idx"))
    get_settings.cache_clear()

    FilingsRetriever(embeddings=HashEmbeddings()).build()

    # A second retriever pointed at the same dir should load, not rebuild.
    second = FilingsRetriever(embeddings=HashEmbeddings())
    docs = second.query("widgets", k=1)
    assert docs


def test_build_raises_when_source_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FILINGS_TEXT_PATH", str(tmp_path / "missing.txt"))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "idx"))
    get_settings.cache_clear()

    retriever = FilingsRetriever(embeddings=HashEmbeddings())
    with pytest.raises(FileNotFoundError):
        retriever.build()
