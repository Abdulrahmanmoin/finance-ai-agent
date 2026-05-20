"""Unit tests for `finance_agent.tools.news`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from finance_agent.tools import news


def _make_news_item(title: str, url: str, summary: str = "") -> dict[str, object]:
    return {
        "content": {
            "contentType": "STORY",
            "title": title,
            "summary": summary,
            "canonicalUrl": {"url": url},
            "pubDate": "2026-05-20T10:00:00Z",
        }
    }


def test_clean_strips_links_and_special_characters() -> None:
    raw = "Hello https://example.com world\n[link](x) /path  twice"
    out = news._clean(raw)
    assert "https" not in out
    assert "[link]" not in out
    assert "/path" not in out
    assert "  " not in out


def test_fetch_news_for_ticker_filters_to_story_and_extracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = [
        _make_news_item("Big move", "https://example.com/a", "summary a"),
        # Should be filtered out (not a STORY)
        {"content": {"contentType": "VIDEO", "title": "vid"}},
        _make_news_item("Mixed signals", "https://example.com/b", "summary b"),
    ]

    class FakeTicker:
        news = raw

    monkeypatch.setattr(news.yf, "Ticker", lambda _t: FakeTicker())

    def fake_convert(self: object, url: str) -> SimpleNamespace:
        return SimpleNamespace(title=f"Title for {url}", text_content="Body text here.")

    monkeypatch.setattr(news.MarkItDown, "convert", fake_convert)

    bundle = news.fetch_news_for_ticker("TEST", limit=5)

    assert bundle.ticker == "TEST"
    assert [a.title for a in bundle.articles] == ["Big move", "Mixed signals"]
    assert all("Title for" in a.full_text for a in bundle.articles)
    assert bundle.errors == []


def test_fetch_news_records_per_article_extraction_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = [
        _make_news_item("Article A", "https://example.com/a"),
        _make_news_item("Article B", "https://example.com/b"),
    ]

    class FakeTicker:
        news = raw

    monkeypatch.setattr(news.yf, "Ticker", lambda _t: FakeTicker())

    def fake_convert(self: object, url: str) -> SimpleNamespace:
        if url.endswith("/b"):
            raise RuntimeError("boom")
        return SimpleNamespace(title="ok", text_content="body")

    monkeypatch.setattr(news.MarkItDown, "convert", fake_convert)

    bundle = news.fetch_news_for_ticker("TEST")
    titles = [a.title for a in bundle.articles]
    assert titles == ["Article A", "Article B"]
    assert bundle.articles[0].full_text  # extracted
    assert bundle.articles[1].full_text == ""  # failed
    assert bundle.articles[1].extraction_error == "boom"
    assert any("boom" in err for err in bundle.errors)


def test_fetch_news_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTicker:
        news: ClassVar[list[dict[str, object]]] = []

    monkeypatch.setattr(news.yf, "Ticker", lambda _t: FakeTicker())

    bundle = news.fetch_news_for_ticker("NOPE")
    assert bundle.articles == []
    assert bundle.errors and "no news" in bundle.errors[0]


def test_fetch_news_handles_yfinance_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_t: str) -> object:
        raise RuntimeError("yahoo died")

    monkeypatch.setattr(news.yf, "Ticker", explode)

    bundle = news.fetch_news_for_ticker("NOPE")
    assert bundle.articles == []
    assert any("yfinance.news failed" in e for e in bundle.errors)
