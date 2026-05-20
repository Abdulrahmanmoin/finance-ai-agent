"""News tools: fetch yfinance news and extract full article text via MarkItDown."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import requests
import yfinance as yf
from markitdown import MarkItDown
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from finance_agent.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NewsArticle:
    """One news article with optional extracted body."""

    title: str
    summary: str
    url: str
    pubdate: str
    full_text: str = ""
    extraction_error: str | None = None

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "pubdate": self.pubdate,
            "full_text": self.full_text or self.summary,
        }


@dataclass(slots=True)
class NewsBundle:
    """Result of `fetch_news_for_ticker`."""

    ticker: str
    articles: list[NewsArticle] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_LINK_PATTERN = re.compile(r"http\S+")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[.*?\]")
_SPECIAL_PATTERN = re.compile(r"[#*()+\-\n]")
_SLASH_PATTERN = re.compile(r"/\S*")
_DOUBLE_SPACE_PATTERN = re.compile(r"  +")


def _clean(text: str) -> str:
    """Strip links and noisy markdown artefacts from extracted article text."""
    text = _LINK_PATTERN.sub("", text)
    text = _MARKDOWN_LINK_PATTERN.sub("", text)
    text = _SPECIAL_PATTERN.sub(" ", text)
    text = _SLASH_PATTERN.sub("", text)
    text = _DOUBLE_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def _build_session(timeout: int) -> requests.Session:
    """Configure a Session with retry-on-5xx and standard headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "finance-agent/0.1 (+https://github.com/iam-armoin/finance-agent)",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # MarkItDown calls session.get without exposing a timeout kwarg; setting a
    # default on the session ensures we never block indefinitely.
    session.request = _with_default_timeout(session.request, timeout)  # type: ignore[method-assign]
    return session


def _with_default_timeout(func: Any, timeout: int) -> Any:
    def wrapper(method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timeout)
        return func(method, url, **kwargs)

    return wrapper


def _fetch_raw_news(ticker: str) -> list[dict[str, Any]]:
    """Pull the raw news payload from yfinance, filtered to story-type items."""
    raw = yf.Ticker(ticker).news or []
    return [item for item in raw if item.get("content", {}).get("contentType") == "STORY"]


def _coerce_article(raw_item: dict[str, Any]) -> NewsArticle | None:
    content = raw_item.get("content") or {}
    url = (content.get("canonicalUrl") or {}).get("url")
    title = content.get("title")
    if not (url and title):
        return None
    pubdate = (content.get("pubDate") or "").split("T", 1)[0]
    return NewsArticle(
        title=title,
        summary=content.get("summary") or "",
        url=url,
        pubdate=pubdate,
    )


def extract_article(url: str, *, md: MarkItDown) -> str:
    """Fetch ``url`` and return cleaned plain-text body. May raise on network errors."""
    result = md.convert(url)
    title = (result.title or "").strip()
    body = _clean((result.text_content or "").strip())
    return f"{title}\n{body}".strip()


def fetch_news_for_ticker(ticker: str, *, limit: int | None = None) -> NewsBundle:
    """Fetch the most recent yfinance news for ``ticker`` and extract full bodies.

    Failures on individual articles are recorded on ``NewsBundle.errors`` rather
    than raised; the bundle returns whatever could be assembled.
    """
    settings = get_settings()
    cap = limit if limit is not None else settings.news_max_articles
    bundle = NewsBundle(ticker=ticker)

    try:
        raw_items = _fetch_raw_news(ticker)
    except Exception as exc:
        logger.warning("yfinance news fetch failed for %s: %s", ticker, exc)
        bundle.errors.append(f"yfinance.news failed: {exc}")
        return bundle

    if not raw_items:
        bundle.errors.append(f"no news returned for {ticker}")
        return bundle

    session = _build_session(settings.news_request_timeout)
    md = MarkItDown(requests_session=session)

    for raw in raw_items[:cap]:
        article = _coerce_article(raw)
        if article is None:
            continue
        try:
            article.full_text = extract_article(article.url, md=md)
        except Exception as exc:
            article.extraction_error = str(exc)
            bundle.errors.append(f"extract failed for {article.url}: {exc}")
            logger.info("article extraction failed for %s: %s", article.url, exc)
        bundle.articles.append(article)

    return bundle
