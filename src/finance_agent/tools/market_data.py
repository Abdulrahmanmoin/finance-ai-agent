"""Market-data tools: yfinance prices + technical indicators + fundamental ratios."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volume import volume_weighted_average_price

logger = logging.getLogger(__name__)

# 18 months of daily history — long enough for stable long-window indicators
# while still keeping the payload small.
_HISTORY_WEEKS = 24 * 3
# How many trailing days of each indicator to surface to the LLM.
_INDICATOR_TAIL = 12


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns for some tickers; flatten to first level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]
    return df


def _series_to_dated_dict(series: pd.Series) -> dict[str, float]:
    return {
        idx.strftime("%Y-%m-%d"): round(float(value), 4)
        for idx, value in series.dropna().items()
    }


@tool
def get_stock_prices(ticker: str) -> dict[str, Any]:
    """Fetch ~18 months of daily OHLCV plus rolling technical indicators.

    Returns a dict with two keys:
      - ``stock_price``: list of ``{Date, Open, High, Low, Close, Volume}`` records.
      - ``indicators``: trailing ``_INDICATOR_TAIL`` values for RSI, Stochastic,
        MACD, MACD signal, and VWAP, keyed by ISO date.
    """
    end = dt.datetime.now()
    start = end - dt.timedelta(weeks=_HISTORY_WEEKS)
    logger.info("Fetching prices for %s from %s to %s", ticker, start.date(), end.date())

    raw = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no price data for {ticker!r}")

    df = _flatten_columns(raw)

    indicators: dict[str, dict[str, float]] = {
        "RSI": _series_to_dated_dict(
            RSIIndicator(df["Close"], window=14).rsi().iloc[-_INDICATOR_TAIL:]
        ),
        "Stochastic_Oscillator": _series_to_dated_dict(
            StochasticOscillator(df["High"], df["Low"], df["Close"], window=14)
            .stoch()
            .iloc[-_INDICATOR_TAIL:]
        ),
    }

    macd = MACD(df["Close"])
    indicators["MACD"] = _series_to_dated_dict(macd.macd().iloc[-_INDICATOR_TAIL:])
    indicators["MACD_Signal"] = _series_to_dated_dict(
        macd.macd_signal().iloc[-_INDICATOR_TAIL:]
    )
    indicators["VWAP"] = _series_to_dated_dict(
        volume_weighted_average_price(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
        ).iloc[-_INDICATOR_TAIL:]
    )

    records = df.reset_index().tail(_INDICATOR_TAIL).copy()
    date_col = "Date" if "Date" in records.columns else records.columns[0]
    records[date_col] = records[date_col].astype(str)

    return {
        "stock_price": records.to_dict(orient="records"),
        "indicators": indicators,
    }


@tool
def get_financial_metrics(ticker: str) -> dict[str, float | None]:
    """Fetch headline fundamental ratios via yfinance's ``Ticker.info``.

    Returns ``pe_ratio`` (forward P/E), ``price_to_book``, ``debt_to_equity``,
    and ``profit_margins``. Any individual missing value is returned as ``None``
    rather than raising — Yahoo doesn't expose every field for every ticker.
    """
    logger.info("Fetching financial metrics for %s", ticker)
    info = yf.Ticker(ticker).info
    if not info:
        raise ValueError(f"yfinance returned no info for {ticker!r}")

    return {
        "pe_ratio": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "debt_to_equity": info.get("debtToEquity"),
        "profit_margins": info.get("profitMargins"),
    }
