"""Unit tests for `finance_agent.tools.market_data`."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest

from finance_agent.tools import market_data


def test_get_stock_prices_returns_expected_shape(
    monkeypatch: pytest.MonkeyPatch, ohlcv_frame: pd.DataFrame
) -> None:
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **kw: ohlcv_frame)

    result: dict[str, Any] = market_data.get_stock_prices.invoke({"ticker": "TEST"})

    assert set(result) == {"stock_price", "indicators"}
    assert len(result["stock_price"]) == 12  # _INDICATOR_TAIL

    indicators = result["indicators"]
    assert set(indicators) == {"RSI", "Stochastic_Oscillator", "MACD", "MACD_Signal", "VWAP"}
    for series in indicators.values():
        assert series, "indicator series should not be empty"
        for date_str, value in series.items():
            assert isinstance(date_str, str)
            assert len(date_str) == 10  # YYYY-MM-DD
            assert isinstance(value, float)


def test_get_stock_prices_handles_multiindex_columns(
    monkeypatch: pytest.MonkeyPatch, ohlcv_frame: pd.DataFrame
) -> None:
    multi = ohlcv_frame.copy()
    multi.columns = pd.MultiIndex.from_tuples([(c, "TEST") for c in multi.columns])

    monkeypatch.setattr(market_data.yf, "download", lambda *a, **kw: multi)

    result = market_data.get_stock_prices.invoke({"ticker": "TEST"})
    assert result["indicators"]["RSI"]


def test_get_stock_prices_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **kw: pd.DataFrame())

    with pytest.raises(ValueError, match="no price data"):
        market_data.get_stock_prices.invoke({"ticker": "NOPE"})


def test_get_financial_metrics_maps_yfinance_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTicker:
        info: ClassVar[dict[str, object]] = {
            "forwardPE": 22.5,
            "priceToBook": 8.0,
            "debtToEquity": 120.0,
            "profitMargins": 0.42,
            "irrelevant": "ignored",
        }

    monkeypatch.setattr(market_data.yf, "Ticker", lambda _t: FakeTicker())

    result = market_data.get_financial_metrics.invoke({"ticker": "TEST"})

    assert result == {
        "pe_ratio": 22.5,
        "price_to_book": 8.0,
        "debt_to_equity": 120.0,
        "profit_margins": 0.42,
    }


def test_get_financial_metrics_tolerates_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTicker:
        info: ClassVar[dict[str, object]] = {"forwardPE": 18.0}

    monkeypatch.setattr(market_data.yf, "Ticker", lambda _t: FakeTicker())

    result = market_data.get_financial_metrics.invoke({"ticker": "TEST"})
    assert result["pe_ratio"] == 18.0
    assert result["price_to_book"] is None
    assert result["debt_to_equity"] is None
    assert result["profit_margins"] is None


def test_get_financial_metrics_raises_on_empty_info(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTicker:
        info: ClassVar[dict[str, object]] = {}

    monkeypatch.setattr(market_data.yf, "Ticker", lambda _t: FakeTicker())

    with pytest.raises(ValueError, match="no info"):
        market_data.get_financial_metrics.invoke({"ticker": "NOPE"})
