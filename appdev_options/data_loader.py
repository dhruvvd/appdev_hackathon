"""Data loading utilities for OHLCV time series."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a single ticker from yfinance."""
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.loc[:, OHLCV_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.ffill(limit=2).dropna(how="any")
    return df


def load_tickers(tickers: list[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch and clean OHLCV data for a list of tickers."""
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_df = fetch_ohlcv(ticker=ticker, start=start, end=end)
        if not ticker_df.empty:
            data[ticker] = ticker_df
    return data
