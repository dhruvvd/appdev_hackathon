"""Simple synthetic options backtest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from appdev_options import config


def _black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    rate: float,
    iv: float,
    option_type: str,
) -> float:
    """Price a European option with Black-Scholes."""
    if time_to_expiry_years <= 0.0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    iv = max(iv, 1e-6)
    sqrt_t = np.sqrt(time_to_expiry_years)
    d1 = (np.log(spot / strike) + (rate + 0.5 * iv**2) * time_to_expiry_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    if option_type == "call":
        return float(spot * norm.cdf(d1) - strike * np.exp(-rate * time_to_expiry_years) * norm.cdf(d2))
    return float(strike * np.exp(-rate * time_to_expiry_years) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def _max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Compute max drawdown from cumulative PnL."""
    if cumulative_pnl.empty:
        return 0.0
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    return float(drawdown.min())


def run_backtest(
    signals_df: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    expiry_days: int = config.SYNTHETIC_EXPIRY_DAYS,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run synthetic ATM options backtest using signal direction."""
    trade_rows: list[dict[str, Any]] = []
    tradable = signals_df.loc[signals_df["signal"] != "NO_TRADE"].copy()
    tradable["date"] = pd.to_datetime(tradable["date"])
    tradable = tradable.sort_values("date")

    for row in tradable.itertuples(index=False):
        ticker = str(row.ticker)
        signal = str(row.signal)
        trade_date = pd.Timestamp(row.date)

        if ticker not in prices_by_ticker:
            continue
        price_df = prices_by_ticker[ticker]
        if trade_date not in price_df.index:
            continue

        close = price_df["Close"].astype(float)
        log_ret = np.log(close).diff()
        rv_20 = log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252.0)
        if trade_date not in rv_20.index or np.isnan(rv_20.loc[trade_date]):
            continue

        position = price_df.index.get_loc(trade_date)
        if isinstance(position, slice):
            continue
        expiry_position = int(position) + expiry_days
        if expiry_position >= len(price_df.index):
            continue

        expiry_date = price_df.index[expiry_position]
        spot_t = float(close.loc[trade_date])
        spot_T = float(close.loc[expiry_date])
        strike = spot_t
        iv = float(rv_20.loc[trade_date])
        t_years = expiry_days / 252.0

        if signal == "BUY_CALL":
            premium = _black_scholes_price(spot_t, strike, t_years, risk_free_rate, iv, "call")
            intrinsic = max(spot_T - strike, 0.0)
            direction = "call"
        else:
            premium = _black_scholes_price(spot_t, strike, t_years, risk_free_rate, iv, "put")
            intrinsic = max(strike - spot_T, 0.0)
            direction = "put"

        pnl = intrinsic - premium
        trade_rows.append(
            {
                "date": trade_date,
                "expiry_date": expiry_date,
                "ticker": ticker,
                "signal": signal,
                "option_type": direction,
                "spot_t": spot_t,
                "spot_T": spot_T,
                "strike": strike,
                "iv": iv,
                "premium": premium,
                "intrinsic": intrinsic,
                "pnl": pnl,
            }
        )

    results = pd.DataFrame(trade_rows).sort_values("date") if trade_rows else pd.DataFrame(columns=["date", "pnl"])
    if not results.empty:
        results["cum_pnl"] = results["pnl"].cumsum()
        win_rate = float((results["pnl"] > 0.0).mean())
        total_pnl = float(results["pnl"].sum())
        avg_pnl = float(results["pnl"].mean())
        max_dd = _max_drawdown(results["cum_pnl"])
    else:
        win_rate = 0.0
        total_pnl = 0.0
        avg_pnl = 0.0
        max_dd = 0.0

    summary = {
        "n_trades": int(len(results)),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_dd,
    }
    return results, summary
