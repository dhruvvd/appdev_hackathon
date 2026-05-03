"""Evaluation and visualization utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def calibration_curve(signals_df: pd.DataFrame, actuals: pd.Series) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot calibration of p_above deciles vs realized ITM rates."""
    merged = signals_df.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    aligned_actuals = actuals.copy()
    aligned_actuals.index = pd.to_datetime(aligned_actuals.index)

    merged["actual_positive"] = merged["date"].map(aligned_actuals).astype(float) > 0.0
    merged = merged.dropna(subset=["p_above"])
    merged["bucket"] = pd.qcut(merged["p_above"], q=10, duplicates="drop")

    stats = (
        merged.groupby("bucket", observed=False)
        .agg(predicted=("p_above", "mean"), actual=("actual_positive", "mean"), count=("actual_positive", "size"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(stats["predicted"], stats["actual"], marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Predicted ITM Probability (p_above)")
    ax.set_ylabel("Observed ITM Frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, stats


def plot_pnl(backtest_results: pd.DataFrame) -> plt.Figure:
    """Plot cumulative PnL with drawdown shading."""
    results = backtest_results.copy()
    if results.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Cumulative PnL (No Trades)")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL")
        fig.tight_layout()
        return fig

    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date")

    cum_pnl = results["cum_pnl"] if "cum_pnl" in results else results["pnl"].cumsum()
    running_max = cum_pnl.cummax()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results["date"], cum_pnl, label="Cumulative PnL")
    ax.fill_between(results["date"], running_max, cum_pnl, alpha=0.25, label="Drawdown")
    ax.set_title("Backtest Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def prepare_option_outcomes(
    inference_df: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    expiry_days: int,
) -> pd.DataFrame:
    """Attach realized expiry outcomes to each inference row."""
    rows: list[dict[str, float | str | pd.Timestamp | bool]] = []
    if inference_df.empty:
        return pd.DataFrame()

    source = inference_df.copy()
    source["date"] = pd.to_datetime(source["date"])

    for row in source.itertuples(index=False):
        ticker = str(row.ticker)
        asof_date = pd.Timestamp(row.date)
        if ticker not in prices_by_ticker:
            continue

        price_df = prices_by_ticker[ticker].copy()
        price_df.index = pd.to_datetime(price_df.index)
        if asof_date not in price_df.index:
            continue

        location = price_df.index.get_loc(asof_date)
        if isinstance(location, slice):
            continue
        expiry_loc = int(location) + expiry_days
        if expiry_loc >= len(price_df.index):
            continue

        expiry_date = price_df.index[expiry_loc]
        spot_t = float(row.spot)
        strike = float(row.strike)
        spot_T = float(price_df.loc[expiry_date, "Close"])

        threshold = float(np.log(strike / spot_t))
        realized_log_return = float(np.log(spot_T / spot_t))
        call_itm = realized_log_return > threshold
        put_itm = realized_log_return < threshold

        rows.append(
            {
                "date": asof_date,
                "expiry_date": expiry_date,
                "ticker": ticker,
                "spot_t": spot_t,
                "spot_T": spot_T,
                "strike": strike,
                "mu": float(row.mu),
                "sigma": float(row.sigma),
                "nu": float(row.nu),
                "p_above": float(row.p_above),
                "p_below": float(row.p_below),
                "threshold_return": threshold,
                "realized_log_return": realized_log_return,
                "actual_call_itm": bool(call_itm),
                "actual_put_itm": bool(put_itm),
            }
        )

    return pd.DataFrame(rows).sort_values(["ticker", "date"])


def plot_distribution_snapshot(
    outcomes_df: pd.DataFrame,
    ticker: str,
    asof_date: str | pd.Timestamp,
    option_type: str = "call",
) -> plt.Figure:
    """Plot one predicted Student-t return distribution against realized outcome."""
    target_date = pd.Timestamp(asof_date)
    subset = outcomes_df.loc[(outcomes_df["ticker"] == ticker) & (pd.to_datetime(outcomes_df["date"]) == target_date)]
    if subset.empty:
        raise ValueError(f"No outcome row found for ticker={ticker}, date={target_date.date()}.")

    row = subset.iloc[0]
    mu = float(row["mu"])
    sigma = max(float(row["sigma"]), 1e-6)
    nu = float(row["nu"])
    threshold = float(row["threshold_return"])
    realized = float(row["realized_log_return"])

    x = np.linspace(mu - 6.0 * sigma, mu + 6.0 * sigma, 500)
    z = (x - mu) / sigma
    pdf = student_t.pdf(z, df=nu) / sigma

    is_call = option_type.lower() == "call"
    predicted_prob = float(row["p_above"] if is_call else row["p_below"])
    actual_prob = float(row["actual_call_itm"] if is_call else row["actual_put_itm"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, pdf, label=f"Predicted Student-t PDF ({ticker})")
    ax.axvline(threshold, linestyle="--", label="Strike threshold return")
    ax.axvline(realized, color="black", alpha=0.8, label="Realized expiry return")
    ax.set_xlabel("Log Return to Expiry")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{ticker} {target_date.date()} {option_type.upper()} | "
        f"Predicted ITM={predicted_prob:.3f}, Actual ITM={actual_prob:.0f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_probability_vs_actual(
    outcomes_df: pd.DataFrame,
    ticker: str,
    option_type: str = "call",
    rolling_window: int = 20,
) -> plt.Figure:
    """Plot predicted ITM probabilities against rolling realized ITM frequency."""
    subset = outcomes_df.loc[outcomes_df["ticker"] == ticker].copy()
    if subset.empty:
        raise ValueError(f"No outcomes found for ticker={ticker}.")

    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date")

    is_call = option_type.lower() == "call"
    pred_col = "p_above" if is_call else "p_below"
    actual_col = "actual_call_itm" if is_call else "actual_put_itm"

    subset["actual_itm"] = subset[actual_col].astype(float)
    subset["actual_itm_rolling"] = subset["actual_itm"].rolling(rolling_window, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(subset["date"], subset[pred_col], alpha=0.5, label="Predicted ITM probability")
    ax.plot(subset["date"], subset["actual_itm_rolling"], linewidth=2.0, label=f"Actual ITM rolling mean ({rolling_window})")
    ax.set_title(f"{ticker} {option_type.upper()} Predicted vs Actual ITM")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_realized_vs_predicted_returns(outcomes_df: pd.DataFrame, ticker: str) -> plt.Figure:
    """Scatter realized return against predicted mean return for one ticker."""
    subset = outcomes_df.loc[outcomes_df["ticker"] == ticker].copy()
    if subset.empty:
        raise ValueError(f"No outcomes found for ticker={ticker}.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(subset["mu"], subset["realized_log_return"], alpha=0.5)
    lower = float(min(subset["mu"].min(), subset["realized_log_return"].min()))
    upper = float(max(subset["mu"].max(), subset["realized_log_return"].max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Predicted Mean Return (mu)")
    ax.set_ylabel("Realized Log Return")
    ax.set_title(f"{ticker} Realized vs Predicted Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
