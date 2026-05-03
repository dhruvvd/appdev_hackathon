"""Directional options signal filter based on ITM probabilities."""

from __future__ import annotations

import pandas as pd


def generate_signals(inference_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Generate directional option signals from model probabilities."""
    signals = inference_df.copy()

    signal_col = pd.Series("NO_TRADE", index=signals.index, dtype="object")
    signal_col.loc[signals["p_above"] > threshold] = "BUY_CALL"
    signal_col.loc[signals["p_below"] > threshold] = "BUY_PUT"
    signals["signal"] = signal_col

    output = signals.loc[
        :,
        ["date", "ticker", "signal", "p_above", "p_below", "mu", "sigma", "nu"],
    ].copy()
    return output
