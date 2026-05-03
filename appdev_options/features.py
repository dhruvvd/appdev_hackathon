"""Feature engineering and normalization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _ensure_1d_series(values: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Coerce a Series-like object to a single 1D float Series."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must be 1D, got shape {values.shape}.")
        series = values.iloc[:, 0]
    else:
        series = values
    return series.astype(float).rename(name)


def engineer_features(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer model features and return aligned target log-returns."""
    close = _ensure_1d_series(ohlcv["Close"], "Close")
    volume = _ensure_1d_series(ohlcv["Volume"], "Volume")

    log_ret = np.log(close).diff().rename("log_ret")
    rv_20 = log_ret.rolling(window=20, min_periods=20).std().rename("rv_20")
    rv_5 = log_ret.rolling(window=5, min_periods=5).std().rename("rv_5")

    vol_mean_20 = volume.rolling(window=20, min_periods=20).mean()
    vol_std_20 = volume.rolling(window=20, min_periods=20).std()
    vol_z = ((volume - vol_mean_20) / (vol_std_20 + config.EPSILON)).rename("vol_z")

    features = pd.concat(
        [
            log_ret,
            rv_20,
            rv_5,
            vol_z,
            log_ret.shift(1).rename("lag_1"),
            log_ret.shift(2).rename("lag_2"),
            log_ret.shift(3).rename("lag_3"),
        ],
        axis=1,
    )

    combined = features.join(log_ret.rename("target"), how="inner").dropna(how="any")
    feature_df = combined.drop(columns=["target"])
    target = combined["target"]
    return feature_df, target


def fit_normalizer(features: pd.DataFrame, train_end: str) -> tuple[pd.Series, pd.Series]:
    """Fit normalization stats using train window only."""
    cutoff = pd.to_datetime(train_end)
    train_slice = features.loc[features.index <= cutoff]
    means = train_slice.mean()
    stds = train_slice.std().replace(0.0, 1.0)
    return means, stds


def apply_normalizer(features: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """Apply z-score normalization with precomputed statistics."""
    normalized = (features - means) / (stds + config.EPSILON)
    return normalized
