"""Inference utilities for distribution forecasts and ITM probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from appdev_options import config
from appdev_options.model import DistributionModel


def load_model_checkpoint(
    ticker: str,
    checkpoint_dir: Path | None = None,
    device: torch.device | None = None,
) -> tuple[DistributionModel, dict[str, Any]]:
    """Load a trained model checkpoint and metadata for one ticker."""
    checkpoint_root = checkpoint_dir or config.CHECKPOINT_DIR
    checkpoint_path = checkpoint_root / f"{ticker}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=target_device)
    model_cfg = payload["config"]

    model = DistributionModel(
        num_features=int(payload["num_features"]),
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(target_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def predict_distribution(
    model: DistributionModel,
    x_window: torch.Tensor,
) -> tuple[float, float, float]:
    """Predict Student-t parameters for one input window."""
    if x_window.ndim == 2:
        x_window = x_window.unsqueeze(0)

    device = next(model.parameters()).device
    x_window = x_window.to(device=device, dtype=torch.float32)
    mu, sigma, nu = model(x_window)
    return float(mu.item()), float(sigma.item()), float(nu.item())


def compute_itm_probabilities(
    mu: float,
    sigma: float,
    nu: float,
    strike: float,
    spot: float,
) -> tuple[float, float]:
    """Compute P(S_T > K) and P(S_T < K) from predicted return distribution."""
    threshold = float(np.log(strike / spot))
    sigma_safe = max(float(sigma), config.EPSILON)
    standardized = (threshold - float(mu)) / sigma_safe

    # Torch StudentT.cdf is not implemented in some runtimes.
    p_below = float(student_t.cdf(standardized, df=float(nu)))
    p_above = 1.0 - p_below
    return p_above, p_below


@torch.no_grad()
def run_inference(
    model: DistributionModel,
    ticker: str,
    features: pd.DataFrame,
    close_prices: pd.Series,
    lookback: int,
    test_start: str,
) -> pd.DataFrame:
    """Run model inference over windows with target dates in test period."""
    rows: list[dict[str, Any]] = []
    test_start_dt = pd.to_datetime(test_start)

    for idx in range(lookback, len(features)):
        date = features.index[idx]
        if date < test_start_dt:
            continue

        x_window = torch.tensor(features.iloc[idx - lookback : idx].to_numpy(), dtype=torch.float32)
        mu, sigma, nu = predict_distribution(model=model, x_window=x_window)

        spot = float(close_prices.loc[date])
        strike = spot
        p_above, p_below = compute_itm_probabilities(mu=mu, sigma=sigma, nu=nu, strike=strike, spot=spot)

        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "spot": spot,
                "strike": strike,
                "mu": mu,
                "sigma": sigma,
                "nu": nu,
                "p_above": p_above,
                "p_below": p_below,
            }
        )

    return pd.DataFrame(rows)
