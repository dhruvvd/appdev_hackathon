#!/usr/bin/env python3
"""
End-to-end Neural SDE (Itô, diagonal noise) for daily closes via torchsde.

Dependencies (Python 3.10+):
    pip install torch torchsde yfinance matplotlib scikit-learn numpy

Uses torchsde.sdeint only — no custom Euler–Maruyama loops.

Example:
    python neural_sde_forecast.py --ticker QQQ
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchsde
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 1. Data pipeline
# -----------------------------------------------------------------------------
def load_and_prepare_ticker(symbol: str):
    """Download ~1 year of daily closes for ``symbol``, scale to [0.1, 0.9], build time grid."""
    symbol = symbol.strip().upper()
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5y", interval="1d")
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(
            f"yfinance returned no usable data for {symbol!r} (check connection/symbol)."
        )

    closes = df["Close"].astype(np.float64).values.reshape(-1, 1)
    # Drop any trailing NaNs from history()
    mask = np.isfinite(closes.squeeze(-1))
    closes = closes[mask]

    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    scaled = scaler.fit_transform(closes).astype(np.float64).squeeze(-1)

    n = scaled.shape[0]
    # Time axis in [0, 1], one point per trading day (matches data length).
    t_np = np.linspace(0.0, 1.0, n, dtype=np.float64)

    train_len = int(np.floor(0.8 * n))
    if train_len < 5 or (n - train_len) < 2:
        raise RuntimeError(f"Train/test split invalid: n={n}, train_len={train_len}")

    return scaled, t_np, train_len, scaler


def ticker_slug(symbol: str) -> str:
    """Filesystem-safe fragment derived from ticker (e.g. BRK-B -> BRK-B kept alphanumeric-ish)."""
    s = symbol.strip().upper()
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s or "TICKER"


class NeuralSDE(nn.Module):
    """
    Neural Itô SDE with diagonal noise:
        dX_t = f_theta(t, X_t) dt + g_phi(t, X_t) dW_t

    torchsde expects:
      - noise_type == "diagonal"
      - sde_type == "ito"
      - f(t, y), g(t, y) with y shape (batch_size, state_size)
      - outputs same batch/state layout as y (diagonal g is a vector per batch row)
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, state_size: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_size = state_size
        self.hidden_dim = hidden_dim

        # Inputs: [state, time] -> state_size
        in_dim = state_size + 1
        self.drift_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_size),
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_size),
            nn.Softplus(),  # strictly positive diffusion (variance contribution > 0)
        )

    def _time_feat(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Broadcast scalar time t to shape (batch, 1) concatenated as extra input dim."""
        batch = y.shape[0]
        # torchsde passes t as a scalar tensor (0-dim); avoid breaking autograd on y.
        t_float = float(t.reshape(-1)[0].detach())
        return torch.full((batch, 1), t_float, device=y.device, dtype=y.dtype)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift f_theta(t, y): (batch, state_size).

        y: (batch_size, state_size)
        """
        # Concatenate state and time along feature dimension -> (batch, state_size + 1)
        t_feat = self._time_feat(t, y)
        x_in = torch.cat([y, t_feat], dim=-1)
        return self.drift_net(x_in)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion g_phi(t, y): (batch_size, state_size) for diagonal noise.

        Positive entries via Softplus.
        """
        t_feat = self._time_feat(t, y)
        x_in = torch.cat([y, t_feat], dim=-1)
        return self.diffusion_net(x_in)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural SDE forecast with uncertainty (yfinance daily closes).")
    p.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Yahoo Finance symbol (default: SPY)",
    )
    return p.parse_args()


def run_forecast(
    ticker: str,
    *,
    num_epochs: int = 300,
    num_paths: int = 100,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    print_every: int = 50,
    verbose: bool = True,
    save_plots: bool = False,
    show_plots: bool = False,
) -> dict[str, Any]:
    """
    Train the neural SDE on yfinance daily closes and sample paths for uncertainty bands.

    Returns JSON-serializable lists (for APIs); set ``save_plots`` / ``show_plots`` for CLI-style matplotlib output.
    """
    ticker_display = ticker.strip().upper()
    scaled_np, t_np, train_len, scaler = load_and_prepare_ticker(ticker_display)
    n = scaled_np.shape[0]

    scaled = torch.tensor(scaled_np, dtype=torch.float32, device=device)
    ts_full = torch.tensor(t_np, dtype=torch.float32, device=device)

    ts_train = ts_full[:train_len].contiguous()
    y_train_target = scaled[:train_len].contiguous()
    y_val_target = scaled[train_len:].contiguous()
    dt_train = ts_train[1] - ts_train[0]
    dt_full = ts_full[1] - ts_full[0]

    y0_train = scaled[0].view(1, 1)

    model = NeuralSDE(state_size=1, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        ys_train = torchsde.sdeint(
            model,
            y0_train,
            ts_train,
            dt=dt_train,
            method="euler",
            adaptive=False,
        )
        pred_train = ys_train.squeeze(-1).squeeze(-1)
        train_loss = torch.mean((pred_train - y_train_target) ** 2)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            ys_full = torchsde.sdeint(
                model,
                y0_train,
                ts_full,
                dt=dt_full,
                method="euler",
                adaptive=False,
            )
            pred_val = ys_full[train_len:].squeeze(-1).squeeze(-1)
            val_loss = torch.mean((pred_val - y_val_target) ** 2)

        train_losses.append(float(train_loss.detach().cpu()))
        val_losses.append(float(val_loss.detach().cpu()))

        if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == num_epochs):
            print(
                f"epoch {epoch:4d}/{num_epochs}  train_mse={train_losses[-1]:.6f}  val_mse={val_losses[-1]:.6f}"
            )

    model.eval()
    y0_batch = scaled[0].view(1, 1).expand(num_paths, 1).contiguous()

    with torch.no_grad():
        ys_paths = torchsde.sdeint(
            model,
            y0_batch,
            ts_full,
            dt=dt_full,
            method="euler",
            adaptive=False,
        )
    paths = ys_paths.squeeze(-1).cpu().numpy()

    mean_path = paths.mean(axis=1)
    std_path = paths.std(axis=1, ddof=0)
    z = 1.96
    lower = mean_path - z * std_path
    upper = mean_path + z * std_path

    def inv_transform(arr_1d: np.ndarray) -> np.ndarray:
        return scaler.inverse_transform(arr_1d.reshape(-1, 1)).squeeze(-1)

    actual_all = inv_transform(scaled_np)
    mean_dollars = inv_transform(mean_path)
    lower_dollars = inv_transform(lower)
    upper_dollars = inv_transform(upper)

    if save_plots or show_plots:
        plt.figure(figsize=(11, 5))
        plt.plot(np.arange(train_len), actual_all[:train_len], color="black", linewidth=1.5, label="Actual (train)")
        plt.plot(np.arange(train_len, n), actual_all[train_len:], color="red", linewidth=1.5, label="Actual (test / held-out)")
        plt.plot(
            np.arange(n),
            mean_dollars,
            color="blue",
            linewidth=1.8,
            label=f"Mean predicted path ({num_paths} SDE samples)",
        )
        plt.fill_between(
            np.arange(n),
            lower_dollars,
            upper_dollars,
            color="blue",
            alpha=0.22,
            label="Approx. 95% band (mean ± 1.96·std across paths)",
        )
        plt.axvline(train_len - 0.5, color="gray", linestyle="--", linewidth=1.0, label="Train/test split")
        plt.title(f"Neural SDE forecast with uncertainty — {ticker_display} daily close (last ~1y)")
        plt.xlabel("Trading day index (chronological)")
        plt.ylabel("Price (USD)")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out_path = f"neural_sde_{ticker_slug(ticker_display)}_forecast.png"
        if save_plots:
            plt.savefig(out_path, dpi=150)
            print(f"Saved figure to {out_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()

        epochs_axis = np.arange(1, num_epochs + 1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs_axis, train_losses, color="tab:blue", label="Train MSE")
        plt.plot(epochs_axis, val_losses, color="tab:orange", label="Val MSE (held-out tail)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (scaled space)")
        plt.title(f"Training curves — {ticker_display}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = f"neural_sde_{ticker_slug(ticker_display)}_loss.png"
        if save_plots:
            plt.savefig(loss_path, dpi=150)
            print(f"Saved figure to {loss_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()

    day_index = np.arange(n, dtype=int).tolist()
    return {
        "ticker": ticker_display,
        "n": int(n),
        "train_len": int(train_len),
        "num_epochs": int(num_epochs),
        "num_paths": int(num_paths),
        "day_index": day_index,
        "actual_usd": actual_all.astype(float).tolist(),
        "mean_usd": mean_dollars.astype(float).tolist(),
        "lower_usd": lower_dollars.astype(float).tolist(),
        "upper_usd": upper_dollars.astype(float).tolist(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_mse": train_losses[-1] if train_losses else None,
        "final_val_mse": val_losses[-1] if val_losses else None,
    }


def main() -> None:
    args = parse_args()
    run_forecast(
        args.ticker,
        save_plots=True,
        show_plots=True,
    )


if __name__ == "__main__":
    main()
