"""Training loop for the distribution model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import TimeSeriesDataset
from loss import student_t_nll_loss
from model import DistributionModel


def _run_epoch(
    model: DistributionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one train/eval epoch and return average loss."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        mu, sigma, nu = model(x_batch)
        loss = student_t_nll_loss(mu=mu, sigma=sigma, nu=nu, y_actual=y_batch, epsilon=config.EPSILON)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * x_batch.size(0)
        total_count += x_batch.size(0)

    return total_loss / max(total_count, 1)


def train_model(
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    num_features: int,
    ticker: str,
    feature_columns: list[str],
    norm_mean: pd.Series,
    norm_std: pd.Series,
    checkpoint_dir: Path | None = None,
) -> tuple[DistributionModel, dict[str, list[float]]]:
    """Train model with early stopping and save best checkpoint."""
    checkpoint_root = checkpoint_dir or config.CHECKPOINT_DIR
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / f"{ticker}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistributionModel(
        num_features=num_features,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    stale_epochs = 0
    history: dict[str, list[float]] = {"train": [], "val": []}

    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = _run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        with torch.no_grad():
            val_loss = _run_epoch(model=model, loader=val_loader, optimizer=None, device=device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch:03d} | train_nll={train_loss:.6f} | val_nll={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "num_features": num_features,
                    "feature_columns": feature_columns,
                    "norm_mean": norm_mean.to_dict(),
                    "norm_std": norm_std.to_dict(),
                    "config": {
                        "hidden_size": config.HIDDEN_SIZE,
                        "num_layers": config.NUM_LSTM_LAYERS,
                        "dropout": config.DROPOUT,
                    },
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("No model checkpoint was saved during training.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, history
