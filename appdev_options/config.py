"""Central configuration for the options ML pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

TICKERS: list[str] = ["SPY", "AAPL", "MSFT", "QQQ"]

LOOKBACK_WINDOW: int = 60

# Walk-forward boundaries (inclusive end points).
TRAIN_START: str = "2015-01-01"
TRAIN_END: str = "2021-12-31"
VAL_END: str = "2023-12-31"
TEST_END: str = "2025-12-31"

LEARNING_RATE: float = 1e-3
HIDDEN_SIZE: int = 64
NUM_LSTM_LAYERS: int = 2
DROPOUT: float = 0.1
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 100
EARLY_STOPPING_PATIENCE: int = 10

ITM_PROB_THRESHOLD: float = 0.60
SYNTHETIC_EXPIRY_DAYS: int = 5
RISK_FREE_RATE: float = 0.04

SEED: int = 42
CHECKPOINT_DIR: Path = Path("checkpoints")
EPSILON: float = 1e-6


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
