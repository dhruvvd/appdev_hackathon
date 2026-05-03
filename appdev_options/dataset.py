"""PyTorch dataset and strict walk-forward split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window time series dataset returning (X, y)."""

    def __init__(self, features: pd.DataFrame, target: pd.Series, lookback: int) -> None:
        if len(features) != len(target):
            raise ValueError("features and target must be aligned and equal length.")
        if lookback <= 0:
            raise ValueError("lookback must be positive.")
        if len(features) <= lookback:
            raise ValueError("Not enough rows for at least one sample.")

        self.features = features.to_numpy(dtype=np.float32)
        self.target = target.to_numpy(dtype=np.float32)
        self.lookback = lookback

    def __len__(self) -> int:
        return self.features.shape[0] - self.lookback

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_window = self.features[index : index + self.lookback]
        y_value = self.target[index + self.lookback]
        return torch.from_numpy(x_window), torch.tensor(y_value, dtype=torch.float32)


@dataclass(frozen=True)
class SplitData:
    """Container for walk-forward split datasets."""

    train: TimeSeriesDataset
    val: TimeSeriesDataset
    test: TimeSeriesDataset


def walk_forward_split(
    features: pd.DataFrame,
    target: pd.Series,
    lookback: int,
    train_end: str,
    val_end: str,
    test_end: str,
) -> SplitData:
    """Create walk-forward train/val/test datasets strictly by date."""
    if not features.index.equals(target.index):
        raise ValueError("features and target indices must be identical.")

    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    test_end_dt = pd.to_datetime(test_end)

    train_mask = features.index <= train_end_dt
    val_mask = (features.index > train_end_dt) & (features.index <= val_end_dt)
    test_mask = (features.index > val_end_dt) & (features.index <= test_end_dt)

    train_features = features.loc[train_mask]
    train_target = target.loc[train_mask]
    if train_features.empty:
        raise ValueError("Empty train split.")

    val_start_pos = features.index.get_indexer(features.index[val_mask])
    test_start_pos = features.index.get_indexer(features.index[test_mask])

    if len(val_start_pos) == 0 or len(test_start_pos) == 0:
        raise ValueError("Validation and test windows must be non-empty.")

    val_first = int(val_start_pos[0])
    test_first = int(test_start_pos[0])

    val_ctx_start = max(0, val_first - lookback)
    test_ctx_start = max(0, test_first - lookback)

    val_features = features.iloc[val_ctx_start : int(val_start_pos[-1]) + 1]
    val_target = target.iloc[val_ctx_start : int(val_start_pos[-1]) + 1]
    test_features = features.iloc[test_ctx_start : int(test_start_pos[-1]) + 1]
    test_target = target.iloc[test_ctx_start : int(test_start_pos[-1]) + 1]

    return SplitData(
        train=TimeSeriesDataset(train_features, train_target, lookback),
        val=TimeSeriesDataset(val_features, val_target, lookback),
        test=TimeSeriesDataset(test_features, test_target, lookback),
    )
