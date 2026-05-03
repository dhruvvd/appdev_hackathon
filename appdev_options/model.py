"""Model definitions for return distribution forecasting."""

from __future__ import annotations

import torch
from torch import nn


class DistributionModel(nn.Module):
    """LSTM encoder that predicts Student-t parameters."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mu, sigma and nu for each input sample."""
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        raw = self.head(last_hidden)

        mu = raw[:, 0]
        sigma = torch.exp(raw[:, 1])
        nu = torch.exp(raw[:, 2]).clamp(min=2.5, max=30.0)
        return mu, sigma, nu
