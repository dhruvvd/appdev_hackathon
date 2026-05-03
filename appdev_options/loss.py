"""Loss functions for probabilistic return models."""

from __future__ import annotations

import torch


def student_t_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
    y_actual: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute mean negative log-likelihood under Student-t."""
    dist = torch.distributions.StudentT(df=nu, loc=mu, scale=sigma + epsilon)
    return -dist.log_prob(y_actual).mean()
