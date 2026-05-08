"""Offline evaluation plots and numeric scoring helpers."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import TARGET_COLUMN


def mean_prediction_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean(pred − actual). Positive values indicate systematic over-prediction on this slice;
    negative values indicate under-prediction.
    """
    return float(np.mean(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)))


def resolve_xgboost_learner(obj: Any) -> Any | None:
    """Unpack ``XGBoostLearner`` from a learner instance, stacked ensemble, or training bundle dict."""
    from models import StackedEnsemble, XGBoostLearner

    if isinstance(obj, XGBoostLearner):
        return obj
    if isinstance(obj, StackedEnsemble):
        for _, lr in obj.base_learners:
            if isinstance(lr, XGBoostLearner):
                return lr
        return None
    if isinstance(obj, dict) and "ensemble" in obj:
        return resolve_xgboost_learner(obj["ensemble"])
    return None


def plot_xgboost_learning_curves(
    learner_or_bundle: Any,
    *,
    title: str | None = None,
) -> plt.Figure | None:
    """
    Plot RMSE recorded during ``fit`` for each boosting round (train vs validation when both exist).

    Requires training with ``eval_set=[(X_train, y_train), (X_val, y_val)]`` so XGBoost stores two curves.
    Early stopping uses the last eval set (validation); a widening gap between train and validation
    curves indicates overfitting to the training slice.
    """
    learner = resolve_xgboost_learner(learner_or_bundle)
    if learner is None:
        return None
    model = getattr(learner, "model", None)
    if model is None or not callable(getattr(model, "evals_result", None)):
        return None
    evals = model.evals_result()
    if not evals:
        return None

    keys = list(evals.keys())
    metric_names = list(evals[keys[0]].keys())
    metric = metric_names[0]
    train_curve = np.asarray(list(evals[keys[0]][metric]), dtype=np.float64)
    rounds = np.arange(1, len(train_curve) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, train_curve, label="train", color="#1f77b4", lw=1.8)

    if len(keys) > 1:
        val_curve = np.asarray(list(evals[keys[1]][metric]), dtype=np.float64)
        ax.plot(rounds, val_curve, label="validation", color="#ff7f0e", lw=1.8)

    best_it = getattr(model, "best_iteration", None)
    if best_it is not None and best_it >= 0:
        ax.axvline(best_it + 1, color="gray", linestyle=":", lw=1.2, label=f"best iter ({best_it + 1})")

    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"XGBoost learning curve ({metric.upper()})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Return core regression diagnostics for held-out next-game predictions.

    Args:
        y_true: Observed target values (see ``config.TARGET_COLUMN``).
        y_pred: Model estimates aligned sample-wise with ``y_true``.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2_score(y_true, y_pred))}


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> plt.Figure:
    """
    Scatter actual vs predicted targets with diagonal reference overlay.

    Residual outliers (top 15% tail of absolute error) emphasize difficult games in red.
    """
    residuals = y_true - y_pred
    threshold = np.quantile(np.abs(residuals), 0.85)
    palette = np.where(np.abs(residuals) > threshold, "tab:red", "tab:blue")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.55, s=24, c=palette)

    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))

    lin = plt.Line2D([mn, mx], [mn, mx], color="black", linestyle="--", lw=1.2)
    ax.add_line(lin)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel(f"Actual {TARGET_COLUMN}")
    ax.set_ylabel(f"Predicted {TARGET_COLUMN}")

    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Plot residuals against fitted values to sanity-check curvature / heteroskedasticity.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.55, s=26, edgecolors="none", color="#2b5580")
    ax.axhline(0.0, color="gray", linestyle="--", lw=1)
    ax.set_xlabel(f"Predicted {TARGET_COLUMN}")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residual Diagnostics")
    return fig


def compare_models(results_dict: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
    """
    Grouped-bar comparison of MAE / RMSE for every contender (bases + stacked head).

    Args:
        results_dict: Maps descriptive model labels to ``(y_true, y_pred)`` pairs.
    """
    names = list(results_dict.keys())
    metrics: dict[str, dict[str, float]] = {}

    for name, (truth, preds) in results_dict.items():
        metrics[name] = compute_metrics(truth, preds)

    xs = np.arange(len(names))
    width = 0.35

    maes = [metrics[n]["mae"] for n in names]
    rmses = [metrics[n]["rmse"] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4.8))
    ax.bar(xs - width / 2, maes, width=width, label="MAE", color="#5470c6")
    ax.bar(xs + width / 2, rmses, width=width, label="RMSE", color="#91cc75")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel(f"{TARGET_COLUMN} — error scale")
    ax.set_title("Model Comparison — Lower is Better")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
