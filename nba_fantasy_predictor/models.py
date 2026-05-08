"""Base learners plus two-level stacked Ridge meta-ensemble."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def _coerce_finite_features(X: np.ndarray) -> np.ndarray:
    """Force finite float64 values — sklearn regressors reject NaNs even when rolling features exist."""
    return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


class RegressionLearner(ABC):
    """Minimal sklearn-like surface shared by pipeline stages."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class XGBoostLearner(RegressionLearner):
    """
    Histogram-based gradient boosting regressor with optional early stopping.

    When validation tensors are forwarded through ``fit(..., eval_set=[...])`` the
    underlying ``xgboost.XGBRegressor`` applies ``early_stopping_rounds`` from construction (or ``fit`` kwargs).
    Pass ``eval_set=[(X_train, y_train), (X_val, y_val)]`` so early stopping tracks the
    **last** set (validation) while ``evals_result()`` retains both train and validation curves.
    """

    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        subsample: float,
        random_state: int,
        *,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        early_stopping_rounds: int = 20,
    ) -> None:
        from xgboost import XGBRegressor

        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            gamma=gamma,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        X = _coerce_finite_features(X)
        eval_set = kwargs.pop("eval_set", None)
        early_rounds = int(kwargs.pop("early_stopping_rounds", self.early_stopping_rounds))
        if eval_set is not None:
            eval_set = [(_coerce_finite_features(ev_x), ev_y) for ev_x, ev_y in eval_set]
            try:
                self.model.fit(
                    X,
                    y,
                    eval_set=eval_set,
                    early_stopping_rounds=early_rounds,
                    verbose=False,
                    **kwargs,
                )
                return
            except TypeError:
                self.model.fit(X, y, eval_set=eval_set, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _coerce_finite_features(X)
        return self.model.predict(X)


class RandomForestLearner(RegressionLearner):
    """Classic bagged forest regressor exposing a thin wrapper."""

    def __init__(
        self,
        n_estimators: int,
        max_depth: int | None,
        random_state: int,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        del kwargs  # Unused — kept for API symmetry.
        X = _coerce_finite_features(X)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _coerce_finite_features(X)
        return self.model.predict(X)


class RidgeLearner(RegressionLearner):
    """
    Regularized ridge regression paired with ``StandardScaler`` fit on supplied training rows.

    The scaler never sees inference-only batches during ``fit``.
    """

    def __init__(self, alpha: float, random_state: int | None = None) -> None:
        del random_state  # Ridge is deterministic given ``alpha``.
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        del kwargs
        X = _coerce_finite_features(X)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _coerce_finite_features(X)
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


class StackedEnsemble:
    """
    Train base regressors on the training window and learn a ridge meta-model on stacked
    *validation* forecasts to avoid leakage from reusing training-set predictions.

    Prediction stacks base outputs in the insertion order supplied at construction time.
    """

    def __init__(
        self,
        base_learners: list[tuple[str, RegressionLearner]],
        meta_learner: Ridge,
    ) -> None:
        self.base_learners: list[tuple[str, RegressionLearner]] = list(base_learners)
        self.meta_learner = meta_learner
        self.base_order: list[str] = [name for name, _ in self.base_learners]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> StackedEnsemble:
        meta_rows_val: list[np.ndarray] = []

        for _, learner in self.base_learners:
            if isinstance(learner, XGBoostLearner):
                learner.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                )
            else:
                learner.fit(X_train, y_train)
            preds = learner.predict(X_val)
            meta_rows_val.append(preds)

        stacked_val = _coerce_finite_features(np.column_stack(meta_rows_val))
        self.meta_learner.fit(stacked_val, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for _, learner in self.base_learners:
            preds.append(learner.predict(X))
        stacked = _coerce_finite_features(np.column_stack(preds))
        return self.meta_learner.predict(stacked)


def instantiate_meta_ridge(alpha: float) -> Ridge:
    """Factory for sklearn ridge solver used purely as the stacking head."""
    return Ridge(alpha=alpha)
