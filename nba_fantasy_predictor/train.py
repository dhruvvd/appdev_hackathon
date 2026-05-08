"""Train next-game regressors and persist artefacts (XGBoost-only or stacked ensemble)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

from config import (
    META_LEARNER_TYPE,
    META_RIDGE_ALPHA,
    MODELS_DIR,
    RANDOM_SEED,
    RIDGE_ALPHA,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    STACKED_ARTIFACT_FILENAME,
    TARGET_COLUMN,
    USE_STACKED_ENSEMBLE,
    XGB_COLSAMPLE_BYLEVEL,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_GAMMA,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_N_ESTIMATORS,
    XGB_REG_ALPHA,
    XGB_REG_LAMBDA,
    XGB_SUBSAMPLE,
)
from data_fetcher import fetch_player_game_logs
from dataset import (
    OpponentEncoder,
    attach_opponent_codes,
    derive_feature_columns,
    frames_to_xy,
    split_by_date,
)
from features import engineer_features
from models import RandomForestLearner, RegressionLearner, RidgeLearner, StackedEnsemble, XGBoostLearner, instantiate_meta_ridge




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train next-game NBA projection model (see USE_STACKED_ENSEMBLE in config).")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download cached raw logs before training.",
    )
    return parser.parse_args()


def main(force_refresh: bool = False) -> None:
    if USE_STACKED_ENSEMBLE and META_LEARNER_TYPE.lower() != "ridge":
        raise ValueError(f"Unsupported meta learner `{META_LEARNER_TYPE}` in this codebase.")

    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    raw = fetch_player_game_logs(force_refresh=force_refresh)
    feature_df, _ = engineer_features(raw, require_target_column=True)

    train_df, val_df, test_df = split_by_date(feature_df)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Insufficient rows after split_by_date — relax VAL_START.")

    opponent_encoder = OpponentEncoder().fit(train_df["opponent"].astype(str))
    train_enc = attach_opponent_codes(train_df, opponent_encoder)
    val_enc = attach_opponent_codes(val_df, opponent_encoder)
    test_enc = attach_opponent_codes(test_df, opponent_encoder)

    feature_cols = derive_feature_columns(train_enc)
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = frames_to_xy(
        train_enc, val_enc, test_enc, feature_cols
    )

    xgb_lr = XGBoostLearner(
        XGB_N_ESTIMATORS,
        XGB_MAX_DEPTH,
        XGB_LEARNING_RATE,
        XGB_SUBSAMPLE,
        RANDOM_SEED,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        reg_lambda=XGB_REG_LAMBDA,
        reg_alpha=XGB_REG_ALPHA,
        gamma=XGB_GAMMA,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        colsample_bylevel=XGB_COLSAMPLE_BYLEVEL,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
    )

    if USE_STACKED_ENSEMBLE:
        bases: list[tuple[str, RegressionLearner]] = [
            ("xgboost", xgb_lr),
            ("random_forest", RandomForestLearner(RF_N_ESTIMATORS, RF_MAX_DEPTH, RANDOM_SEED)),
            ("ridge_scaled", RidgeLearner(RIDGE_ALPHA, RANDOM_SEED)),
        ]

        stacked = StackedEnsemble(base_learners=bases, meta_learner=instantiate_meta_ridge(META_RIDGE_ALPHA))
        stacked.fit(x_tr, y_tr, x_val, y_val)
        predictor = stacked

        print("\nValidation diagnostics (MAE):")
        for name, learner in stacked.base_learners:
            preds = learner.predict(x_val)
            mae_val = mean_absolute_error(y_val, preds)
            print(f"  {name:>15} {mae_val:.3f}")

        stacked_val_pred = stacked.predict(x_val)
        print(f"  {'stacked_ridge':>15} {mean_absolute_error(y_val, stacked_val_pred):.3f}\n")

        _, rf_lr = bases[1]
        _, ridge_lr = bases[2]
        joblib.dump(rf_lr, models_dir / "random_forest.joblib")
        joblib.dump(ridge_lr, models_dir / "ridge.joblib")
        joblib.dump(stacked.meta_learner, models_dir / "ridge_meta.joblib")
    else:
        xgb_lr.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_val, y_val)])
        predictor = xgb_lr

        xgb_val_pred = xgb_lr.predict(x_val)
        print("\nValidation diagnostics (MAE):")
        print(f"  {'xgboost':>15} {mean_absolute_error(y_val, xgb_val_pred):.3f}\n")

    joblib.dump(opponent_encoder, models_dir / "opponent_encoder.joblib")
    joblib.dump(xgb_lr, models_dir / "xgboost.joblib")

    bundle = {
        "ensemble": predictor,
        "opponent_encoder": opponent_encoder,
        "feature_columns": feature_cols,
        "target_column": TARGET_COLUMN,
    }
    joblib.dump(bundle, models_dir / STACKED_ARTIFACT_FILENAME)
    print(f"Saved artefacts to {models_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(force_refresh=args.force_refresh)
