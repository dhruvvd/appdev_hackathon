"""Inference utilities for projecting the next game's box-score outcome."""

from __future__ import annotations

from data_fetcher import fetch_logs_for_player_seasons
from dataset import attach_opponent_codes
from features import append_inference_row, engineer_features
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import numpy as np

from config import (
    INFERENCE_LAST_N_GAMES,
    MIN_GAMES_FOR_INFERENCE,
    MODELS_DIR,
    SEASONS,
    STACKED_ARTIFACT_FILENAME,
    TARGET_COLUMN,
)


def _season_slice(configured_seasons: list[str], through_season: str) -> list[str]:
    """Return seasons from the start through ``through_season`` inclusive; fallback to entire list."""
    if through_season not in configured_seasons:
        return list(configured_seasons)
    end_idx = configured_seasons.index(through_season) + 1
    return configured_seasons[:end_idx]


def load_trained_bundle(models_dir: Path | None = None) -> dict[str, Any]:
    """Load pickled ensemble artefacts produced by ``train.py``."""
    root = Path(models_dir or MODELS_DIR)
    path = root / STACKED_ARTIFACT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing bundle at {path} — train the models first.")
    return joblib.load(path)


def predict_next_game(
    player_id: int,
    season: str,
    *,
    bundle: dict[str, Any] | None = None,
    models_dir: Path | None = None,
    last_n_games: int | None = None,
) -> float:
    """
    Pull recent games from ``nba_api`` and infer ``TARGET_COLUMN`` for the hypothetical next matchup.

    Args:
        player_id: Official NBA STATS ``PLAYER_ID`` integer.
        season: NBA season slug (for example ``"2024-25"``) — controls how many seasons hydrate history.
        bundle: Optional pre-loaded bundle for batch inference.
        models_dir: Optional override pointing to artefacts.
        last_n_games: Trim history to the last N completed games after download (default: ``INFERENCE_LAST_N_GAMES``).

    Returns:
        Scalar projection for the upcoming contest.
    """
    kit = bundle or load_trained_bundle(models_dir)
    bundled_target = kit.get("target_column")
    if bundled_target is not None and bundled_target != TARGET_COLUMN:
        raise ValueError(
            f"Saved models target `{bundled_target}` but config.TARGET_COLUMN is `{TARGET_COLUMN}` — retrain with train.py.",
        )
    ensemble = kit["ensemble"]
    encoder = kit["opponent_encoder"]
    feature_cols = list(kit["feature_columns"])

    window_seasons = _season_slice(list(SEASONS), season)
    n_recent = INFERENCE_LAST_N_GAMES if last_n_games is None else last_n_games
    fetch_kw = {}
    if n_recent is not None:
        fetch_kw["last_n_games"] = int(n_recent)
    raw_logs = fetch_logs_for_player_seasons(player_id, window_seasons, **fetch_kw)
    if raw_logs.shape[0] < MIN_GAMES_FOR_INFERENCE:
        raise RuntimeError(
            f"Insufficient history for PID {player_id} — need ≥{MIN_GAMES_FOR_INFERENCE} games "
            "(raise INFERENCE_LAST_N_GAMES / widen SEASONS / lower rolling windows).",
        )

    player_name = str(raw_logs["PLAYER_NAME"].iloc[-1])
    augmented = append_inference_row(raw_logs)
    feature_frame, _ = engineer_features(augmented, require_target_column=False)

    pred_row = feature_frame.iloc[-1:].copy()
    if len(feature_frame) >= 2:
        prev = feature_frame.iloc[-2]
        idx = pred_row.index[0]
        for col in feature_cols:
            if col not in pred_row.columns:
                continue
            if pd.isna(pred_row.at[idx, col]):
                pred_row.at[idx, col] = prev[col]

    row_encoded = attach_opponent_codes(pred_row, encoder)
    numeric_x = row_encoded.loc[:, feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_x).all():
        numeric_x = np.nan_to_num(numeric_x, nan=0.0, posinf=0.0, neginf=0.0)

    preds = ensemble.predict(numeric_x)
    projected = float(preds.squeeze())

    last_five_actual = pd.to_numeric(
        raw_logs.sort_values("GAME_DATE")[TARGET_COLUMN].tail(5),
        errors="coerce",
    )
    avg_last5 = float(last_five_actual.mean())

    print(
        (
            f"\nPredicted next-game {TARGET_COLUMN} for PID {player_id} ({player_name}): "
            f"{projected:.2f}\n"
            f"  Last-five actual {TARGET_COLUMN} avg: {avg_last5:.2f}\n"
        )
    )
    return projected


if __name__ == "__main__":
    pid = int(input("NBA player ID: ").strip() or "2544")
    season_slug = input("Season tag (YYYY-YY, e.g. 2024-25): ").strip() or SEASONS[-1]
    predict_next_game(pid, season_slug)
