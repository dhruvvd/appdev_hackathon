"""Central configuration for the NBA stacked regression pipeline."""

from __future__ import annotations

from pathlib import Path

# Project root directory (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Next-game regression target (must exist on PlayerGameLogs rows)
TARGET_COLUMN = "PTS"

# After downloading season logs for inference, optionally keep only the last N games.
# None keeps full history. Needs N >= max(ROLLING_WINDOWS) + headroom (~25).
INFERENCE_LAST_N_GAMES: int | None = 30

# Minimum completed games required before building a phantom next-game row
MIN_GAMES_FOR_INFERENCE = 25

# Star / high-usage players to model (NBA numeric player IDs)
PLAYER_IDS = [
    2544,  # LeBron James
    201939,  # Stephen Curry
    201142,  # Kevin Durant
    203507,  # Giannis Antetokounmpo
    203999,  # Nikola Jokic
    1629029,  # Luka Doncic
    203954,  # Joel Embiid
    1628369,  # Jayson Tatum
    1626164,  # Devin Booker
    203076,  # Anthony Davis
    202695,  # Kawhi Leonard
    202331,  # Paul George
    203081,  # Damian Lillard
    201935,  # James Harden
    202681,  # Kyrie Irving
    1629630,  # Ja Morant
    1629027,  # Trae Young
    1628378,  # Donovan Mitchell
    1628983,  # Shai Gilgeous-Alexander
    1627759,  # Jaylen Brown
    1627734,  # Jamal Murray
]

# Deduplicate while preserving order (defensive)
_seen: set[int] = set()
PLAYER_IDS = [pid for pid in PLAYER_IDS if not (pid in _seen or _seen.add(pid))]

# Season strings accepted by nba_api PlayerGameLogs (2025-26 includes games played in calendar 2026).
# If you extend this list or PLAYER_IDS, run ``train.py --force-refresh`` once so the CSV cache repulls.
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

ROLLING_WINDOWS = (5, 10)

# Chronological split boundaries (inclusive of game date on val/test start)
# Train: GAME_DATE < VAL_START
# Val:   VAL_START <= GAME_DATE < TEST_START
# Test:  GAME_DATE >= TEST_START
VAL_START = "2025-10-01"
TEST_START = "2026-01-01"

META_LEARNER_TYPE = "ridge"
META_RIDGE_ALPHA = 1.0

# When False, train a single XGBoostLearner (validation used for early stopping only).
# When True, stack XGBoost + RandomForest + Ridge with a Ridge meta-learner.
USE_STACKED_ENSEMBLE = False

RANDOM_SEED = 42

# Raw combined game logs cache (CSV)
DATA_CACHE_PATH = PROJECT_ROOT / "data" / "raw_player_gamelogs.csv"

# Trained artifacts
MODELS_DIR = PROJECT_ROOT / "models_artifacts"

# Single-file bundle for inference (ensemble + encoder + feature order)
STACKED_ARTIFACT_FILENAME = "stacked_bundle.joblib"

# XGBoost — defaults biased toward generalization (narrow train vs validation gap).
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.045
XGB_SUBSAMPLE = 0.75
XGB_COLSAMPLE_BYTREE = 0.75
XGB_COLSAMPLE_BYLEVEL = 0.75
XGB_MIN_CHILD_WEIGHT = 6
XGB_REG_LAMBDA = 6.0
XGB_REG_ALPHA = 0.4
XGB_GAMMA = 0.2
XGB_EARLY_STOPPING_ROUNDS = 35

# Random Forest
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 12

# Ridge (base + default meta)
RIDGE_ALPHA = 1.0
