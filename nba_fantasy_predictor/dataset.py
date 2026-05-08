"""Train / validation / test splits and opponent encoding (fit on train only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from config import TARGET_COLUMN, TEST_START, VAL_START

ID_METADATA_COLS = frozenset(
    {
        "PLAYER_ID",
        "PLAYER_NAME",
        "GAME_DATE",
        "MATCHUP",
        "opponent",
        "SEASON",
    }
)


def columns_blocked_from_features() -> frozenset[str]:
    """Schedule / identity labels plus target (and leaky totals when predicting PTS)."""
    blocked: set[str] = set(ID_METADATA_COLS) | {TARGET_COLUMN}
    if TARGET_COLUMN == "PTS":
        blocked.add("NBA_FANTASY_PTS")
    return frozenset(blocked)


@dataclass
class OpponentEncoder:
    """
    Label-style encoding for opponent abbreviations learned solely on the training fold.

    Any opponent unseen at ``fit`` time maps to a reserved unknown bucket to avoid
    sklearn ``LabelEncoder`` failures at inference.
    """

    UNK_TOKEN: str = "__UNK_OPPONENT__"
    classes_: list[str] | None = None
    _lookup: dict[str, int] | None = None

    def fit(self, opponents: Iterable[str]) -> OpponentEncoder:
        """Build the vocabulary from training opponents plus a synthetic unknown label."""
        unique = sorted({str(o) for o in opponents})
        self.classes_ = unique + [self.UNK_TOKEN]
        self._lookup = {label: idx for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, opponents: Iterable[str]) -> np.ndarray:
        """Encode opponent strings to contiguous integer indices (as float64 for numpy stacks)."""
        if self._lookup is None:
            raise RuntimeError("OpponentEncoder must be fit before transform.")
        unk = self._lookup[self.UNK_TOKEN]
        idxs = [self._lookup.get(str(o), unk) for o in opponents]
        return np.asarray(idxs, dtype=np.float64)


def split_by_date(
    df: pd.DataFrame,
    val_start: str | pd.Timestamp = VAL_START,
    test_start: str | pd.Timestamp = TEST_START,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition rows into train / validation / test using calendar ordering only.

    Args:
        df: Feature table that includes a parsed ``GAME_DATE`` column.
        val_start: First date (inclusive) belonging to the validation window.
        test_start: First date (inclusive) belonging to the held-out test window.
    """
    frame = df.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    val_start_ts = pd.to_datetime(val_start)
    test_start_ts = pd.to_datetime(test_start)

    train_mask = frame["GAME_DATE"] < val_start_ts
    val_mask = (frame["GAME_DATE"] >= val_start_ts) & (frame["GAME_DATE"] < test_start_ts)
    test_mask = frame["GAME_DATE"] >= test_start_ts

    return (
        frame.loc[train_mask].copy(),
        frame.loc[val_mask].copy(),
        frame.loc[test_mask].copy(),
    )


def attach_opponent_codes(df: pd.DataFrame, encoder: OpponentEncoder) -> pd.DataFrame:
    """Return a copy of ``df`` with an ``opponent_encoded`` column."""
    out = df.copy()
    out["opponent_encoded"] = encoder.transform(out["opponent"].astype(str))
    return out


def derive_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Infer model feature names from a training frame after opponent encoding.

    Everything except metadata, the regression target, and ``opponent`` text becomes a feature
    (including ``opponent_encoded``).
    """
    blocked = columns_blocked_from_features()
    return sorted(c for c in df.columns if c not in blocked)


def frames_to_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Materialize ``(X, y)`` numpy arrays for each split using a shared feature list.

    Rows with missing targets (e.g., live inference placeholders) should be omitted
    before calling this helper. ``y`` uses ``config.TARGET_COLUMN``.
    """
    def _pair(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
        y = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
        return x, y

    return _pair(train_df), _pair(val_df), _pair(test_df)
