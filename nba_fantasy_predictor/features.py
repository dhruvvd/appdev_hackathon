"""Time-aware rolling features for next-game NBA projection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROLLING_WINDOWS, TARGET_COLUMN

ROLL_STATS = (
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "MIN",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
)


def infer_season_string(dates: pd.Series) -> pd.Series:
    """Map game dates to ``'YYYY-YY'`` NBA campaign labels (October–July)."""
    y = dates.dt.year
    adj = np.where(dates.dt.month >= 10, y, y - 1).astype(np.int64)
    tail = ((adj + 1) % 100).astype(np.int64)
    return pd.Series([f"{sy}-{tg:02d}" for sy, tg in zip(adj.tolist(), tail.tolist())], index=dates.index)


def extract_opponent_abbr(matchup: str) -> str:
    """Return opponent abbreviation from NBA ``MATCHUP`` string (home or away)."""
    if matchup is None or (isinstance(matchup, float) and np.isnan(matchup)):
        return "UNK"
    tokens = str(matchup).strip().split()
    return tokens[-1] if tokens else "UNK"


def engineer_features(
    raw_df: pd.DataFrame,
    rolling_windows: tuple[int, ...] | None = None,
    *,
    require_target_column: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build shifted rolling features per player — no lookahead into the same game's boxscore.

    For each statistic, ``shift(1)`` precedes rolling so row *t* uses only strictly prior games.

    Returns ``(feature_frame, target)`` on rows where all engineered rolling fields are finite.

    The first ``max(rolling_windows)`` games per player's career omit long-window aggregates and drop.
    Target labels come from ``config.TARGET_COLUMN``.
    """
    rolling_windows = tuple(rolling_windows or ROLLING_WINDOWS)
    df = raw_df.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=(True, True)).reset_index(drop=True)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"])

    for col in ROLL_STATS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_home"] = df["MATCHUP"].astype(str).str.contains(r"\bvs\.").astype(np.int8)
    df["opponent"] = df["MATCHUP"].map(extract_opponent_abbr)

    if "SEASON" not in df.columns:
        df["SEASON"] = infer_season_string(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    df["season_game_number"] = df.groupby(["PLAYER_ID", "SEASON"], observed=True).cumcount() + 1

    g = df.groupby("PLAYER_ID", group_keys=False)
    df["days_rest"] = g["GAME_DATE"].diff().dt.days
    is_first_team_game = df.groupby("PLAYER_ID", observed=True).cumcount() == 0
    df.loc[is_first_team_game, "days_rest"] = np.nan

    median_rest = df["days_rest"].median(skipna=True)
    if np.isnan(median_rest):
        median_rest = 2.0
    df["days_rest"] = df["days_rest"].fillna(median_rest).clip(lower=0)
    df["is_back_to_back"] = ((df["days_rest"] == 1) & (~is_first_team_game)).astype(np.int8)

    g2 = df.groupby("PLAYER_ID", group_keys=False)
    for window in rolling_windows:
        for col in ROLL_STATS:
            df[f"{col}_avg_{window}"] = g2[col].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
            )
    df["NBA_FANTASY_PTS_std_5"] = g2["NBA_FANTASY_PTS"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=5).std()
    )

    feature_checks = []
    for w in rolling_windows:
        feature_checks.extend([f"{c}_avg_{w}" for c in ROLL_STATS])
    feature_checks.append("NBA_FANTASY_PTS_std_5")

    df = df.dropna(subset=feature_checks).reset_index(drop=True)
    if require_target_column:
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    y = df[TARGET_COLUMN].copy()
    return df, y


def append_inference_row(history_df: pd.DataFrame, next_day_offset: int = 2) -> pd.DataFrame:
    """
    Append a synthetic next-game row for live prediction.

    The placeholder copies the prior game's box score into raw stat columns (included where allowed);
    the regression target column is cleared for the synthetic row. Rolling columns still use ``shift(1)``.

    Args:
        history_df: Single-player slice of cached raw logs (already sorted ascending by date).
        next_day_offset: Calendar-day gap after the player's last logged game before the tentative next tip.
    """
    base = history_df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True).copy()
    last = base.iloc[-1]

    matchup_tokens = str(last.get("MATCHUP", "")).strip().split()
    team = matchup_tokens[0] if matchup_tokens else "UNK"

    # Copy last game's box score so raw stat columns (included as model features) are finite.
    # Rolling columns still use shift(1), so they only reflect completed games.
    row = last.copy()
    row["GAME_DATE"] = pd.to_datetime(last["GAME_DATE"]) + pd.Timedelta(days=max(1, int(next_day_offset)))
    row["MATCHUP"] = f"{team} vs. UNK"
    next_ts = pd.to_datetime(row["GAME_DATE"])
    row["SEASON"] = (
        last["SEASON"]
        if pd.notna(last.get("SEASON"))
        else infer_season_string(pd.Series([next_ts], dtype="datetime64[ns]")).iloc[0]
    )
    row[TARGET_COLUMN] = np.nan
    if TARGET_COLUMN != "NBA_FANTASY_PTS":
        row["NBA_FANTASY_PTS"] = np.nan
    phantom = pd.DataFrame([row.to_dict()])
    return pd.concat([base, phantom], ignore_index=True)
