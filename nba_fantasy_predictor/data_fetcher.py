"""Fetch player game logs from nba_api with caching and rate limiting."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config import DATA_CACHE_PATH, PLAYER_IDS, SEASONS

logger = logging.getLogger(__name__)

WANTED_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "MATCHUP",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
]


def _fetch_player_season_logs(player_id: int, season: str) -> pd.DataFrame | None:
    """Call NBA Stats for one player-season; return None if empty or failed."""
    from nba_api.stats.endpoints.playergamelogs import PlayerGameLogs

    def call() -> pd.DataFrame:
        resp = PlayerGameLogs(
            player_id_nullable=str(player_id),
            season_nullable=season,
            season_type_nullable="Regular Season",
        )
        return resp.get_data_frames()[0]

    df: pd.DataFrame | None = None
    try:
        df = call()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 429:
            logger.warning("HTTP 429 for player %s season %s; retry after 5s", player_id, season)
            time.sleep(5)
            try:
                df = call()
            except Exception as exc2:
                logger.error("Retry failed for player %s season %s: %s", player_id, season, exc2)
                time.sleep(1)
                return None
        else:
            logger.error("HTTP error for player %s season %s: %s", player_id, season, exc)
            time.sleep(1)
            return None
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429 or "429" in str(exc).lower():
            logger.warning("Rate limited player %s season %s; retry after 5s", player_id, season)
            time.sleep(5)
            try:
                df = call()
            except Exception as exc2:
                logger.error("Retry failed for player %s season %s: %s", player_id, season, exc2)
                time.sleep(1)
                return None
        else:
            logger.error("API failure for player %s season %s: %s", player_id, season, exc)
            time.sleep(1)
            return None

    time.sleep(1)

    if df is None or df.empty:
        logger.info("Skipping empty PlayerGameLogs for player %s season %s", player_id, season)
        return None

    return df


def _normalize_columns(df: pd.DataFrame, season_tag: str) -> pd.DataFrame:
    """Select / rename columns to the project schema."""
    rename_map = {
        # Some seasons / endpoints omit prefix
        "FANTASY_PTS": "NBA_FANTASY_PTS",
    }

    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    missing = [c for c in WANTED_COLUMNS if c not in df.columns]
    if "NBA_FANTASY_PTS" in missing:
        pts = pd.to_numeric(df.get("PTS", 0), errors="coerce").fillna(0)
        reb = pd.to_numeric(df.get("REB", 0), errors="coerce").fillna(0)
        ast = pd.to_numeric(df.get("AST", 0), errors="coerce").fillna(0)
        stl = pd.to_numeric(df.get("STL", 0), errors="coerce").fillna(0)
        blk = pd.to_numeric(df.get("BLK", 0), errors="coerce").fillna(0)
        tov = pd.to_numeric(df.get("TOV", 0), errors="coerce").fillna(0)
        fg3m = pd.to_numeric(df.get("FG3M", 0), errors="coerce").fillna(0)
        dd = pd.to_numeric(df.get("DD2", 0), errors="coerce").fillna(0).astype(int)
        td = pd.to_numeric(df.get("TD3", 0), errors="coerce").fillna(0).astype(int)
        df["NBA_FANTASY_PTS"] = (
            pts
            + 1.2 * reb
            + 1.5 * ast
            + 3 * stl
            + 3 * blk
            + 3 * fg3m
            - 1 * tov
            + 12 * dd
            + 18 * td
        )
        missing.remove("NBA_FANTASY_PTS")

    remaining = [c for c in WANTED_COLUMNS if c not in df.columns]
    if remaining:
        raise ValueError(f"Missing required columns after normalization: {remaining}")

    out = df[WANTED_COLUMNS].copy()
    out["SEASON"] = season_tag
    return out


def fetch_player_game_logs(
    cache_path: Path | None = None,
    player_ids: list[int] | None = None,
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load combined player game logs for configured players and seasons.

    Uses CSV cache when present unless ``force_refresh`` is True. Between every
    successful API request, sleeps 1 second to respect rate limits. Empty
    player-season responses are skipped without error.
    """
    cache_path = Path(cache_path or DATA_CACHE_PATH)
    player_ids = player_ids or PLAYER_IDS
    seasons = seasons or SEASONS

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
        return df.sort_values(["PLAYER_NAME", "GAME_DATE"], ascending=True).reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for pid in player_ids:
        for season in seasons:
            raw = _fetch_player_season_logs(pid, season)
            if raw is None or raw.empty:
                continue
            try:
                frames.append(_normalize_columns(raw, season))
            except ValueError as exc:
                logger.warning("Column normalization failed for %s %s: %s", pid, season, exc)
                continue

    if not frames:
        raise RuntimeError("No game log data retrieved; check player IDs, seasons, and connectivity.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(cache_path, index=False)
    return combined.sort_values(["PLAYER_NAME", "GAME_DATE"], ascending=True).reset_index(drop=True)


def fetch_logs_for_player_seasons(
    player_id: int,
    seasons: list[str],
    *,
    last_n_games: int | None = None,
) -> pd.DataFrame:
    """
    Fetch regular-season logs for one player across given seasons without the global CSV cache.

    Used for inference so forecasts always reflect the latest outings. Applies the same
    1-second pacing and graceful empty skips as the bulk downloader.

    The NBA Stats ``PlayerGameLogs`` endpoint returns full seasons; ``last_n_games`` optionally
    trims to the most recent rows after merging seasons (client-side only).
    """
    frames: list[pd.DataFrame] = []
    for season in seasons:
        raw = _fetch_player_season_logs(player_id, season)
        if raw is None or raw.empty:
            continue
        try:
            frames.append(_normalize_columns(raw, season))
        except ValueError as exc:
            logger.warning("normalize failed pid=%s season=%s err=%s", player_id, season, exc)
    if not frames:
        raise RuntimeError(f"No game logs returned for player_id={player_id} seasons={seasons}.")
    out = pd.concat(frames, ignore_index=True).sort_values("GAME_DATE", ascending=True)
    if last_n_games is not None and last_n_games > 0:
        out = out.tail(int(last_n_games)).reset_index(drop=True)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = fetch_player_game_logs(force_refresh=False)
    print(out.shape)
    print(out.head())
