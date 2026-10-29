"""Data loading and bar construction (Databento 1-minute OHLCV CSVs)."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

TZ = "America/New_York"


def load_minute_bars(folder: str, pattern: str = "*.csv", session=("09:30", "16:00"),
                     tz: str = TZ) -> pd.DataFrame:
    """Load yearly CSVs, convert UTC -> New York time, keep the session window.

    NOTE: pandas ``between_time`` is inclusive on both ends, so ("09:30", "16:00")
    keeps the 16:00 minute bar (which for Databento is the minute *starting* at 16:00,
    i.e. after the regular close). Use ("09:30", "15:59") for a strict RTH session.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    frames = []
    for p in paths:
        t = pd.read_csv(p)
        t.columns = t.columns.str.lower()
        ts_col = [c for c in t.columns if "ts" in c or "time" in c or "date" in c][0]
        t = t.rename(columns={ts_col: "ts_event"})
        t["ts_event"] = pd.to_datetime(t["ts_event"], utc=True)
        keep = [c for c in ["ts_event", "open", "high", "low", "close", "volume",
                            "instrument_id", "symbol"] if c in t.columns]
        frames.append(t[keep])
    df = pd.concat(frames, ignore_index=True).set_index("ts_event").sort_index()
    df.index = df.index.tz_convert(tz)
    df["volume"] = df["volume"].astype(float)
    return df.between_time(*session)


def split_adjust_intraday(df: pd.DataFrame, lo: float = 0.60, hi: float = 1.67) -> pd.DataFrame:
    """Back-adjust equity data for splits detected as large overnight gaps.

    Mirrors the original notebooks: the empirical gap ratio is used as the factor.
    """
    daily = df.resample("1D").agg({"open": "first", "close": "last"}).dropna()
    daily["prev_close"] = daily["close"].shift(1)
    daily["gap_ratio"] = daily["open"] / daily["prev_close"]
    cands = daily[(daily["gap_ratio"] < lo) | (daily["gap_ratio"] > hi)]
    adj = df.copy()
    for split_date, row in cands.iterrows():
        mask = adj.index < split_date
        adj.loc[mask, ["open", "high", "low", "close"]] *= row["gap_ratio"]
        adj.loc[mask, "volume"] /= row["gap_ratio"]
    return adj


def to_5m(df: pd.DataFrame) -> pd.DataFrame:
    return (df.resample("5min")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna())


def make_days(df5: pd.DataFrame) -> dict:
    """Group 5-minute bars by calendar date into numpy arrays (fast to scan)."""
    days = {}
    for d, g in df5.groupby(df5.index.date):
        days[pd.Timestamp(d)] = dict(
            t=g.index, o=g["open"].to_numpy(), h=g["high"].to_numpy(),
            l=g["low"].to_numpy(), c=g["close"].to_numpy(), v=g["volume"].to_numpy())
    return days


def daily_atr(df5: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Daily true range / ATR(n) from session bars. ATR is shifted one day (no look-ahead)."""
    g = df5.groupby(df5.index.date).agg(high=("high", "max"), low=("low", "min"),
                                        close=("close", "last"), open=("open", "first"),
                                        volume=("volume", "sum"))
    g.index = pd.to_datetime(g.index)
    g["prev_close"] = g["close"].shift(1)
    tr = pd.concat([g["high"] - g["low"],
                    (g["high"] - g["prev_close"]).abs(),
                    (g["low"] - g["prev_close"]).abs()], axis=1).max(axis=1)
    g["true_range"] = tr
    g["ATR"] = tr.rolling(n).mean().shift(1)
    return g
