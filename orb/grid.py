"""Fast SL/TP grid evaluation on 1-minute paths (vectorised with searchsorted)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_paths(df1: pd.DataFrame, days: dict, trades: pd.DataFrame) -> list:
    """For every ORB trade store running-max favourable/adverse excursion (in points)
    from the entry minute onward, plus the end-of-day P&L."""
    by_day = {pd.Timestamp(d): g for d, g in df1.groupby(df1.index.date)}
    paths = []
    for d, direction, entry in zip(trades["date"], trades["direction"], trades["entry"]):
        entry_time = days[d]["t"][1]
        g = by_day[d]
        g = g[g.index >= entry_time]
        hi, lo, cl = g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()
        if direction == 1:
            fav, adv = hi - entry, entry - lo
        else:
            fav, adv = entry - lo, hi - entry
        paths.append(dict(date=d, cmA=np.maximum.accumulate(adv), cmF=np.maximum.accumulate(fav),
                          eod=direction * (cl[-1] - entry)))
    return paths


def pnl_tensor(paths: list, SL: np.ndarray, TP: np.ndarray) -> np.ndarray:
    """P&L (points) of every trade under every (SL, TP). Ties inside one minute -> stop first."""
    out = np.empty((len(paths), len(SL), len(TP)))
    for i, p in enumerate(paths):
        n = len(p["cmA"])
        si = np.searchsorted(p["cmA"], SL, side="left")
        ti = np.searchsorted(p["cmF"], TP, side="left")
        s_hit = (si < n)[:, None]
        t_hit = (ti < n)[None, :]
        stop_first = s_hit & (~t_hit | (si[:, None] <= ti[None, :]))
        out[i] = np.where(stop_first, -SL[:, None], np.where(t_hit, TP[None, :], p["eod"]))
    return out


def grid_metrics(T: np.ndarray) -> dict:
    """total, max drawdown and profit/DD for every grid cell of a (trades, SL, TP) tensor."""
    eq = np.cumsum(T, axis=0)
    dd = (eq - np.maximum.accumulate(eq, axis=0)).min(axis=0)
    total = eq[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pdd = np.where(dd != 0, total / np.abs(dd), -np.inf)
    return dict(total=total, dd=dd, profit_dd=pdd)
