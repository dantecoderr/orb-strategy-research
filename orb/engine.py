"""Opening-range-breakout trade generation (one trade per day, no position sizing)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_orb(days: dict, mode: str = "orb", atr_stop: pd.Series | None = None,
            cost_points: float = 0.0, entry_mode: str = "second_open",
            target_r: float | None = None) -> pd.DataFrame:
    """Direction = colour of the first 5-minute bar; enter at the OPEN of the second bar.

    mode="orb": stop at the opposite extreme of the first bar.
    mode="atr": stop = entry -/+ atr_stop[date] (points / price units).
    Exit at the stop (filled exactly at the stop price) or at the last bar's close.
    ``cost_points`` is a round-trip cost in price units deducted from every trade.
    entry_mode="second_open": enter at the open of bar 2 (NQ notebooks);
    entry_mode="first_close": enter at the close of bar 1 (TQQQ notebooks).
    target_r: optional take-profit at that many R (stop is checked first on ties).
    """
    rows = []
    for d, x in days.items():
        o, h, l, c = x["o"], x["h"], x["l"], x["c"]
        if len(o) < 2 or o[0] == c[0]:
            continue
        direction = 1 if c[0] > o[0] else -1
        entry = o[1] if entry_mode == "second_open" else c[0]
        if mode == "orb":
            stop = l[0] if direction == 1 else h[0]
        else:
            s = atr_stop.get(d, np.nan)
            if not np.isfinite(s) or s <= 0:
                continue
            stop = entry - direction * s
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        hit = np.nonzero(l[1:] <= stop)[0] if direction == 1 else np.nonzero(h[1:] >= stop)[0]
        thit = np.array([], dtype=int)
        if target_r is not None:
            target = entry + direction * target_r * risk
            thit = np.nonzero(h[1:] >= target)[0] if direction == 1 else np.nonzero(l[1:] <= target)[0]
        if hit.size and (not thit.size or hit[0] <= thit[0]):
            exit_price, result, k = stop, "stop", hit[0] + 1
        elif thit.size:
            exit_price, result, k = target, "target", thit[0] + 1
        else:
            exit_price, result, k = c[-1], "eod", len(c) - 1
        pnl = direction * (exit_price - entry) - cost_points
        rng = h[0] - l[0]
        rows.append(dict(date=d, direction=direction, entry=entry, stop=stop,
                         exit_price=exit_price, result=result, exit_idx=k,
                         risk=risk, pnl_points=pnl, R=pnl / risk,
                         first_range=rng, first_body=abs(c[0] - o[0]),
                         first_volume=x["v"][0]))
    return pd.DataFrame(rows)
