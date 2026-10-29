"""Performance metrics for fixed-size (points) and compounding (equity) backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def points_summary(trades: pd.DataFrame, col: str = "pnl_points") -> dict:
    pnl = trades[col].to_numpy()
    eq = np.cumsum(pnl)
    dd = (eq - np.maximum.accumulate(eq)).min()
    R = trades["R"] if "R" in trades else pd.Series(dtype=float)
    return dict(trades=len(pnl), total=pnl.sum(), avg=pnl.mean(),
                win_rate=(pnl > 0).mean() * 100, max_dd=dd,
                profit_dd=pnl.sum() / abs(dd) if dd != 0 else np.nan,
                avg_R=R.mean() if len(R) else np.nan,
                trade_sharpe_R=R.mean() / R.std() if len(R) else np.nan,
                sharpe_points_ann=pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))
