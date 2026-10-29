"""Risk-based position sizing for an equity/ETF ORB strategy (compounding, leverage cap)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def size_trades(trades: pd.DataFrame, start_equity: float = 25_000, risk_per_trade: float = 0.01,
                max_leverage: float = 4.0, cost_per_share: float = 0.0,
                extra_stop_slip: float = 0.0) -> pd.DataFrame:
    """Sequentially size one trade per day from CLOSED-trade equity.

    shares = min(floor(equity*risk / risk_per_share), floor(equity*max_leverage / entry)).
    cost_per_share: round-trip $ per share; extra_stop_slip: extra $ per share on stop exits.
    Costs are applied inside the loop so they compound through position sizing.
    """
    eq = start_equity
    rows = []
    entry_a, risk_a = trades["entry"].to_numpy(), trades["risk"].to_numpy()
    pnl_a, stop_exit = trades["pnl_points"].to_numpy(), (trades["result"] == "stop").to_numpy()
    for i in range(len(trades)):
        e, r = entry_a[i], risk_a[i]
        if r <= 0 or e <= 0:
            continue
        by_risk, by_lev = int(np.floor(eq * risk_per_trade / r)), int(np.floor(eq * max_leverage / e))
        shares = max(min(by_risk, by_lev), 0)
        if shares <= 0:
            continue
        net_ps = pnl_a[i] - cost_per_share - extra_stop_slip * stop_exit[i]
        pnl = net_ps * shares
        rows.append((i, shares, eq, eq + pnl, pnl, r * shares, e * shares, by_lev < by_risk, net_ps))
        eq += pnl
    out = pd.DataFrame(rows, columns=["idx", "shares", "equity_before", "equity_after", "pnl_dollars",
                                      "dollar_risk", "notional", "lev_cap_binds", "net_pnl_ps"])
    tr = trades.iloc[out["idx"]].reset_index(drop=True)
    out = pd.concat([tr, out.drop(columns="idx")], axis=1)
    out["risk_pct_used"] = out["dollar_risk"] / out["equity_before"]
    out["leverage_used"] = out["notional"] / out["equity_before"]
    out["peak"] = np.maximum(out["equity_after"].cummax(), start_equity)
    out["drawdown"] = out["equity_after"] - out["peak"]
    out["dd_pct"] = out["equity_after"] / out["peak"] - 1
    return out


def equity_summary(x: pd.DataFrame, start_equity: float = 25_000) -> dict:
    if x.empty:
        return {}
    fin = x["equity_after"].iloc[-1]
    yrs = (x["date"].max() - x["date"].min()).days / 365.25
    cagr = (fin / start_equity) ** (1 / yrs) - 1 if fin > 0 else np.nan
    ret = (x["equity_after"] / x["equity_before"] - 1).to_numpy()
    mdd = x["dd_pct"].min()
    prof, dd = x["pnl_dollars"].sum(), x["drawdown"].min()
    return dict(trades=len(x), final_equity=fin, return_pct=(fin / start_equity - 1) * 100, cagr_pct=cagr * 100,
                max_dd_usd=dd, profit_dd=prof / abs(dd) if dd else np.nan, max_dd_pct=mdd * 100,
                calmar=cagr / abs(mdd) if mdd else np.nan, sharpe_ann=ret.mean() / ret.std(ddof=1) * np.sqrt(252),
                win_rate=(x["pnl_dollars"] > 0).mean() * 100, avg_R=x["R"].mean(), avg_shares=x["shares"].mean(),
                avg_risk_pct=x["risk_pct_used"].mean() * 100, avg_leverage=x["leverage_used"].mean(),
                cap_binds_pct=x["lev_cap_binds"].mean() * 100)
