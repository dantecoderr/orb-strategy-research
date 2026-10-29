"""TQQQ robustness: session fix, cost sensitivity (inside sizing), stop size/ticks, leverage cap, participation, by-year, concentration."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from orb.data import load_minute_bars, split_adjust_intraday, to_5m, make_days, daily_atr
from orb.engine import run_orb
from orb.equity import size_trades, equity_summary

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/tqqq"
pd.set_option("display.width", 220); os.makedirs("results", exist_ok=True)
cols = ["trades", "final_equity", "cagr_pct", "max_dd_pct", "calmar", "sharpe_ann", "avg_leverage", "cap_binds_pct"]

def build(session):
    df = split_adjust_intraday(load_minute_bars(DATA, "tqqq_*.csv", session=session))
    df5 = to_5m(df); days = make_days(df5); atr = daily_atr(df5)
    return dict(orb=run_orb(days, "orb", entry_mode="first_close"),
                atr=run_orb(days, "atr", atr_stop=atr["ATR"] * 0.05, entry_mode="first_close"))

orig, strict = build(("09:30", "16:00")), build(("09:30", "15:59"))
rows = []
for lab, res in [("16:00 inclusive (orig)", orig), ("15:59 strict", strict)]:
    for n in ["orb", "atr"]:
        rows.append(dict(session=lab, strategy=n, **{k: equity_summary(size_trades(res[n]))[k] for k in cols}))
print("=== 1. Session end ==="); print(pd.DataFrame(rows).round(2).to_string(index=False))
T = strict

# ---- 2. stop size in cents / ticks ---------------------------------------------------------
print("\n=== 2. Stop distance (risk per share) ===")
for n in ["orb", "atr"]:
    t = T[n]; c = t["risk"] * 100
    print(f"{n}: median {c.median():.1f}c, p10 {c.quantile(.1):.1f}c, p90 {c.quantile(.9):.1f}c | share <=2c: {(c<=2).mean()*100:.0f}%, <=3c: {(c<=3).mean()*100:.0f}%, <=5c: {(c<=5).mean()*100:.0f}% "
          f"| avg profit/share {t['pnl_points'].mean()*100:.2f}c")
yr = {n: (T[n].groupby(T[n]["date"].dt.year)["risk"].median() * 100).round(1) for n in ["orb", "atr"]}
print("median stop (cents) by year:\n", pd.DataFrame(yr).T.to_string())

# ---- 3. costs inside the sizing loop ---------------------------------------------------------
rows = []
for n in ["orb", "atr"]:
    for rt in [0, 0.01, 0.02, 0.03]:
        for ss in [0, 0.01]:
            s = equity_summary(size_trades(T[n], cost_per_share=rt, extra_stop_slip=ss))
            rows.append(dict(strategy=n, roundtrip_cents=rt * 100, extra_stop_slip_cents=ss * 100, **{k: s[k] for k in ["final_equity", "cagr_pct", "max_dd_pct", "calmar", "sharpe_ann"]}))
cost = pd.DataFrame(rows); cost.to_csv("results/tqqq_costs.csv", index=False)
print("\n=== 3. Cost sensitivity (compounded, costs inside sizing) ==="); print(cost.round(1).to_string(index=False))

# ---- 4. leverage cap + participation --------------------------------------------------------
print("\n=== 4. Leverage cap and participation (uncosted) ===")
for n in ["orb", "atr"]:
    x = size_trades(T[n]); part = x["shares"] / x["first_volume"]
    late = x[x["date"].dt.year >= 2024]
    print(f"{n}: cap binds on {x['lev_cap_binds'].mean()*100:.0f}% of trades | shares/first-5m-volume: median {part.median()*100:.1f}%, p90 {part.quantile(.9)*100:.1f}%, max {part.max()*100:.0f}% "
          f"| >5% of bar volume: {(part>0.05).mean()*100:.0f}% of trades | 2024-25 avg notional ${late['notional'].mean()/1e6:.1f}M, avg shares {late['shares'].mean():,.0f}")

# ---- 5. by year ---------------------------------------------------------------------------------
rows = {}
for n in ["orb", "atr"]:
    x = size_trades(T[n]); x["yr"] = x["date"].dt.year
    g = x.groupby("yr").agg(start=("equity_before", "first"), end=("equity_after", "last"), win=("pnl_dollars", lambda s: (s > 0).mean() * 100))
    rows[n + "_ret_%"] = (g["end"] / g["start"] - 1) * 100
print("\n=== 5. Calendar-year return (%, compounded, uncosted) ==="); print(pd.DataFrame(rows).round(0).to_string())

# ---- 6. concentration: rerun sizing with the best k trades (by R) removed -------------------
rows = []
for n in ["orb", "atr"]:
    t = T[n]; order = t["R"].sort_values(ascending=False).index
    for k in [0, 1, 5, 10, 20]:
        s = equity_summary(size_trades(t.drop(order[:k]).reset_index(drop=True)))
        rows.append(dict(strategy=n, removed_best=k, final_equity=s["final_equity"], cagr_pct=s["cagr_pct"], max_dd_pct=s["max_dd_pct"]))
print("\n=== 6. Compounded result with the best k trades (by R) removed ==="); print(pd.DataFrame(rows).round(1).to_string(index=False))
