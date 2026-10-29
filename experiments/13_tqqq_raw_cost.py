"""TQQQ costs in AS-TRADED cents: spread/slippage assumptions are converted to split-adjusted terms per date."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from orb.data import load_minute_bars, split_adjust_intraday, to_5m, make_days, daily_atr
from orb.engine import run_orb
from orb.equity import size_trades, equity_summary

raw = load_minute_bars(sys.argv[1] if len(sys.argv) > 1 else "data/tqqq", "tqqq_*.csv", session=("09:30", "15:59"))
daily = raw.resample("1D").agg({"open": "first", "close": "last"}).dropna()
daily["gap"] = daily["open"] / daily["close"].shift(1)
splits = daily[(daily["gap"] < 0.60) | (daily["gap"] > 1.67)]["gap"]
print("detected splits:", {str(k.date()): round(v, 3) for k, v in splits.items()})
def factor(d):   # adjusted price = raw price * factor
    f = 1.0
    for sd, r in splits.items():
        if sd.tz_localize(None) > d: f *= r
    return f

df = split_adjust_intraday(raw); df5 = to_5m(df); days = make_days(df5); atr = daily_atr(df5)
T = {"ORB stop": run_orb(days, "orb", entry_mode="first_close"),
     "ATR 5%": run_orb(days, "atr", atr_stop=atr["ATR"] * 0.05, entry_mode="first_close")}

print("\nmedian stop distance in AS-TRADED cents, by year")
for n, t in T.items():
    f = t["date"].map(factor); raw_c = (t["risk"] / f) * 100
    print(n, raw_c.groupby(t["date"].dt.year).median().round(1).to_dict())

def net(t, rt_raw_c, stop_extra_raw_c):
    f = t["date"].map(factor).to_numpy()
    cost = (rt_raw_c / 100) * f + (stop_extra_raw_c / 100) * f * (t["result"] == "stop").to_numpy()
    x = t.copy(); x["pnl_points"] = t["pnl_points"].to_numpy() - cost
    return x

rows = []
for n, t in T.items():
    for rt, ex, lab in [(0, 0, "no cost"), (1, 0, "1c round trip"), (1, 1, "1c + 1c extra on stop exits"), (2, 1, "2c + 1c extra on stops")]:
        s = equity_summary(size_trades(net(t, rt, ex)))
        rows.append(dict(strategy=n, scenario=lab, final_equity=s["final_equity"], cagr_pct=s["cagr_pct"], max_dd_pct=s["max_dd_pct"],
                         calmar=s["calmar"], sharpe=s["sharpe_ann"]))
print(); print(pd.DataFrame(rows).round(1).to_string(index=False))
