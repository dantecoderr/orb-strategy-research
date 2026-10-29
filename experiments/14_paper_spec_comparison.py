"""Run the reference paper's stated rules on TQQQ and NQ and set the results next to the paper's reported numbers.

Reference: Zarattini & Aziz, "Can Day Trading Really Be Profitable? Evidence of Sustainable Long-term Profits from
Opening Range Breakout (ORB) Day Trading Strategy vs. Benchmark in the US Stock Market" (Concretum Research / Peak Capital
Trading; first version 24 Apr 2023, version 22 Sep 2025). The paper tests QQQ/TQQQ, 2016 - 17 Feb 2023, with Interactive
Brokers data. This repo has TQQQ and NQ 2019-2025 from Databento, so only the 2019 - Feb 2023 overlap is comparable,
and the periods differ (the paper includes 2016-2018).

Paper rules: direction = first 5-minute candle; enter at the OPEN of the second candle; stop at the first candle's
low/high; target 10R or end of day; 1% risk, 4x max leverage, $25,000 start, $0.0005/share commission, no slippage.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from orb.data import load_minute_bars, split_adjust_intraday, to_5m, make_days, daily_atr
from orb.engine import run_orb
from orb.equity import size_trades, equity_summary

tq_dir = sys.argv[1] if len(sys.argv) > 1 else "data/tqqq"
nq_dir = sys.argv[2] if len(sys.argv) > 2 else "data/nq"

# ---------------- TQQQ ----------------
raw = load_minute_bars(tq_dir, "tqqq_*.csv", session=("09:30", "15:59"))
daily = raw.resample("1D").agg({"open": "first", "close": "last"}).dropna()
daily["gap"] = daily["open"] / daily["close"].shift(1)
splits = daily[(daily["gap"] < 0.60) | (daily["gap"] > 1.67)]["gap"]
def factor(d):
    f = 1.0
    for sd, r in splits.items():
        if sd.tz_localize(None) > d: f *= r
    return f
df5 = to_5m(split_adjust_intraday(raw)); days = make_days(df5); atr = daily_atr(df5)

def with_commission(t, per_share_per_side=0.0005):
    f = t["date"].map(factor).to_numpy(); x = t.copy()
    x["pnl_points"] = t["pnl_points"].to_numpy() - 2 * per_share_per_side * f   # as-traded $ -> split-adjusted $
    x["R"] = x["pnl_points"] / x["risk"]; return x

variants = {
    "ORB stop + 10R target / EoD (paper's main TQQQ rule)": run_orb(days, "orb", entry_mode="second_open", target_r=10),
    "ORB stop + EoD": run_orb(days, "orb", entry_mode="second_open"),
    "5% ATR stop + EoD (paper Sec. 4)": run_orb(days, "atr", atr_stop=atr["ATR"] * 0.05, entry_mode="second_open"),
}
windows = {"2019-01-01 to 2023-02-17 (paper's end date)": ("2019-01-01", "2023-02-17"), "2019-2025 (full sample here)": ("2019-01-01", "2025-12-31")}
rows = []
for vn, t in variants.items():
    t = with_commission(t)
    for wn, (a, b) in windows.items():
        w = t[(t["date"] >= a) & (t["date"] <= b)].reset_index(drop=True)
        s = equity_summary(size_trades(w))
        rows.append(dict(rule=vn, window=wn, trades=s["trades"], total_return_pct=s["return_pct"], cagr_pct=s["cagr_pct"],
                         max_dd_pct=s["max_dd_pct"], sharpe=s["sharpe_ann"], win_rate=s["win_rate"], avg_R=s["avg_R"],
                         cap_binds_pct=s["cap_binds_pct"]))
tq = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print("TQQQ under the paper's rules ($0.0005/share/side commission, no slippage, no other costs)\n")
print(tq.round(2).to_string(index=False))

# ---------------- NQ: signal-level comparison with the paper's QQQ ORB (win rate, trade mix, average R) ----------------
rawn = load_minute_bars(nq_dir, "nq_*.csv", session=("09:30", "15:59"))
dn = make_days(to_5m(rawn)); n = run_orb(dn, "orb", entry_mode="second_open")
print("\nNQ futures ORB (ORB stop + EoD, 2019-2025), signal-level statistics, no costs")
print(f"  trades {len(n)}, win rate {(n.pnl_points>0).mean():.1%}, long share {(n.direction==1).mean():.1%}, avg R {n.R.mean():.3f}")
nw = n[(n.date >= "2019-01-01") & (n.date <= "2023-02-17")]
print(f"  2019 - Feb 2023 only: trades {len(nw)}, win rate {(nw.pnl_points>0).mean():.1%}, avg R {nw.R.mean():.3f}")

print("""
Paper's reported numbers (2016 - 17 Feb 2023, IB data, QQQ / TQQQ):
  ORB QQQ : 1,795 trades, 51% long / 49% short, win rate 24%, avg PnL 0.13R, total return 676%, 33%/yr, Sharpe 1.13, MDD 22%
  ORB TQQQ: total return 1,484%, 48%/yr, Sharpe 1.19, MDD 28%, avg 0.18R, exposure capped on ~10% of days
  TQQQ, 5% ATR stop + EoD: +9,350% (text; the equity curve axis reaches $6.4M, which would be a larger multiple), alpha 93%
""")
