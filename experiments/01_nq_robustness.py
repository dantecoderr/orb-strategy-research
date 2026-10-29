"""NQ robustness: session fix, concentration, bootstrap, costs, equal-risk % metrics, by-year."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from orb.data import load_minute_bars, to_5m, make_days, daily_atr
from orb.engine import run_orb
from orb.metrics import points_summary

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/nq"
OUT = "results"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)

def build(session):
    df = load_minute_bars(DATA, "nq_*.csv", session=session)
    df5 = to_5m(df); days = make_days(df5); atr = daily_atr(df5)
    return dict(orb=run_orb(days, "orb"), atr=run_orb(days, "atr", atr_stop=atr["ATR"] * 0.05))

# ---- 1. session end: inclusive 16:00 (original) vs strict 15:59 ------------------------
orig, strict = build(("09:30", "16:00")), build(("09:30", "15:59"))
rows = []
for name in ["orb", "atr"]:
    for lab, res in [("session to 16:00 (orig)", orig), ("session to 15:59", strict)]:
        s = points_summary(res[name]); rows.append(dict(strategy=name, session=lab, **{k: s[k] for k in ["trades", "total", "max_dd", "profit_dd", "win_rate"]}))
print("\n=== 1. EOD exit bar: 16:00 vs 15:59 ==="); print(pd.DataFrame(rows).round(2).to_string(index=False))
T = strict   # strict RTH exit (15:59 close) is the basis for everything below

# ---- 2. concentration ------------------------------------------------------------------
rows = []
for name in ["orb", "atr"]:
    p = T[name]["pnl_points"].sort_values(ascending=False).to_numpy()
    for k in [0, 1, 5, 10, 20, 50]:
        rows.append(dict(strategy=name, removed_top=k, total=p[k:].sum(), avg_per_trade=p[k:].mean()))
print("\n=== 2. Total points after removing the best k trades ==="); print(pd.DataFrame(rows).round(2).to_string(index=False))
for name in ["orb", "atr"]:
    p = T[name]["pnl_points"]; print(f"{name}: best trade = {p.max():.1f} pts = {p.max()/p.sum()*100:.0f}% of total; "
          f"top-20 = {p.nlargest(20).sum():.0f} pts; median trade = {p.median():.1f}; winners = {(p>0).mean()*100:.1f}%")

# ---- 3. bootstrap ----------------------------------------------------------------------
rng = np.random.default_rng(42); rows = []
for name in ["orb", "atr"]:
    p = T[name]["pnl_points"].to_numpy(); n = len(p)
    idx = rng.integers(0, n, size=(10000, n)); b = p[idx]
    tot = b.sum(1); sh = b.mean(1) / b.std(1, ddof=1) * np.sqrt(252)
    rows.append(dict(strategy=name, total_5pct=np.percentile(tot, 5), total_median=np.median(tot), total_95pct=np.percentile(tot, 95),
                     P_total_le_0=(tot <= 0).mean(), sharpe_5pct=np.percentile(sh, 5), sharpe_median=np.median(sh), sharpe_95pct=np.percentile(sh, 95)))
print("\n=== 3. i.i.d. bootstrap of trades (10,000 resamples) ==="); print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- 4. cost sensitivity (NQ: 1 tick = 0.25 pt = $5) ------------------------------------
rows = []
for name in ["orb", "atr"]:
    t = T[name]; is_stop = (t["result"] == "stop").to_numpy()
    for rt in [0, 0.5, 1, 1.5, 2, 3, 4]:
        for ss in [0, 0.5]:
            net = t["pnl_points"].to_numpy() - rt - ss * is_stop
            rows.append(dict(strategy=name, roundtrip_pts=rt, extra_stop_slip_pts=ss, usd_rt=rt * 20, total=net.sum(), avg=net.mean(),
                             sharpe_ann=net.mean() / net.std(ddof=1) * np.sqrt(252)))
cost = pd.DataFrame(rows); cost.to_csv(f"{OUT}/nq_costs.csv", index=False)
print("\n=== 4. Cost sensitivity (points per NQ contract; $ per round trip in usd_rt) ==="); print(cost.round(2).to_string(index=False))
for name in ["orb", "atr"]:
    t = T[name]; st = (t["result"] == "stop").mean()
    print(f"{name}: break-even round-trip cost = {t['pnl_points'].mean():.2f} pts (${t['pnl_points'].mean()*20:.0f}); {st*100:.0f}% of trades exit at a stop")

# ---- 5. equal-risk compounding (risk f of equity per trade) -> % metrics ----------------
rows = []
for name in ["orb", "atr"]:
    t = T[name]; yrs = (t["date"].iloc[-1] - t["date"].iloc[0]).days / 365.25
    for f in [0.005, 0.01]:
        r = f * t["R"].to_numpy(); eq = np.cumprod(1 + r)
        cagr = eq[-1] ** (1 / yrs) - 1; mdd = (eq / np.maximum.accumulate(eq) - 1).min()
        rows.append(dict(strategy=name, risk_per_trade=f, CAGR_pct=cagr * 100, maxDD_pct=mdd * 100, Calmar=cagr / abs(mdd),
                         sharpe_ann=r.mean() / r.std(ddof=1) * np.sqrt(252), final_mult=eq[-1],
                         avg_notional_leverage=(f * t["entry"] / t["risk"]).mean()))
print("\n=== 5. Equal-risk compounding on R-multiples (no costs; fractional contracts) ==="); print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- 6. by year -------------------------------------------------------------------------
yr = pd.concat({n: T[n].groupby(T[n]["date"].dt.year)["pnl_points"].agg(["sum", "count"]) for n in ["orb", "atr"]}, axis=1)
yr.columns = ["orb_pts", "orb_n", "atr_pts", "atr_n"]
print("\n=== 6. Points by year ==="); print(yr.round(1).to_string())
print("ORB: share of total from 2025:", round(yr.orb_pts.loc[2025] / yr.orb_pts.sum() * 100), "%")
