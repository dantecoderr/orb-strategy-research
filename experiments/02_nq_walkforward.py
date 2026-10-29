"""NQ walk-forward: re-select SL/TP (and an opening-range filter) each year using only prior years."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from orb.data import load_minute_bars, to_5m, make_days
from orb.engine import run_orb
from orb.grid import build_paths, pnl_tensor, grid_metrics

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/nq"
pd.set_option("display.width", 200)
SL = np.arange(20, 201, 10).astype(float); TP = np.arange(40, 401, 10).astype(float)

def prep(session):
    df1 = load_minute_bars(DATA, "nq_*.csv", session=session)
    days = make_days(to_5m(df1)); tr = run_orb(days, "orb")
    paths = build_paths(df1, days, tr)
    return tr, pnl_tensor(paths, SL, TP)

# ---- validation against the notebook (inclusive 16:00 session) ---------------------------
tr0, T0 = prep(("09:30", "16:00")); m = grid_metrics(T0)
i, j = np.unravel_index(np.argmax(m["total"]), m["total"].shape)
print(f"[validate] best total: SL {SL[i]:.0f} / TP {TP[j]:.0f}: {m['total'][i,j]:.2f} pts, maxDD {m['dd'][i,j]:.2f}, profit/DD {m['profit_dd'][i,j]:.2f}  (notebook: 30/140, 10253.00, -947.25, 10.82)")
i5, j5 = list(SL).index(50), list(TP).index(140)
print(f"[validate] SL50/TP140: {m['total'][i5,j5]:.2f} pts, maxDD {m['dd'][i5,j5]:.2f}  (notebook: 7998.50, -1770.25)")

# ---- walk-forward on the strict RTH session ----------------------------------------------
tr, T = prep(("09:30", "15:59")); yrs = tr["date"].dt.year.to_numpy()
mf = grid_metrics(T); fi, fj = np.unravel_index(np.argmax(mf["profit_dd"]), mf["profit_dd"].shape)
print(f"\nfull-sample (look-ahead) best by profit/DD on strict session: SL {SL[fi]:.0f} / TP {TP[fj]:.0f}, "
      f"{mf['total'][fi,fj]:.0f} pts vs original ORB {tr['pnl_points'].sum():.0f} pts")

rows, oos_pnl, oos_tot_pnl = [], [], []
for y in range(2020, 2026):
    tr_i, te_i = np.where(yrs < y)[0], np.where(yrs == y)[0]
    mm = grid_metrics(T[tr_i])
    a, b = np.unravel_index(np.argmax(mm["profit_dd"]), mm["profit_dd"].shape)      # select on profit/DD
    c, d = np.unravel_index(np.argmax(mm["total"]), mm["total"].shape)              # select on total points
    p1, p2 = T[te_i, a, b], T[te_i, c, d]; oos_pnl.append(p1); oos_tot_pnl.append(p2)
    rows.append(dict(year=y, sel_SL=SL[a], sel_TP=TP[b], oos_pts_profitDD=p1.sum(),
                     sel2_SL=SL[c], sel2_TP=TP[d], oos_pts_total=p2.sum(),
                     fullsample_best_pts=T[te_i, fi, fj].sum(), original_orb_pts=tr["pnl_points"].to_numpy()[te_i].sum()))
wf = pd.DataFrame(rows); wf.to_csv("results/nq_walkforward_slTP.csv", index=False)
print("\n=== NQ walk-forward SL/TP (expanding window, points per contract) ==="); print(wf.round(0).to_string(index=False))
def dd(p): e = np.cumsum(p); return (e - np.maximum.accumulate(e)).min()
base = tr[tr["date"].dt.year >= 2020]["pnl_points"].to_numpy()
for lab, p in [("WF select by profit/DD", np.concatenate(oos_pnl)), ("WF select by total points", np.concatenate(oos_tot_pnl)), ("original ORB 2020-25", base)]:
    print(f"{lab:28s} total {p.sum():8.0f} pts  maxDD {dd(p):8.0f}  profit/DD {p.sum()/abs(dd(p)):5.2f}  trades {len(p)}")

# ---- walk-forward of the opening-range (stop distance) filter ---------------------------
thr_grid = [25, 35, 50, 60, 75, 100, 150, np.inf]; rows, kept, allp = [], [], []
pnl, risk = tr["pnl_points"].to_numpy(), tr["risk"].to_numpy()
def score(p): return p.sum() / abs(dd(p)) if len(p) and dd(p) != 0 else -np.inf
for y in range(2020, 2026):
    trn, tst = yrs < y, yrs == y
    best = max(thr_grid, key=lambda th: score(pnl[trn & (risk <= th)]))
    sel = tst & (risk <= best); kept.append(pnl[sel]); allp.append(pnl[tst])
    rows.append(dict(year=y, chosen_max_range=best, trades_kept=int(sel.sum()), of=int(tst.sum()),
                     filtered_pts=pnl[sel].sum(), unfiltered_pts=pnl[tst].sum()))
fw = pd.DataFrame(rows); fw.to_csv("results/nq_walkforward_filter.csv", index=False)
print("\n=== NQ walk-forward opening-range filter (max first-bar range / stop distance) ==="); print(fw.round(0).to_string(index=False))
for lab, p in [("filtered (walk-forward)", np.concatenate(kept)), ("unfiltered", np.concatenate(allp))]:
    print(f"{lab:26s} total {p.sum():8.0f} pts  maxDD {dd(p):8.0f}  profit/DD {p.sum()/abs(dd(p)):5.2f}  trades {len(p)}")
