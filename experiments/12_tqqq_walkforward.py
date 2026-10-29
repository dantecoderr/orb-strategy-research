"""TQQQ walk-forward: (a) ATR-multiplier selection under different objectives and costs, (b) regression tree with raw vs normalised features."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from orb.data import load_minute_bars, split_adjust_intraday, to_5m, make_days, daily_atr
from orb.engine import run_orb
from orb.equity import size_trades, equity_summary

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/tqqq"
pd.set_option("display.width", 220)
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MULTS = np.round(np.arange(0.02, 0.3001, 0.01), 2)

def prep(session):
    df = split_adjust_intraday(load_minute_bars(DATA, "tqqq_*.csv", session=session))
    df5 = to_5m(df); return make_days(df5), daily_atr(df5)

def mult_tables(days, atr):
    return {m: run_orb(days, "atr", atr_stop=atr["ATR"] * m, entry_mode="first_close") for m in MULTS}

def nb_profit_dd(x):   # notebook definition (peak excludes the starting equity)
    return x["pnl_dollars"].sum() / abs((x["equity_after"] - x["equity_after"].cummax()).min())

def score(x, how):
    if x.empty: return -np.inf
    if how == "profit_dd_notebook": return nb_profit_dd(x)
    if how == "calmar":
        return np.log(x["equity_after"].iloc[-1] / 25_000) / abs(x["dd_pct"].min()) if x["equity_after"].iloc[-1] > 0 else -np.inf
    if how == "sharpe":
        r = (x["equity_after"] / x["equity_before"] - 1).to_numpy(); return r.mean() / r.std(ddof=1)

def walkforward_mult(tables, how, cost=0.0):
    sel, chosen = [], {}
    for y in YEARS:
        best = max(MULTS, key=lambda m: score(size_trades(tables[m][tables[m]["date"].dt.year < y], cost_per_share=cost), how))
        chosen[y] = best; t = tables[best]; sel.append(t[t["date"].dt.year == y])
    return chosen, pd.concat(sel).sort_values("date").reset_index(drop=True)

# ---------------- (a) ATR multiplier ---------------------------------------------------------------
days, atr = prep(("09:30", "16:00")); tab = mult_tables(days, atr)
ch, wf = walkforward_mult(tab, "profit_dd_notebook")
x = size_trades(wf)
print("=== [validate] notebook objective, original session ===")
print("chosen:", ch, "\nfinal equity %.0f over %d trades (notebook: 458,399 / 1,492; chosen 0.02,0.08,0.02,0.10,0.06,0.11)" % (x["equity_after"].iloc[-1], len(x)))

days, atr = prep(("09:30", "15:59")); tab = mult_tables(days, atr)
rows = []; fixed5 = tab[0.05]
t_idx = lambda t: t[t["date"].dt.year >= 2020].reset_index(drop=True)
for lab, t in [("fixed 5% ATR (2020-25 only)", t_idx(fixed5))]:
    for c in [0, 0.01]:
        s = equity_summary(size_trades(t, cost_per_share=c)); rows.append(dict(method=lab, select_cost_c=np.nan, test_cost_c=c * 100, chosen="-", **{k: s[k] for k in ["final_equity", "cagr_pct", "max_dd_pct", "calmar", "sharpe_ann", "avg_leverage"]}))
for how, sc in [("profit_dd_notebook", 0.0), ("calmar", 0.0), ("sharpe", 0.0), ("calmar", 0.01)]:
    ch, wf = walkforward_mult(tab, how, cost=sc)
    for c in [0, 0.01]:
        s = equity_summary(size_trades(wf, cost_per_share=c))
        rows.append(dict(method=f"WF by {how}", select_cost_c=sc * 100, test_cost_c=c * 100, chosen="/".join(f"{v:.2f}" for v in ch.values()),
                         **{k: s[k] for k in ["final_equity", "cagr_pct", "max_dd_pct", "calmar", "sharpe_ann", "avg_leverage"]}))
res = pd.DataFrame(rows); res.to_csv("results/tqqq_wf_multiplier.csv", index=False)
print("\n=== ATR-multiplier walk-forward, strict session, test years 2020-25 (compounded from $25k) ===")
print(res.round(1).to_string(index=False))

# ---------------- (b) regression tree ---------------------------------------------------------------
def feature_table(days, atr, session_label):
    t = run_orb(days, "atr", atr_stop=atr["ATR"] * 0.05, entry_mode="first_close")
    d = atr.copy(); d["prev_open"] = d["open"].shift(1); d["prev_high"] = d["high"].shift(1); d["prev_low"] = d["low"].shift(1)
    d["prev_vol"] = d["volume"].shift(1)
    d["prev_day_range"] = (d["high"] - d["low"]).shift(1); d["prev_day_return"] = (d["close"] - d["open"]).shift(1)
    d["prev_close_position"] = ((d["close"] - d["low"]) / (d["high"] - d["low"])).shift(1)
    fv = pd.Series({k: v["v"][0] for k, v in days.items()}); d["fv20"] = fv.rolling(20).mean().shift(1)
    t = t.join(d[["prev_close", "prev_open", "prev_vol", "prev_day_range", "prev_day_return", "prev_close_position", "ATR", "fv20"]], on="date")
    t["year"] = t["date"].dt.year; t["weekday"] = t["date"].dt.weekday; t["month"] = t["date"].dt.month
    t["direction_num"] = t["direction"]; t["body_to_range"] = np.where(t["first_range"] != 0, t["first_body"] / t["first_range"], 0)
    # raw (notebook) features, in dollars / shares
    t["gap_from_prev_close"] = t["entry"] - t["prev_close"]
    # normalised features
    t["gap_pct"] = t["gap_from_prev_close"] / t["prev_close"]; t["range_atr"] = t["first_range"] / t["ATR"]; t["body_atr"] = t["first_body"] / t["ATR"]
    t["vol_rel"] = t["first_volume"] / t["fv20"]; t["prev_ret_pct"] = t["prev_day_return"] / t["prev_open"]
    t["prev_range_atr"] = t["prev_day_range"] / t["ATR"]; t["stop_pct"] = t["risk"] / t["entry"]
    return t.dropna().sort_values("date").reset_index(drop=True)

RAW = ["risk", "ATR", "first_range", "first_body", "body_to_range", "first_volume", "gap_from_prev_close", "prev_day_range",
       "prev_day_return", "prev_vol", "prev_close_position", "direction_num", "weekday", "month"]
NORM = ["gap_pct", "range_atr", "body_atr", "body_to_range", "vol_rel", "prev_ret_pct", "prev_range_atr", "prev_close_position", "stop_pct", "direction_num", "weekday"]

def tree_wf(t, feats, depth, leaf, clip):
    sel, rows = [], []
    for y in YEARS:
        tr, te = t[t["year"] < y], t[t["year"] == y]
        ytr = tr["R"].clip(upper=tr["R"].quantile(0.99)) if clip else tr["R"]
        m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=42).fit(tr[feats], ytr)
        p = m.predict(te[feats]); te = te.assign(pred=p); sel.append(te[te["pred"] > 0])
        rows.append(dict(year=y, test_R2=r2_score(te["R"], p), rank_IC=spearmanr(p, te["R"])[0], kept=int((p > 0).sum()), of=len(te),
                         kept_avgR=te.loc[te["pred"] > 0, "R"].mean(), dropped_avgR=te.loc[te["pred"] <= 0, "R"].mean()))
    return pd.DataFrame(rows), pd.concat(sel).sort_values("date").reset_index(drop=True)

d0, a0 = prep(("09:30", "16:00")); t0 = feature_table(d0, a0, "orig")
r0, s0 = tree_wf(t0, RAW, 4, 40, False); x0 = size_trades(s0)
print("\n=== [validate] raw-feature tree, original session ===")
print(r0[["year", "test_R2"]].round(4).T.to_string(header=False)); print("tree-filtered final equity %.0f, trades %d (notebook: 173,662 / 679; test R2 -0.062,-0.030,-0.056,-0.002,-0.139,-0.075)" % (x0["equity_after"].iloc[-1], len(x0)))

t1 = feature_table(days, atr, "strict")
base = t1[t1["year"] >= 2020].reset_index(drop=True)
rows = [dict(variant="unfiltered ATR (2020-25)", **{k: equity_summary(size_trades(base))[k] for k in ["trades", "final_equity", "cagr_pct", "max_dd_pct", "calmar"]})]
for lab, feats, depth, leaf, clip in [("raw features (notebook)", RAW, 4, 40, False), ("raw + clipped R", RAW, 4, 40, True),
                                      ("normalised, depth 3", NORM, 3, 60, True), ("normalised, depth 2", NORM, 2, 100, True)]:
    r, s = tree_wf(t1, feats, depth, leaf, clip); sm = equity_summary(size_trades(s))
    print(f"\n--- {lab}: mean test R2 {r['test_R2'].mean():.3f} | mean rank IC {r['rank_IC'].mean():.3f} | years with IC>0: {(r['rank_IC']>0).sum()}/6 ---")
    print(r.round(3).to_string(index=False))
    rows.append(dict(variant="tree: " + lab, **{k: sm[k] for k in ["trades", "final_equity", "cagr_pct", "max_dd_pct", "calmar"]}))
print("\n=== Tree-filtered vs unfiltered (uncosted, strict session, 2020-25) ==="); print(pd.DataFrame(rows).round(1).to_string(index=False))
