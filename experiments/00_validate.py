"""Step 0: reproduce the headline NQ numbers from the original notebooks."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from orb.data import load_minute_bars, to_5m, make_days, daily_atr
from orb.engine import run_orb
from orb.metrics import points_summary

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/nq"
t0 = time.time()
df = load_minute_bars(DATA, "nq_*.csv", session=("09:30", "16:00"))
df5 = to_5m(df)
days = make_days(df5)
print(f"loaded {len(df):,} 1m rows, {len(df5):,} 5m rows, {len(days)} days in {time.time()-t0:.0f}s")

t_orb = run_orb(days, "orb")
print("\nORIGINAL ORB", {k: round(v, 3) for k, v in points_summary(t_orb).items()})

atr = daily_atr(df5)
t_atr = run_orb(days, "atr", atr_stop=atr["ATR"] * 0.05)
print("ATR 5% ORB  ", {k: round(v, 3) for k, v in points_summary(t_atr).items()})
