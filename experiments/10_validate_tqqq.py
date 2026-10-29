"""Reproduce the TQQQ headline numbers from the original notebooks."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from orb.data import load_minute_bars, split_adjust_intraday, to_5m, make_days, daily_atr
from orb.engine import run_orb
from orb.equity import size_trades, equity_summary

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/tqqq"
pd.set_option("display.width", 220)
df = split_adjust_intraday(load_minute_bars(DATA, "tqqq_*.csv", session=("09:30", "16:00")))
df5 = to_5m(df); days = make_days(df5); atr = daily_atr(df5)
print(f"session rows {len(df):,} (notebook: 686,594) | 5m rows {len(df5):,} (notebook: 138,919)")
res = {
 "ORB stop + 10R TP": run_orb(days, "orb", entry_mode="first_close", target_r=10),
 "ORB stop + EOD": run_orb(days, "orb", entry_mode="first_close"),
 "5% ATR stop + EOD": run_orb(days, "atr", atr_stop=atr["ATR"] * 0.05, entry_mode="first_close")}
rows = {k: equity_summary(size_trades(v)) for k, v in res.items()}
print(pd.DataFrame(rows).T.round(2).to_string())
print("\nnotebook: 10R  final 242,911 / 1742 trades / DD -49,052 | ORB+EOD final 297,282 / DD -56,586 | ATR final 1,736,021 / 1728 trades / DD -570,950")
