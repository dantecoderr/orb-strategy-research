# Opening Range Breakout (ORB) research: NQ futures and TQQQ, 2019-2025

Python code to implement and stress-test the opening-range-breakout strategy of Zarattini & Aziz,
*Can Day Trading Really Be Profitable?* (Concretum Research / Peak Capital Trading, 2023; QQQ and TQQQ, 2016-2023), on one-minute
Databento data for NQ futures and TQQQ, originally developed for a *Data Mining and Computational Statistics* project.
The emphasis is on **how robust the results are** (concentration, costs, walk-forward selection,
position sizing), not on producing the best-looking equity curve.

## Summary

- **Implemented and stress-tested** the Zarattini-Aziz ORB on 7 years (2019-2025) of one-minute Databento data: NQ futures
  (1,799 trades) and TQQQ (1% risk, 4x cap). Signal-level statistics match the paper's (win rate 23-24%, 51/49 long/short, average
  0.13-0.16R), but the paper tests QQQ/TQQQ over 2016-2023, so returns are not like-for-like (see "Comparison with the reference paper").
  The code also reproduces my own earlier notebooks exactly on NQ and within 0.01% on TQQQ final equity.
- **The headline return is fragile.** On NQ the best 10 trades are 84% (ORB stop) to 95% (ATR stop) of total points. On TQQQ, charging
  as-traded costs (1c spread + 1c extra on stop exits) cuts CAGR from 47% to 29% (ORB stop) and from 93% to 45% (ATR stop).
- **The ATR stop's edge is mostly leverage.** At equal risk it roughly doubles NQ CAGR (81% vs 38%) and drawdown (-70% vs -36%).
- **Selection bias was found and removed.** An in-sample search over 703 SL/TP combinations (10,253 points) is rebuilt as walk-forward, and a
  regression-tree trade filter fails out of sample (negative R-squared in nearly every year); it is reported as a negative result.

*Research code, not investment advice. Past results, including simulated ones, say nothing reliable about future returns.*

## Strategy (as implemented)

- Direction = colour of the first 5-minute bar of the regular session (09:30 ET); one trade per day.
- NQ: enter at the open of bar 2, stop at the opposite extreme of bar 1 (**ORB stop**) or at
  5% of the prior 14-day ATR (**ATR stop**); exit at the stop or at the session close. One contract.
- TQQQ: enter at the close of bar 1; same two stops; positions sized to risk 1% of equity per trade
  with a 4x maximum leverage cap, $25,000 start, split-adjusted prices.
- ATR is `rolling(14).mean().shift(1)` of daily true range (no look-ahead).

## Run it

```bash
pip install -r requirements.txt
# put your Databento CSVs in data/nq and data/tqqq  (see data/README.md)
./run_all.sh              # writes every table below to results/
```

Each script prints its tables and the matching CSVs are saved in `results/`.

| Script | What it does |
|---|---|
| `experiments/00_validate.py`, `10_validate_tqqq.py` | Re-derive the original notebook numbers (sanity check of the engine) |
| `experiments/01_nq_robustness.py` | Session-end fix, concentration, bootstrap, costs, equal-risk % metrics, by-year |
| `experiments/02_nq_walkforward.py` | Walk-forward SL/TP grid and opening-range filter on NQ |
| `experiments/11_tqqq_robustness.py` | Costs inside the sizing loop, stop size in cents, leverage cap, participation, concentration |
| `experiments/12_tqqq_walkforward.py` | Walk-forward ATR multiplier (several objectives, with/without costs); regression tree with raw vs normalised features |
| `experiments/13_tqqq_raw_cost.py` | TQQQ costs in **as-traded** cents (converted to split-adjusted terms per date); preferred cost test |
| `experiments/14_paper_spec_comparison.py` | The reference paper's exact rules on TQQQ and NQ, next to the paper's reported numbers |

Library code is in `orb/` (`data.py`, `engine.py`, `equity.py`, `grid.py`, `metrics.py`).

## Comparison with the reference paper

The paper uses QQQ and TQQQ (Interactive Brokers data), 1 Jan 2016 to 17 Feb 2023, enters at the **open of the second 5-minute candle**,
stops at the first candle's low/high, exits at 10R or the close, risks 1% with a 4x leverage cap from $25,000, charges $0.0005 per
share and assumes no slippage. This repo has TQQQ and NQ from 2019 only, so the comparable window is 2019 - 17 Feb 2023 and the
periods differ by three years (2016-2018). `experiments/14_paper_spec_comparison.py` runs the paper's rules on the data here.

| | Paper (2016 - Feb 2023) | This repo, 2019 - Feb 2023 | This repo, 2019 - 2025 |
|---|---|---|---|
| Signal check, NQ (paper: QQQ): win rate | 24% | 23.4% | 23.7% |
| Long share of trades | 51% | - | 50.8% |
| Average R per trade (before costs) | 0.13 | 0.128 | 0.161 |
| TQQQ, ORB stop + 10R/EoD: total return | 1,484% | 259% | 746% |
| TQQQ, ORB stop + 10R/EoD: yearly return / Sharpe / max DD | 48% / 1.19 / 28% | 36% / 1.01 / -35% | 36% / 0.99 / -35% |
| TQQQ, 5% ATR stop + EoD: total return | 9,350% (text) | 1,091% | 4,340% |
| TQQQ, 5% ATR stop + EoD: yearly return / max DD | 93% alpha (net of commission) | 84% / -48% | 73% / -53% |
| Days or trades where the 4x cap binds (TQQQ ORB / ATR) | about 10% of days | 5% / 45% | 6% / 57% |

- The trade-level behaviour matches (low win rate, long/short balance, average R). Returns here are lower than the paper's for the ORB
  stop, which is plausible given the missing 2016-2018, a different data vendor and no QQQ series, but this repo cannot separate those causes.
  The 5% ATR stop gives a similar yearly return to the paper's 93% (84% for 2019 - Feb 2023) with a -48% drawdown.
- The paper's own text is not internally consistent on the ATR result: it reports +9,350% but also an account of $6,400,000 from $25,000
  (about +25,500%). This repo does not try to resolve that.
- The paper charges only $0.0005 per share and assumes no slippage, and notes itself that a stop of about $0.08 would likely be exceeded
  with large positions. The cost experiments in this repo (1 cent spread plus 1 cent extra on stop exits, as traded) are an attempt to price that caveat:
  they cut TQQQ CAGR from 47% to 29% (ORB stop) and from 93% to 45% (ATR stop) on the 2019-2025 sample.

## Validation against the original notebooks

| Quantity | Original notebook | This repo |
|---|---|---|
| NQ original ORB: trades / total points / max DD | 1,799 / 6,980.5 / -1,741 | identical |
| NQ ATR-stop ORB: trades / total points / max DD | 1,785 / 4,427.01 / -1,135.67 | identical |
| NQ best SL/TP on full sample | 30/140: 10,253 pts, DD -947.25 | identical |
| TQQQ final equity (ORB+EOD / 10R / ATR) | 297,282 / 242,911 / 1,736,021 | 297,281 / 242,925 / 1,735,926 |
| Walk-forward ATR multiplier: chosen values, final equity | 0.02, 0.08, 0.02, 0.10, 0.06, 0.11; 458,399 | identical choices; 458,319 |

Known differences (not resolved):
- Average share count is about 5.6% lower than the notebook's printed table for every TQQQ strategy
  while equity, drawdown, leverage and risk per trade match. That pattern fits a constant price-scale
  difference in split adjustment, which leaves dollar results unchanged. Not verified.
- The walk-forward regression tree (raw features, depth 4) reproduces the notebook's out-of-sample R-squared for
  2023 and 2025 but not exactly for the other years, and its filtered equity differs (about 217-235k here
  vs 174k in the notebook). Small tree differences move this number a lot, which is itself a sign of
  instability. Conclusions below do not depend on it.

## Findings

All results are before costs unless stated. "Strict" means the session ends at 15:59, so the exit is the
close of the last regular-session minute. See limitations for what OHLCV data cannot tell us.

### 1. The original session filter includes the 16:00 minute bar
`between_time("09:30", "16:00")` is inclusive, so end-of-day exits used the close of the 16:00 minute
(after the regular close). Using 15:59 moves NQ ORB from 6,980 to **7,764 points**, NQ ATR from 4,427 to 4,315,
and TQQQ ORB / ATR final equity from 297k / 1.74M to 373k / 2.40M. All numbers below use the strict session.

### 2. NQ futures (1 contract, strict session)

| | ORB stop | 5% ATR stop |
|---|---|---|
| Trades | 1,799 | 1,785 |
| Total points | 7,764 | 4,315 |
| Max drawdown (points) | -1,743 | -1,321 |
| Profit / drawdown | 4.45 | 3.27 |
| Win rate | 23.7% | 10.1% |
| Annualised Sharpe (points, 1 contract) | 0.68 | 0.69 |
| Total points after removing best 10 trades | 1,210 | 213 |
| Total points after removing best 20 trades | -2,660 | -2,595 |
| Bootstrap P(total <= 0), 10,000 resamples | 3.2% | 2.9% |
| Break-even round-trip cost | about $86 | about $48 |
| Net points at $20 round trip (Sharpe) | 5,966 (0.52) | 2,530 (0.41) |
| Net points at $40 round trip (Sharpe) | 4,167 (0.37) | 745 (0.12) |
| Equal-risk, 1% per trade: CAGR / max DD / Calmar / Sharpe | 38.3% / -36.3% / 1.06 / 0.93 | 81.4% / -70.1% / 1.16 / 1.11 |
| Average notional leverage implied by 1% risk | about 6x | about 13x |

- Profit is concentrated: the best 10 of about 1,800 trades are essentially all of the total; the best 20 are more than all of it.
- The ATR stop earns fewer points per contract but more per unit of risk. At equal risk it roughly doubles CAGR and drawdown
  with similar Calmar, at about twice the implied leverage. Tight stops make it much more sensitive to costs.
- Walk-forward SL/TP selection (expanding window, test years 2020-2025):

| | Total points | Max DD | Profit / DD |
|---|---|---|---|
| Original ORB, same years | 7,241 | -1,743 | 4.15 |
| Walk-forward, selected by profit/DD | 6,965 | -1,168 | 5.96 |
| Walk-forward, selected by total points | 5,249 | -2,567 | 2.04 |
| Walk-forward opening-range filter | 5,676 | -1,114 | 5.10 |
| In-sample best 30/140 (look-ahead, not performance) | 10,415 | | |

Walk-forward selection does not beat the baseline in points. Selecting on profit/DD reduces drawdown; selecting on total points is worse.
The opening-range filter threshold is unstable from year to year (25, 35, 35, 50, 35, 75).

### 3. TQQQ (1% risk, 4x cap, $25k start, strict session, 2019-2025)

| | ORB stop + EOD | 5% ATR stop + EOD |
|---|---|---|
| Final equity | $373k | $2.40M |
| CAGR / max DD / Calmar / Sharpe | 47.1% / -35.0% / 1.35 / 1.12 | 93.1% / -54.4% / 1.71 / 1.35 |
| Median stop distance, as traded | 22-87 cents by year | 8-32 cents by year |
| Average profit per share | 2.61 cents | 1.79 cents |
| Trades limited by the 4x leverage cap | 7% | 57% |
| CAGR, 1c round trip + 1c extra on stop exits (as traded; max DD) | 28.9% (-41.4%) | 44.5% (-70.6%) |
| CAGR, 2c round trip + 1c extra on stop exits (as traded) | 19.8% | 24.2% |
| CAGR after removing the best 20 trades | 1.6% | 13.3% |
| Median position vs first 5-minute bar volume | 0.4% | 4.8% (49% of trades above 5%) |

- Average profit per share is a couple of split-adjusted cents, but costs must be charged in as-traded cents: an as-traded cent is
  about an eighth of an adjusted cent in 2019-2020 (three 2:1 splits). `experiments/13` does this; `experiments/11` charges a flat
  cent per adjusted share, which overstates early-year costs and is kept only for reference. Costs are assumptions, because OHLCV
  data contains no quotes; market-impact cost of large positions is not modelled.
- For the ATR stop the 1% risk rule is not the binding constraint on most trades; the 4x cap is. Much of its compounded growth
  comes from 2019-2020 (calendar-year returns of +242% and +172% before costs).
- Participation figures reflect the compounded account size; a small account would not hit them.

Walk-forward ATR multiplier (2-30% grid, expanding window, test years 2020-2025, compounded from $25k):

| Selection | Final equity | CAGR | Max DD | Calmar | CAGR, flat 1 adjusted cent |
|---|---|---|---|---|---|
| Fixed 5% | $701k | 74.4% | -54.4% | 1.4 | 19.8% |
| Notebook objective (dollar profit/DD) | $613k | 70.5% | -46.9% | 1.5 | 17.0% |
| Calmar | $1.04M | 86.4% | -40.8% | 2.1 | 28.7% |
| Sharpe | $924k | 82.6% | -42.5% | 1.9 | 18.8% |
| Calmar, selected net of 1c costs | $540k | 66.9% | -40.8% | 1.6 | 24.4% |

Selecting on a percentage-drawdown objective improves out-of-sample Calmar versus the fixed 5% stop. The last column
uses the flat adjusted-cent cost from `experiments/12`, which is too punitive for the early test years, so read it as a lower bound. Several objectives were tried after seeing
results, so treat the best row as optimistic.

Regression tree filter (predicts R from entry-time features; yearly walk-forward):
- Out-of-sample R-squared is negative in nearly every year for every variant (mean -0.05 raw, -0.03 / -0.02 normalised).
- Rank correlation between predicted and realised R is small: about 0.03-0.07 on average, positive in 5 of 6 years, which is
  within noise for about 250 trades per year.
- Normalising features (gap %, range / ATR, volume vs 20-day average) did not rescue it. The filter lowers return; its drawdown and Calmar
  effects vary by variant (compounded 2020-2025: unfiltered $701k vs $410k-$669k filtered).

## Limitations

- OHLCV only: no bid/ask, no queue position, no short-borrow data. Costs are scenario assumptions, not measurements.
- Stops fill exactly at the stop price; real stop fills at the open are worse, especially with stops of a few cents.
- Five-minute bars decide the stop in the baseline; the SL/TP grid uses one-minute paths and assumes the stop fills first if both levels trade in the same minute.
- The NQ series is Databento's unadjusted volume-roll continuous contract; ATR on roll days includes the calendar spread.
- The bootstrap resamples trades independently and ignores regime clustering.
- Several analysis variants were chosen after seeing results (objectives, thresholds, cost grids). All are reported.

## License

MIT, see `LICENSE`.
