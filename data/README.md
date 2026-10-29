# Data (not included)

Place your own Databento 1-minute OHLCV CSV exports here:

```
data/nq/nq_2019.csv ... nq_2025.csv        # NQ continuous front-month futures
data/tqqq/tqqq_2019.csv ... tqqq_2025.csv  # TQQQ (unadjusted; splits are back-adjusted in code)
```

Required columns: `ts_event` (UTC), `open`, `high`, `low`, `close`, `volume`
(extra Databento columns such as `rtype`, `publisher_id`, `symbol` are ignored).

Raw vendor data is deliberately not committed: check your data licence before redistributing it.
