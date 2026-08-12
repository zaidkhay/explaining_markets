# data/historical

Local cache for **realized** competition events used by
`src/explaining_markets/backtest.py` and `historical.py`. Place one or more
gzip-JSONL (or plain JSONL) files here, shaped like the competition's
`/archive` endpoint:

```
EARNINGS_RELEASE_2025Q3.jsonl.gz
EARNINGS_RELEASE_2025Q4.jsonl.gz
```

Each line is one realized event record, e.g.:

```json
{
  "event_id": "...",
  "event_type": "EARNINGS_RELEASE",
  "event_datetime": "2025-07-31T21:00:00Z",
  "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "AAPL"}],
  "disclosure": {"items": [{"kind": "facts", "content": ["...", "..."]}]},
  "event_returns": {"AAPL": {"car1": 0.05, "return_status": "ok"}},
  "metrics": {"earnings_surprise": {"surprise": 0.01, "surprise_status": "ok"}}
}
```

`historical.py::load_historical_events` reads every `*.jsonl*` file in this
directory and is silently a no-op (returns `[]`) if it's empty or missing —
**this directory is not required for `predict.py` to run.** It only feeds
offline backtesting/training via `backtest.py`. When it's empty, the live
prediction strategy still runs, using the deterministic/heuristic model
defined in `model.py` (see that module and `predict.py` for the fallback
chain).

`car1` and `event_returns`/`metrics.earnings_surprise` are realized,
**post-event** fields — they are loaded here strictly as labels/benchmarks
for `backtest.py`, and `features.py` is structured so they can never reach a
model as an input. See `HISTORICAL_DATA_INVESTIGATION.md` (repo root, if
present) for the full provenance and leakage discussion behind this design.

This directory is gitignored except for this README and `.gitkeep` — never
commit real archive data here.
