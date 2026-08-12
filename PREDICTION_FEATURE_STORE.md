# Prediction-Time Feature Store

**Status:** Infrastructure only. `predict.py` and `model.py` are unchanged —
this document describes `src/explaining_markets/feature_store.py`, a new,
standalone module that builds the `SAFE_IF_TIMESTAMPED` historical features
specified in `docs/PREDICTION_TIME_INFORMATION_AUDIT.md` §5, with enforced
walk-forward safety and per-feature provenance. Nothing here is wired into
the live prediction path yet.

---

## 1. Purpose and scope

`docs/PREDICTION_TIME_INFORMATION_AUDIT.md` classified eight rolling/
aggregate, same-ticker historical features as `SAFE_IF_TIMESTAMPED`:
previous CAR1, rolling mean/volatility of CAR1, previous earnings surprise,
rolling mean surprise, count of previous positive surprises, historical
reaction asymmetry, and count of prior earnings events. "Conditionally
safe" in that audit meant: safe **only if** every value used to build them
comes from an event strictly earlier than the one being predicted.

This module is the infrastructure that makes that condition **structural,
not aspirational** — every feature it produces is accompanied by proof of
which source event(s) it came from and why they were eligible, and that
proof is checked, not just asserted, at build time.

It does **not**:
- modify `predict.py`, `model.py`, or any other production file,
- decide how these features should be combined with the existing
  disclosure-based features in `features.py`,
- optimize or retrain anything.

Those are deliberately left for a later step.

---

## 2. The governing rule

> For a target event `E` (ticker `t`, `event_datetime = T`), a feature may
> only be built from events for the **same ticker** with `event_datetime`
> **strictly before** `T`. Never `E` itself. Never an event at or after `T`.

This single rule is enforced at three independent points in
`feature_store.py`, deliberately redundant:

1. **Structurally** — `eligible_prior_events()` is the *only* function that
   selects candidate source events, and it filters on this rule by parsing
   both timestamps to `datetime` objects (never comparing raw strings, so
   differing-but-equivalent ISO-8601 formatting can't cause a false pass).
2. **At runtime** — `assert_no_target_leakage()` re-checks the same
   invariant on whatever `eligible_prior_events()` actually returned,
   immediately before any feature computation touches it. If a future
   change to the selection logic reintroduces leakage, this raises
   `ValueError` rather than silently producing a bad feature.
3. **On the output** — every `HistoricalFeatures` instance carries a
   `provenance: dict[str, tuple[ProvenanceRecord, ...]]`, one entry per
   feature name, naming the exact source event(s) and a human-readable rule
   string. This makes leakage auditable *after* the fact, not just prevented
   at construction — see §5.

---

## 3. Data flow

```
data/historical/*.jsonl.gz  (6,287 sealed-quarter records)
        │
        ▼
historical.load_historical_events()          [unchanged, existing module]
        │  list[HistoricalEvent]  (event_id, ticker, event_type,
        │                          event_datetime, disclosure, car1,
        │                          earnings_surprise, quarter)
        ▼
feature_store.build_ticker_timelines(events)
        │  dict[ticker -> chronologically sorted list[HistoricalEvent]]
        │  (this is the walk-forward backbone: a per-ticker timeline)
        ▼
feature_store.build_feature_store(events)
        │  for each ticker's timeline, for each event E acting as a target:
        │    1. eligible_prior_events(timeline, E)         -> strictly-earlier events
        │    2. assert_no_target_leakage(E, prior)          -> re-check, raise if violated
        │    3. compute_historical_features(E, timeline)    -> HistoricalFeatures
        ▼
list[HistoricalFeatures]   (one row per input event; 6,287 rows for the full archive)
        │  each row: 8 numeric fields + provenance dict
        ▼
   (not yet wired anywhere - future step: combine with features.FeatureVector,
    feed into model.py, or use as backtest.py training input)
```

Note that `build_feature_store` is given the **full, multi-quarter** event
list (all 6,287 records across 2025Q4/2026Q1/2026Q2) in one call — this is
intentional. Quarter boundaries are irrelevant to this module; only
per-ticker chronology matters, so a target event in 2026Q1 correctly sees
realized outcomes from that same ticker's 2025Q4 event, if any.

---

## 4. Schema

### `ProvenanceRecord` (frozen dataclass)

| Field | Type | Meaning |
|---|---|---|
| `feature_name` | `str` | Which feature this record justifies (e.g. `"previous_car1"`) |
| `source_event_id` | `str` | The prior event actually used |
| `source_ticker` | `str` | The prior event's ticker (always equal to the target's — see §6 tests) |
| `source_event_datetime` | `str` | The prior event's own timestamp |
| `target_event_id` | `str` | The event this feature was built *for* |
| `target_event_datetime` | `str` | That target's timestamp |
| `rule` | `str` | Human-readable justification, e.g. `"source_event_datetime(...) < target_event_datetime(...); most recent prior event with car1 realized"` |

### `HistoricalFeatures` (frozen dataclass)

| Field | Type | Minimum-history rule (else `None`/`0`) |
|---|---|---|
| `ticker` | `str` | — |
| `target_event_id` | `str` | — |
| `target_event_datetime` | `str` | — |
| `previous_car1` | `float \| None` | >= 1 prior event with `car1` realized |
| `rolling_mean_car1` | `float \| None` | >= 1 prior event with `car1` realized, within trailing `window` |
| `rolling_car1_volatility` | `float \| None` | >= 2 prior events with `car1` realized, within trailing `window` (`stdev` undefined below n=2) |
| `previous_earnings_surprise` | `float \| None` | >= 1 prior event with `earnings_surprise` realized |
| `rolling_mean_surprise` | `float \| None` | >= 1 prior event with `earnings_surprise` realized, within trailing `window` |
| `number_of_previous_positive_surprises` | `int` | Always defined (0 if no qualifying history) |
| `historical_reaction_asymmetry` | `float \| None` | >= 1 prior event with `car1 > 0` **and** >= 1 with `car1 < 0`, over the **full**, unwindowed history (per the audit's literal formula — the one feature in this family that ignores `window`) |
| `number_of_prior_earnings_events` | `int` | Counts **all** prior events regardless of whether `car1`/`earnings_surprise` is known — existence only, matching the audit's finding that this one feature has a materially better availability profile than the outcome-bearing ones |
| `provenance` | `dict[str, tuple[ProvenanceRecord, ...]]` | One key per feature that was actually computed (a feature with insufficient history has no key here at all, not an empty tuple) |

Two accessor methods:
- `feature_values()` — the 8 numeric fields only, model-ready, guaranteed to
  contain no key in `features.FORBIDDEN_KEYS` (checked by
  `assert_feature_is_leakage_free`).
- `as_dict()` — `feature_values()` plus `ticker`, for joining/inspection.
  Deliberately excludes `provenance` and the target identity fields (not
  signal about the company).

**Window size:** `DEFAULT_WINDOW = 4` (a module constant), used by every
windowed feature except `historical_reaction_asymmetry`. A single documented
constant rather than a per-feature magic number, overridable per call via
`window=` on `build_feature_store`/`compute_historical_features`.

---

## 5. What is explicitly prevented from ever entering a feature

Per the task's requirement, `CAR1`, `earnings_surprise`, `event_returns`,
and `baseline_predictions` can **never** reach a feature value, for two
independent reasons:

1. **No code path reads them from the target.** Every function in
   `feature_store.py` that takes a `target: HistoricalEvent` argument only
   ever reads `target.ticker` / `target.event_id` / `target.event_datetime`
   from it. There is no line of code anywhere in the module that reads
   `target.car1`, `target.earnings_surprise`, or any other outcome field off
   the event being predicted. (`HistoricalEvent` itself, from `historical.py`,
   doesn't even model `event_returns`/`baseline_predictions` as separate
   fields — only the already-extracted `car1`/`earnings_surprise` scalars
   are carried, and even those are read only off *source* events, never the
   target.)
2. **A prior event's own `car1`/`earnings_surprise` can only reach a
   feature if it is on the `eligible_prior_events()` allow-list** — which is
   re-verified by `assert_no_target_leakage()` immediately before use. There
   is no code path that reads a source event's fields before that check has
   run for it.
3. **`assert_feature_is_leakage_free()`** is a final, redundant check on the
   *output* — it inspects `feature_values()` for any key in
   `features.FORBIDDEN_KEYS` (`car1`, `earnings_surprise`, `surprise`,
   `predicted_percentile`, `y`). This should never trigger given (1) and
   (2), but costs nothing to keep as a guard against a future field being
   added to `HistoricalFeatures` without updating this discipline.

---

## 6. Automated leakage assertions and tests

`tests/test_feature_store.py` (29 tests) covers:

- **Walk-forward mechanics:** `build_ticker_timelines` groups and sorts
  correctly, including events with missing/unparseable timestamps (excluded,
  never guessed); `eligible_prior_events` excludes the target itself,
  excludes equal-timestamp "prior" events, and skips unparseable candidates.
- **`assert_no_target_leakage` as an active check**, not just a passive
  filter: explicit tests that it *raises* when a source event is the target
  itself, is later, is at the same timestamp, or has no parseable timestamp
  — and a positive test that it does not raise for genuinely valid input.
- **Per-feature minimum-history rules**, each with an exact expected value:
  first-event-in-a-timeline has no history at all; `previous_car1` picks the
  single most recent *qualifying* prior event (skipping ones without a
  realized `car1`); rolling mean/volatility respect the trailing `window`
  exactly (verified against hand-computed expected values); volatility
  requires n>=2; `historical_reaction_asymmetry` requires both a positive
  and a negative observation and is verified against a hand-computed value;
  `number_of_prior_earnings_events` is verified to count existence even when
  `car1` is `None`, distinguishing it from every other feature in the table.
- **Provenance invariants:** every `ProvenanceRecord` produced in a
  multi-feature scenario is checked to have a strictly-earlier
  `source_event_datetime` than the target's, a `source_event_id` different
  from the target's, and the correct `target_event_id` back-reference.
- **`feature_values()`/`as_dict()` never contain a forbidden key.**
- **Cross-ticker isolation:** a constructed scenario where a different
  ticker's earlier event could *chronologically* qualify is checked to
  never appear in another ticker's provenance.
- **Cross-quarter continuity:** a target in one quarter is confirmed to see
  a realized prior event from an *earlier* quarter for the same ticker.
- **Full real-archive integration sweep** (`test_build_feature_store_on_real_archive_has_no_leakage`):
  runs `build_feature_store` over all 6,287 real, downloaded records and
  checks **every single provenance record produced** (not a sample) for the
  three leakage conditions — self-reference, non-strictly-earlier timestamp,
  cross-ticker mismatch — asserting **zero violations** across the real
  dataset. This test gracefully skips (rather than fails) in an environment
  where `data/historical/` is empty, so the suite remains runnable without
  the real archive present.

All 100 tests in the full suite pass (71 pre-existing + 29 new), including
this real-data sweep.

---

## 7. Example output (real data)

Ticker `APOG`, which has 4 events across the downloaded quarters, walk-forward
built via `build_feature_store(load_historical_events())`:

```
event_datetime            prior_n  previous_car1     rolling_mean_car1   previous_surprise
2025-10-09T20:00:00+00:00      0   None              None                None
2026-01-07T11:03:00+00:00      1   -0.01760664934...  -0.01760664934...   0.00313706563...
2026-04-24T10:03:00+00:00      2   -0.13551157036...  -0.07655910985...   0.0
2026-06-26T10:30:00+00:00      3    0.06943175974...  -0.02789548665...   0.00140488901...
```

The first event correctly has no history (`prior_n=0`, all fields `None`).
Each subsequent event's `prior_n` increments by exactly one, and
`rolling_mean_car1` is visibly the running mean of the *previous* rows'
`previous_car1`-style values — never including the row's own future outcome.

---

## 8. What is intentionally left for later

- **Combining these features with `features.FeatureVector`** (the existing
  disclosure-based features) into one model-ready row. Not done here —
  `feature_store.py` has no dependency on, and no coupling to, `model.py`.
- **Wiring any of this into `predict.py`.** The live path is completely
  untouched; these features are not available to a live prediction today.
  Per the audit (§4/§6 of `PREDICTION_TIME_INFORMATION_AUDIT.md`), doing so
  live would additionally require solving the archive's sealing-lag problem
  (our own `/archive` download lags real time by ~6 weeks), which this
  module does not attempt to solve — it operates only on whatever
  `HistoricalEvent`s it is given.
- **Model training or evaluation using these features.** `backtest.py` is
  unchanged; using `HistoricalFeatures` there is a natural next step but was
  explicitly out of scope for this task.
