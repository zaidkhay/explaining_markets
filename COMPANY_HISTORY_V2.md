# Company-History V2 Report

## Summary

This report documents the design, training, evaluation, and deployment status of
`fls_company_history_ridge_v2`, a Ridge regression model that augments the
existing forward-looking-disclosure (FLS) features with company-specific
historical features sourced from the sealed Explaining Markets archive.

**Result: V2 was NOT promoted.** The predeclared promotion gate failed — V2
showed zero validation improvement over V1 and regressed materially on the
locked holdout. The production model remains `fls_ridge_v1`.

---

## 1. Files Created and Modified

### New source files

| File | Purpose |
|------|---------|
| `src/explaining_markets/data_providers/__init__.py` | Vendor-agnostic provider package |
| `src/explaining_markets/data_providers/records.py` | `PriceBar`, `EarningsRecord` with point-in-time provenance |
| `src/explaining_markets/data_providers/protocols.py` | `MarketDataProvider`, `EarningsDataProvider` Protocols |
| `src/explaining_markets/data_providers/cache.py` | SQLite cache implementing both Protocols |
| `src/explaining_markets/data_providers/fixtures.py` | `InMemoryProvider` for tests |
| `src/explaining_markets/company_history.py` | Company-history feature layer (price, earnings, surprise, kNN, recency) |
| `src/explaining_markets/competition_history.py` | Archive-sourced walk-forward history + live snapshot provider |
| `src/explaining_markets/features_v2.py` | V2 feature specification (`MODEL_FEATURE_NAMES_V2`, vector assembly) |
| `src/explaining_markets/leakage_audit.py` | Point-in-time leakage audit |
| `src/explaining_markets/v2_training.py` | Chronological training, ablations, promotion gate |

### New artifacts

| File | Size | Description |
|------|------|-------------|
| `src/explaining_markets/artifacts/fls_company_history_ridge_v2.json` | 34 KB | V2 model artifact (promoted: false) |
| `src/explaining_markets/artifacts/company_history_snapshot_v1.json` | 950 KB | Per-ticker archive snapshot for live inference |

### New test files

| File | Tests |
|------|-------|
| `tests/test_data_providers.py` | 11 |
| `tests/test_company_history.py` | 17 |
| `tests/test_competition_history.py` | 13 |
| `tests/test_features_v2.py` | 7 |
| `tests/test_leakage_audit.py` | 8 |
| `tests/test_v2_training.py` | 5 |
| `tests/test_model_v2.py` | 8 |
| `tests/test_predict_v2.py` | 7 |

### Modified files

| File | Change |
|------|--------|
| `src/explaining_markets/model.py` | Added `CompanyHistoryRidgeModel`, V2-aware `get_default_model()` |
| `predict.py` | V2-first fallback chain with history diagnostics logging |
| `modal_app.py` | `add_local_python_source(..., ignore=[])` to mount JSON artifacts (pre-existing) |
| `.gitignore` | Accommodate new cache/artifact layout |

---

## 2. Data-Provider / Cache Architecture

```
OFFLINE (research / cache building)
    data_providers.records.PriceBar / EarningsRecord
        ↑ normalized by
    data_providers.protocols.{MarketDataProvider, EarningsDataProvider}
        ↑ implemented by
    data_providers.cache.CompanyHistoryCache (SQLite)
    data_providers.fixtures.InMemoryProvider (tests)

TRAINING (offline, sealed archive only)
    competition_history.walk_forward_history()
        → company_history.compute_company_history_features()
        → features_v2.build_feature_vector_v2()

LIVE (Modal worker)
    competition_history.SnapshotCompanyHistoryProvider
        reads artifacts/company_history_snapshot_v1.json
        → company_history.compute_company_history_features()
        → features_v2.build_feature_vector_v2()
        → model.CompanyHistoryRidgeModel.predict_vector()
```

Every record carries `value_timestamp`, `available_at`, `retrieved_at`, and
`source`. A record is usable at cutoff `T` only when `available_at < T`
(strict). Unknown availability fails closed (excluded).

No vendor implementation ships yet — no market-data credentials are configured.
The interfaces, SQLite schema, and point-in-time logic are complete and
fixture-tested so a vendor (Polygon, Alpha Vantage, FMP, Tiingo) can be added
by implementing the two Protocols without touching the model layer.

---

## 3. V2 Feature List (65 features)

### FLS block (30 features, unchanged from V1)

`fls_count`, `fls_ratio`, `earnings_fls_count`, `earnings_fls_ratio`,
`non_earnings_fls_count`, `non_earnings_fls_ratio`, `quantitative_fls_count`,
`quantitative_fls_ratio`, `non_quantitative_fls_count`,
`non_quantitative_fls_ratio`, `quant_earnings_fls_count`,
`quant_earnings_fls_ratio`, `other_fls_count`, `other_fls_ratio`,
`quant_non_earnings_ratio`, `nonquant_earnings_ratio`,
`other_to_quant_earnings_ratio`, `positive_forward_count`,
`negative_forward_count`, `signed_forward_tone`, `guidance_raised`,
`guidance_lowered`, `guidance_maintained`, `guidance_direction`,
`signed_fls_intensity`, `signed_quant_earnings_intensity`,
`signed_other_fls_intensity`, `guidance_fls_interaction`,
`guidance_quant_earnings_interaction`, `other_minus_quant_earnings`

### Price history (9 features)

`return_3m`, `return_6m`, `return_1y`, `return_3y`, `return_5y`,
`volatility_3m`, `volatility_1y`, `max_drawdown_1y`, `max_drawdown_5y`

### Earnings history (5 features)

`prior_earnings_count`, `mean_prior_earnings_abnormal_return`,
`median_prior_earnings_abnormal_return`, `std_prior_earnings_abnormal_return`,
`positive_prior_earnings_rate`

### Surprise history (6 features)

`mean_prior_eps_surprise`, `std_prior_eps_surprise`,
`positive_eps_surprise_rate`, `negative_eps_surprise_rate`,
`mean_reaction_after_positive_surprise`,
`mean_reaction_after_negative_surprise`

### Current vs history (3 features)

`current_eps_surprise`, `current_eps_surprise_zscore`,
`current_eps_surprise_percentile_company`

### Similar events (3 features)

`similar_surprise_mean_reaction`, `similar_surprise_median_reaction`,
`similar_surprise_count`

### Recency-weighted reaction (1 feature)

`recency_weighted_earnings_reaction`

**Note:** This feature was not in the original requested feature list but was
added as an intentional additional feature. It computes an exponentially
decayed weighted mean of prior abnormal returns (half-life = 365 days),
preferring more recent reactions. It is documented here as part of the
deployed V2 contract.

### Competition history (4 features)

`prior_competition_event_count`, `mean_prior_competition_car1`,
`last_prior_competition_car1`, `has_competition_history`

### Availability indicators (4 features)

`has_1y_price_history`, `has_5y_price_history`, `has_eps_surprise_history`,
`has_similar_surprise_history`

---

## 4. Point-in-Time Cutoff Rules

Every feature for a focal event at cutoff `T` uses only information available
strictly before `T`:

- **Prices:** A daily close is usable when `available_at < T`. Late
  back-adjustments must set `available_at` to the revision time.
- **Earnings reactions:** A prior event's reaction is usable only when
  `reaction_available_at < T`. If `reaction_available_at` is unknown, the
  reaction is EXCLUDED (fail closed).
- **Competition archive:** A prior event's outcome (CAR1, surprise) is treated
  as available only at `prior_event_datetime + 7 calendar days`, and must
  precede the focal event by at least 1 additional guard day:
  `prior_event_datetime + 7d < focal_event_datetime - 1d`.
- **Current surprise:** Populated only when a verified point-in-time source
  proves availability before `T`. No such source exists today, so these
  features are neutral (0.0 with indicator 0) in both training and live.

Forbidden inputs: current-event CAR1, future prices, future earnings, future
analyst estimates, current-event surprise without verified pre-deadline
availability, post-event prices.

---

## 5. Training Sample Counts

| Quarter | Role | Rows |
|---------|------|------|
| 2025Q4 | Train | 1,849 |
| 2026Q1 | Validation | 2,390 |
| 2026Q2 | Locked holdout | 2,047 |
| **Total** | **Final refit** | **6,286** |

### Row-count discrepancy (6,287 vs 6,286)

The archive contains 6,287 event/ticker rows. `labeled_events()` filters to
6,286 rows — one event has no `car1` label and is excluded from training. This
is the sole source of the discrepancy.

### History availability

- `n_rows_with_history`: 3,686 (58.7% of rows)
- `n_source_records`: 5,140
- 2025Q4: 0 rows with prior history (cold-start training quarter)
- 2026Q1: 1,722 rows with prior history
- 2026Q2: 1,964 rows with prior history

---

## 6. Selected Hyperparameters

- **Selected alpha:** 10.0 (same as V1)
- **Selection rule:** Max validation official `delta_r_squared`; Pearson fallback
- **Alpha grid:** (0.1, 1.0, 10.0, 100.0, 300.0)
- **Clip bounds:** (0.05, 0.95)

---

## 7. Validation Metrics (2026Q1)

| Variant | Alpha | delta_R² | Pearson |
|---------|-------|----------|---------|
| fls_only | 10.0 | 0.007528 | 0.099317 |
| history_only | 0.1 | 0.000000 | None |
| fls_plus_price | 10.0 | 0.007528 | 0.099317 |
| fls_plus_earnings_reaction | 10.0 | 0.007528 | 0.099317 |
| fls_plus_current_surprise | 10.0 | 0.007528 | 0.099317 |
| fls_plus_competition | 10.0 | 0.007528 | 0.099317 |
| all_v2 | 10.0 | 0.007528 | 0.099317 |

**Validation gain: 0.0** — no ablation improved over FLS-only.

---

## 8. Locked Holdout Metrics (2026Q2)

| Model | Pearson | MAE | Pred Std | delta_R² |
|-------|---------|-----|----------|----------|
| **fls_ridge_v1 (FLS-only)** | **0.166980** | **0.245178** | **0.058264** | **0.020449** |
| all_v2 (V2) | 0.029364 | 0.312183 | 0.193272 | 0.000937 |
| constant 0.5 | — | 0.250122 | 0.000000 | — |
| surprise benchmark | 0.263217 | 0.269427 | 0.288810 | — |

V2 regressed materially: holdout delta-R² fell from 0.020449 to 0.000937.

---

## 9. Official delta-R² Comparison

| Model | Validation delta-R² | Holdout delta-R² |
|-------|---------------------|------------------|
| V1 (fls_ridge_v1) | 0.007528 | 0.020449 |
| V2 (all features) | 0.007528 | 0.000937 |
| V2 gain (val) | 0.000 | — |
| V2 regression (holdout) | — | 0.019512 |

---

## 10. V1 vs V2 Comparison

| Metric | V1 | V2 |
|--------|----|----|
| Features | 30 | 65 |
| Holdout Pearson | 0.166980 | 0.029364 |
| Holdout MAE | 0.245178 | 0.312183 |
| Holdout delta-R² | 0.020449 | 0.000937 |
| Holdout pred std | 0.058264 | 0.193272 |
| Promoted | yes (existing) | **no** |

V2 has higher prediction variance (0.193 vs 0.058) but worse correlation and
delta-R². The additional features added noise without signal on the holdout.

---

## 11. Ablation Results by Feature Family

All ablations selected alpha=10.0 and produced identical validation delta-R²
(0.007528) to FLS-only, except `history_only` which was degenerate
(delta-R²=0.0, Pearson=None). No feature family added useful validation signal
beyond the FLS block.

---

## 12. Real Historical Examples

Four examples from the archive with `prior_earnings_count >= 2`:

### Example 1: CMC (2026Q1)
- Event: 2026-03-26T10:45:00Z
- Prior earnings count: 2
- Mean prior CAR1: -0.0503
- Last prior competition CAR1: -0.0359
- FLS ratio: 0.5, tone: 0.2, guidance: 0
- Actual CAR1: -0.0297, actual percentile: 0.3759
- Earnings surprise: -0.0022

### Example 2: MSM (2026Q2)
- Event: 2026-04-01T10:30:00Z
- Prior earnings count: 2
- Mean prior CAR1: -0.0088
- Last prior competition CAR1: -0.042
- FLS ratio: 0.3, tone: 0.667, guidance: 0
- Actual CAR1: -0.0154, actual percentile: 0.4477
- Earnings surprise: -0.0002

### Example 3: FC (2026Q2)
- Event: 2026-04-01T20:15:00Z
- Prior earnings count: 2
- Mean prior CAR1: -0.0423
- Last prior competition CAR1: 0.0048
- FLS ratio: 0.1, tone: 0.0, guidance: 0
- Actual CAR1: 0.4424, actual percentile: 0.9941
- Earnings surprise: -0.0149

### Example 4: LNN (2026Q2)
- Event: 2026-04-02T10:45:00Z
- Prior earnings count: 2
- Mean prior CAR1: 0.0086
- Last prior competition CAR1: 0.0681
- FLS ratio: 0.4, tone: 0.0, guidance: 0
- Actual CAR1: -0.1222, actual percentile: 0.1061
- Earnings surprise: -0.0043

These examples illustrate the difficulty: with only 3 sealed quarters, the
maximum prior history is 2 events. The relationship between prior CAR1 and
current CAR1 is weak and inconsistent (FC had negative prior history but a
massive positive reaction; LNN had positive prior history but a negative
reaction).

---

## 13. Full Test Summary

```
211 passed, 1 warning in 13.23s
```

- 121 pre-existing tests (all still passing)
- 90 new V2 tests
- 1 pre-existing Starlette/httpx deprecation warning (unrelated)

New test coverage:
- Point-in-time history cutoff enforcement
- No current-event CAR1 leakage
- No future price/earnings leakage
- Same-ticker isolation (no cross-ticker leakage)
- Missing-data behavior (fail closed, indicators not fabricated values)
- Provider/cache behavior (SQLite round-trip, strict cutoff, ticker isolation)
- Price history calculations (returns, volatility, drawdowns, window requirements)
- Surprise z-score and percentile (clipping, zero-variance, low-sample)
- Similar-event matching (kNN on surprise distance)
- Competition-history walk-forward behavior
- V2 feature ordering and imputation
- Artifact loading and coefficient dimensionality
- V2 fallback to V1
- V1 fallback to heuristic
- Heuristic fallback to baseline
- TEST-event neutral behavior (0.50)
- Live prediction bounded output
- Disclosure failure graceful degradation

---

## 14. Leakage-Audit Result

```
n_rows: 6286
n_source_records: 5140
n_rows_with_history: 3686
violations:
  current_event_used_as_source: 0
  future_event_used_as_source: 0
  cross_ticker_source: 0
  outcome_not_yet_available: 0
  source_at_or_after_cutoff: 0
```

The audit runs over every training row before any model is fit. All 5,140
source records satisfy the conservative availability rule. Zero violations.

---

## 15. Artifact Path and Model Version

- **V2 artifact:** `src/explaining_markets/artifacts/fls_company_history_ridge_v2.json`
- **V2 model version:** `fls_company_history_ridge_v2`
- **V2 selected alpha:** 10.0
- **V2 promoted:** false
- **Snapshot artifact:** `src/explaining_markets/artifacts/company_history_snapshot_v1.json`
- **Snapshot size:** 950 KB (2,600 tickers, 6,287 rows)
- **V1 artifact (live):** `src/explaining_markets/artifacts/fls_ridge_v1.json`

---

## 16. Promotion Decision

**V2 was NOT promoted.** The predeclared promotion gate requires ALL of:

| Gate | Threshold | Observed | Pass? |
|------|-----------|----------|-------|
| Validation delta-R² gain | >= 0.002 | 0.000 | No |
| Holdout delta-R² regression | <= 0.002 | 0.019512 | No |
| Holdout prediction std | >= 0.01 | 0.193272 | Yes |

Two of three gates failed. V2 remains disabled in production. The live model
is still `fls_ridge_v1`.

---

## 17. Modal Deployment

**Modal deployment was NOT verified.** No Modal credentials
(`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) are present in the environment.

Artifact packaging was verified locally:
- `modal_app.py` uses `.add_local_python_source("explaining_markets", "predict", ignore=[])`
  which mounts all files in the `explaining_markets` package directory,
  including the `artifacts/` subdirectory.
- All three artifacts (`fls_ridge_v1.json`, `fls_company_history_ridge_v2.json`,
  `company_history_snapshot_v1.json`) load successfully from the package
  directory path that a Modal worker would use.
- The `SnapshotCompanyHistoryProvider` loads the 950 KB snapshot in
  milliseconds and returns correct point-in-time-filtered features.

A real Modal deployment or remote worker smoke test should be run once
credentials are available.

---

## 18. Smoke Test (Local)

```
[MODEL] v2 artifact present but not promoted; using fls_ridge_v1
default model: ForwardLookingRidgeModel fls_ridge_v1
v2 loads: fls_company_history_ridge_v2 promoted: False
v2 AAPL sample prediction (no history): 0.5573 prior_n: 0.0
v2 AAPL sample prediction (with snapshot history): 0.5630 prior_n: 2.0 has_competition: 1.0
v1 AAPL sample prediction: 0.5628
```

This confirms:
- The V2 artifact loads and validates correctly.
- The default model correctly remains V1 (V2 is not promoted).
- V2 produces differentiated predictions when history is available.
- The snapshot provider returns real prior history for AAPL (2 prior events).

---

## 19. Conclusion

The company-history V2 model was designed, implemented, trained, and evaluated
with strict point-in-time discipline. The infrastructure is production-safe:
V2 loads, predicts, and falls back correctly; the live model remains V1; TEST
events still return 0.50; the full test suite passes (211/211).

However, V2 does not improve on V1. With only three sealed quarters, the
maximum prior history per company is 2 events, and the 2025Q4 training quarter
is a complete cold start (0 rows with prior history). The additional features
added noise without signal, causing holdout regression. This is a genuine
negative result, not an implementation error.

V2 should remain disabled unless future research with more historical data
demonstrates a genuine chronological improvement without holdout regression
under the same predeclared promotion gate.
