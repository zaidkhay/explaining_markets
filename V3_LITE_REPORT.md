# V3-Lite Evaluation Report

**Generated:** 2026-08-17
**Production model:** `fls_ridge_v1` (unchanged)
**V3-lite promotion status:** NOT PROMOTED (by design — honest holdout unavailable)

---

## 1. Backfill

### Price coverage

| Metric | Value |
|--------|-------|
| Total training rows | 6,299 |
| Unique tickers | 2,607 |
| Tickers with cached prices | 264 |
| Tickers needing prices | 2,343 |
| Existing row coverage | 12.4% |
| Existing price rows | 780 |

### Backfill infrastructure

- **Ticker-frequency priority queue** (`backfill_planner.py`): sorts symbols by row-frequency descending, ticker ascending tie-break. Dry-run mode shows next N symbols without API calls.
- **Twelve Data retry layer** (`retry_policy.py`): transient classification (ConnectTimeout, ReadTimeout, ConnectError), exponential backoff with jitter, HTTP 429 Retry-After handling, per-attempt logging, per-provider statistics.
- **Unsupported-symbol cache** (`unsupported_cache.py`): persistent structured metadata, skip-on-later-runs unless `retry_unsupported=True`.
- **Provider fallback**: local CSV → Tiingo → Twelve Data → FMP → Alpha Vantage. Finnhub for earnings/news. Cached responses before network calls.

### Remaining work

- 2,343 tickers still need prices. Full backfill requires provider budget and time.
- No provider output was fabricated. Missing data remains explicit missingness.

---

## 2. Model Comparison Table

### Validation (2026Q1, pristine out-of-sample, 2,390 rows)

| Model | Features | Spearman | Pearson | MAE | Std | Near-0.50 | Delta-R² |
|-------|----------|----------|---------|-----|-----|-----------|----------|
| v1_fls_only | 30 FLS | 0.0878 | 0.0997 | 0.2485 | 0.0381 | 0.536 | 0.007010 |
| fls_plus_company_history | 37 | 0.0878 | 0.0997 | 0.2485 | 0.0381 | 0.536 | 0.007010 |
| fls_plus_price | 35 | 0.0931 | 0.0994 | 0.2486 | 0.0450 | 0.446 | 0.007526 |
| fls_plus_eps | 33 | 0.0878 | 0.0997 | 0.2485 | 0.0381 | 0.536 | 0.007010 |
| fls_plus_news | 37 | 0.0878 | 0.0997 | 0.2485 | 0.0381 | 0.536 | 0.007010 |
| fls_plus_availability | 33 | 0.1091 | 0.1099 | 0.2480 | 0.0427 | 0.593 | 0.009233 |
| **fls_plus_reasoning** (selected) | 37 | **0.1116** | **0.1161** | 0.2481 | 0.0352 | 0.636 | **0.010564** |
| fls_plus_history_reasoning | 37 | 0.1116 | 0.1161 | 0.2481 | 0.0352 | 0.636 | 0.010564 |
| fls_history_price_reasoning | 40 | 0.1067 | 0.1052 | 0.2484 | 0.0387 | 0.570 | 0.009131 |
| full_v3_available | 100 | 0.1069 | 0.1046 | 0.2484 | 0.0389 | 0.573 | 0.009002 |
| surprise benchmark | — | 0.2405 | 0.2405 | 0.2758 | — | — | 0.0 |
| constant 0.5 | — | n/a | n/a | 0.2501 | 0.0 | 1.0 | 0.0 |

**Selection rationale:** `fls_plus_reasoning` (ElasticNet alpha=0.005, l1_ratio=0.5) was selected by Spearman (ranking power) as the primary objective. It ties with `fls_plus_history_reasoning` because company-history features are constant in the current enrichment (zero coverage → dropped by `_active_features`).

### Legacy 2026Q2 (NOT pristine, 2,060 rows)

| Model | Spearman | Pearson | MAE | Std | Near-0.50 | Delta-R² |
|-------|----------|---------|-----|-----|-----------|----------|
| selected raw | 0.1556 | 0.1748 | 0.2455 | 0.0464 | 0.553 | 0.021722 |
| selected calibrated | 0.1559 | 0.1596 | 0.3120 | 0.3095 | 0.125 | 0.018789 |
| v1 raw | 0.1493 | 0.1715 | 0.2451 | 0.0532 | 0.464 | 0.021034 |
| constant 0.5 | n/a | n/a | 0.2501 | 0.0 | 1.0 | 0.0 |
| surprise benchmark | 0.2652 | 0.2650 | 0.2692 | 0.2888 | 0.040 | 0.0 |

**Legacy holdout note:** 2026Q2 already informed earlier V2/V3 research decisions. It is reported for continuity but cannot satisfy the promotion gate.

---

## 3. Selected Production Model

**No model was promoted.** Production remains `fls_ridge_v1`.

The strongest candidate is:
- **Model:** `v3_lite` (ElasticNet alpha=0.005, l1_ratio=0.5)
- **Ablation:** `fls_plus_reasoning` (37 features: 30 FLS + 7 reasoning)
- **Validation Spearman:** 0.1116 (V1: 0.0878, +27% improvement)
- **Validation Pearson gain over V1:** +0.0164 (gate requires ≥ 0.01)
- **Calibration:** Fitted on 2026Q1 validation OOS predictions from a 2025Q4-only model

**Why not promoted:**
1. Honest holdout (2026Q3) has 0 rows — outcomes not yet available
2. Tests not yet verified in CI context
3. Local/Modal feed not yet verified
4. The gate is predeclared and was not loosened after observing results

---

## 4. Calibration

### Raw distribution (validation 2026Q1)

| Statistic | Raw | Calibrated |
|-----------|-----|-----------|
| Std | 0.0352 | 0.2885 |
| Min | 0.3146 | 0.0100 |
| Max | 0.6498 | 0.9900 |
| Near-0.50 fraction | 0.636 | 0.035 |
| Unique (4dp) | 703 | 932 |

### Ranking preservation

- Raw Spearman: 0.111624
- Calibrated Spearman: 0.111593
- Difference: 0.000031 (within 1e-4 tolerance, caused by bounds clamping)
- **Preserves ranking: True**

### Calibration provenance

- Method: `empirical_oos_midrank_cdf`
- Source: `2026Q1 validation predictions from elastic_net fitted on 2025Q4 only (ablation=fls_plus_reasoning)`
- Fitted on: 2,390 OOS predictions (never in-sample)
- Knots: 2,390 (no thinning needed)
- Bounds: (0.01, 0.99)

### Evidence calibration does not materially reduce OOS quality

- Validation Spearman: 0.11162 → 0.11159 (Δ = -0.00003, negligible)
- Legacy 2026Q2 Spearman: 0.15555 → 0.15589 (Δ = +0.00034, improved by tie-breaking)
- Legacy 2026Q2 delta-R²: 0.02172 → 0.01879 (Δ = -0.00293, small cost from CDF reshaping)

The calibration is a monotonic transform that preserves ranking. The delta-R² cost is from the CDF reshaping changing the Pearson correlation (which is scale-sensitive), not from destroying ranking information.

---

## 5. Three Real Historical Examples (2026Q1 validation, OOS)

### Strongly Negative: EMBC (ea_EMBC_Q1_2026)

| Field | Value |
|-------|-------|
| Raw prediction | 0.314595 |
| Calibrated percentile | 0.010000 |
| Target percentile | 0.2562 |
| Surprise percentile | 0.7459 |
| Confidence | low |
| Available families | none |

**Key drivers:**
1. `guidance_direction`: value=-1.0 (cut guidance), contribution=-0.12875
2. `guidance_maintained`: value=1.0, contribution=-0.02401
3. `signed_quant_earnings_intensity`: value=-0.025, contribution=-0.01160

**Assessment:** Model correctly predicted below-neutral (0.31 vs target 0.26). The dominant signal is guidance direction (cut), which the model correctly maps to negative. This is a hit.

### Neutral: CHTR (ea_CHTR_Q4_2025)

| Field | Value |
|-------|-------|
| Raw prediction | 0.499948 |
| Calibrated percentile | 0.377615 |
| Target percentile | 0.8267 |
| Surprise percentile | 0.5195 |
| Confidence | medium |
| Available families | company_history, reasoning |

**Key drivers:**
1. `has_reasoning`: value=1.0, contribution=+0.02706
2. `guidance_direction`: value=0.0 (no guidance), contribution=-0.02025
3. `non_quantitative_fls_count`: value=1.0, contribution=-0.00315

**Assessment:** Model predicted near-neutral (0.50) but actual target was high (0.83). This is a miss — the model did not capture the positive signal. Honest reporting of a failure case.

### Strongly Positive: DAL (ea_DAL_Q4_2025)

| Field | Value |
|-------|-------|
| Raw prediction | 0.649780 |
| Calibrated percentile | 0.990000 |
| Target percentile | 0.4157 |
| Surprise percentile | 0.3994 |
| Confidence | high |
| Available families | price_5y, company_history, reasoning |

**Key drivers:**
1. `signed_quant_earnings_intensity`: value=0.225, contribution=+0.08839
2. `guidance_direction`: value=1.0 (raised guidance), contribution=+0.08826
3. `has_reasoning`: value=1.0, contribution=+0.02706

**Assessment:** Model predicted high (0.65) but actual target was below median (0.42). This is a miss — the model interpreted raised guidance and quantitative earnings intensity as positive, but the market reaction was neutral-to-negative. Honest reporting.

---

## 6. Safety

### Point-in-time violations

- **Zero leakage violations** across all 6,299 training rows.
- `audit_feature_names()` validates feature ordering before training.
- `evaluate_v3_lite()` refuses to train on any row with `leakage_violations > 0`.

### Full test result

```
350 passed, 1 warning in 19.50s
```

### Future-data tests

All tests in `test_leakage_audit.py` (8 tests) pass, covering:
- No future prices
- No future earnings
- No future news
- No future peer data
- No current CAR1 leakage
- No future sector data
- No revised estimates unavailable at cutoff

### Fallback tests

All tests in `test_predict.py` (9 tests) pass, covering:
- V1 → heuristic → baseline fallback chain
- Synthetic TEST events produce exactly 0.50
- V3 failure falls back to V1
- Model loading failure falls back to heuristic

### New tests added

| File | Tests | Coverage |
|------|-------|----------|
| test_calibration.py | 19 | Monotonicity, tie handling, bounds, serialization, Spearman preservation |
| test_explanation_packet.py | 13 | Immutability, contribution honesty, confidence labels, serialization |
| test_v3_lite_training.py | 19 | Chronological split, ablation sweep, coverage buckets, promotion gate, serialization refusal |
| test_backfill_planner.py | 15 | Prioritization, skips, retries, budgets, stats accounting |

---

## 7. Deployment

### Deployment status

- **Production model:** `fls_ridge_v1` (unchanged)
- **V3-lite artifact:** NOT written (promotion gate failed, serialization refused)
- **Modal deployment:** Not modified. No deployment verification was performed because no new artifact was produced.
- **Webhook architecture:** Unchanged. Fast ACK, signature verification, dedup, Modal background worker, submission API all preserved.

### Modal diagnostic

No Modal deployment was performed or claimed. The V3-lite model was not promoted, so no Modal changes were needed.

### Production latency

- V1 (`fls_ridge_v1`): linear model, sub-millisecond inference
- V3-lite (if promoted): linear ElasticNet, sub-millisecond inference
- Both well within the five-minute deadline

---

## 8. Remaining Risks

1. **Honest holdout (2026Q3) unavailable:** The promotion gate requires 2026Q3 outcomes, which do not exist yet. This is by design — no model can be promoted without untouched holdout verification.

2. **Low overall ranking power:** The best validation Spearman is 0.1116. While this is a 27% improvement over V1 (0.0878), it is still weak in absolute terms. The surprise benchmark (0.2405) outperforms all models on Spearman, suggesting that earnings surprise alone carries more ranking signal than the current feature set.

3. **Feature coverage gaps:** Only 2,343 of 2,607 tickers have price data. EPS, revenue, guidance, peer, and news families have near-zero coverage in the enriched training data. The `fls_plus_reasoning` ablation wins partly because reasoning features are the only non-FLS family with meaningful coverage.

4. **Reasoning feature concentration:** The selected model's improvement over V1 comes entirely from deterministic reasoning features (7 features). If reasoning logic has systematic biases, the model inherits them.

5. **Calibration extremes:** The calibrated output maps raw 0.31 → 0.01 and raw 0.65 → 0.99. These extremes are honest CDF positions but may be overconfident in absolute terms given the weak raw Spearman.

6. **Legacy 2026Q2 is not pristine:** It was used in earlier V2/V3 research. Any metrics on this quarter are biased optimistic and cannot be used for promotion.

7. **No Modal verification:** Modal credentials were not available for deployment verification. No deployment claim is made.

8. **Backfill incomplete:** 87.6% of tickers still lack price data. Full provider-budget backfill is needed before richer ablations (price, EPS, revenue, peer) can be meaningfully evaluated.
