# Historical-Model Experiment

**Status:** Complete. Locked holdout (2026Q2) evaluated exactly once, after all
modeling decisions were made on TRAIN (2025Q4) → VALIDATION (2026Q1) only.

**Scope:** Offline research only. Nothing in this document changes
`predict.py`, `modal_app.py`, `model.py`, or any production path. The
experiment lives entirely in
<ref_file file="/Users/zkhayyat/Projects/markets/src/explaining_markets/experiment.py" />
and is tested by
<ref_file file="/Users/zkhayyat/Projects/markets/tests/test_experiment.py" />.

---

## 1. Question

Do the point-in-time-safe historical features built by
`feature_store.py` add genuine out-of-sample predictive signal for the
next-day abnormal-return percentile, beyond:

1. a constant `0.5` baseline,
2. the existing disclosure-only `HeuristicFactModel`,
3. the earnings-surprise benchmark (evaluation only — not a live feature)?

---

## 2. Data

| Quarter       | Role                              | Labeled rows |
| ------------- | --------------------------------- | -----------: |
| `2025Q4`      | TRAIN                             |        1,849 |
| `2026Q1`      | VALIDATION (model selection only) |        2,390 |
| `2026Q2`      | **LOCKED HOLDOUT** (evaluated once)|        2,047 |
| **Total**     |                                   |    **6,286** |

Source: `data/historical/EARNINGS_RELEASE_*.jsonl.gz` (sealed archives, see
`data/historical/PROVENANCE.json`). One record across the three quarters
lacked a valid `car1` and was excluded from labeled evaluation, matching
`backtest.py`'s existing behavior.

The target `y` is the within-quarter percentile rank of `car1`, taken
verbatim from `backtest.run_backtest` — this module does **not** recompute
the target.

---

## 3. Chronological discipline

```
TRAIN       = 2025Q4   (earliest sealed quarter)
VALIDATION  = 2026Q1   (used ONLY for hyperparameter selection)
HOLDOUT     = 2026Q2   (LOCKED — evaluated exactly once, post-selection)
```

* Hyperparameters (Ridge `alpha`, forest `max_depth`) were selected by
  fitting on TRAIN and scoring on VALIDATION. The holdout was **not**
  consulted during selection.
* After selection, models were refit on TRAIN+VALIDATION and evaluated
  exactly once on the holdout.
* The holdout is never iterated against. All numbers in §6 for 2026Q2 are
  the single, final, post-hoc read.

This is enforced structurally in `run_experiment()` and verified by
`test_run_experiment_preserves_chronological_split` and
`test_run_experiment_on_real_archive_has_no_leakage_and_clean_split`.

---

## 4. Features

### 4.1 Disclosure features (from `features.py`)

`n_facts`, `text_length`, `positive_hits`, `negative_hits`,
`net_sentiment`, `has_guidance_mention`.

These are exactly what `predict.py` and `backtest.py` already use —
extracted via `extract_features(ticker, event_type, disclosure)`.

### 4.2 Historical features (from `feature_store.py`)

| Family               | Fields                                                                 |
| -------------------- | ---------------------------------------------------------------------- |
| `historical_car1`    | `previous_car1`, `rolling_mean_car1`, `rolling_car1_volatility`, `historical_reaction_asymmetry` |
| `historical_surprise`| `previous_earnings_surprise`, `rolling_mean_surprise`, `number_of_previous_positive_surprises` |
| `historical_count`   | `number_of_prior_earnings_events`                                      |

Every historical feature is built **only** from strictly-earlier events
for the **same ticker** (walk-forward, cross-ticker-isolated, with full
provenance — see `PREDICTION_FEATURE_STORE.md`). The six imputed fields
use train-only median imputation plus a missing-indicator column; the two
count fields are always defined.

### 4.3 Forbidden fields (never in any feature matrix)

`car1`, `earnings_surprise`, `surprise`, `event_returns`,
`baseline_predictions`, `y`, `predicted_percentile` — enforced by
`assert_matrix_is_leakage_free()` and swept across all 6,286 real records
in `test_run_experiment_on_real_archive_has_no_leakage_and_clean_split`.

---

## 5. Models

| Model                  | Family        | Notes                                                     |
| ---------------------- | ------------- | --------------------------------------------------------- |
| `constant_0.5`         | baseline      | No signal; correlation undefined (zero variance).         |
| `disclosure_heuristic` | disclosure    | Existing `HeuristicFactModel` from `model.py`.            |
| `historical_ridge`     | historical    | Ridge on historical features only.                        |
| `historical_forest`    | historical    | Shallow random forest (depth ≤ 6) on historical features. |
| `combined_ridge`       | disclosure+historical | Ridge on both feature sets.                        |
| `combined_forest`      | disclosure+historical | Shallow random forest on both feature sets.        |
| `surprise_benchmark`   | eval-only     | `surprise_percentile` — **not** a live feature.           |

Ridge `alpha` ∈ {0.1, 1.0, 10.0, 100.0}; forest `max_depth` ∈ {2, 3, 4, 6}.
Selected on VALIDATION only:

```
ridge_alpha_historical:    0.1
forest_max_depth_historical: 2
ridge_alpha_combined:      100.0
forest_max_depth_combined: 2
```

No LLM is used. All models are interpretable, regularized, or shallow.

---

## 6. Results

### 6.1 Per-quarter metrics

Pearson `r` and MAE. `r = None` means undefined (zero variance in
predictions — the constant-baseline case).

**2025Q4 (TRAIN, in-sample):**

| Model                  | n    | r       | MAE    |
| ---------------------- | ---: | ------: | -----: |
| constant_0.5           | 1849 | None    | 0.2501 |
| disclosure_heuristic   | 1849 |  0.1712 | 0.2741 |
| historical_ridge       | 1849 |  0.0000 | 0.2502 |
| historical_forest      | 1849 |  0.0000 | 0.2501 |
| combined_ridge         | 1849 |  0.1805 | 0.2460 |
| combined_forest        | 1849 |  0.1856 | 0.2455 |
| surprise_benchmark     | 1849 |  0.2223 | 0.2799 |

**2026Q1 (VALIDATION, used for selection):**

| Model                  | n    | r       | MAE    |
| ---------------------- | ---: | ------: | -----: |
| constant_0.5           | 2390 | None    | 0.2501 |
| disclosure_heuristic   | 2390 |  0.0955 | 0.2813 |
| historical_ridge       | 2390 |  0.0705 | 0.2494 |
| historical_forest      | 2390 |  0.1747 | 0.2475 |
| combined_ridge         | 2390 |  0.1201 | 0.2472 |
| combined_forest        | 2390 |  0.1553 | 0.2462 |
| surprise_benchmark     | 2307 |  0.2405 | 0.2758 |

**2026Q2 (LOCKED HOLDOUT — evaluated once, post-selection):**

| Model                  | n    | r       | MAE    |
| ---------------------- | ---: | ------: | -----: |
| constant_0.5           | 2047 | None    | 0.2501 |
| disclosure_heuristic   | 2047 |  0.1643 | 0.2913 |
| historical_ridge       | 2047 | -0.0253 | 0.2619 |
| historical_forest      | 2047 | -0.0275 | 0.2506 |
| combined_ridge         | 2047 |  0.1548 | 0.2470 |
| combined_forest        | 2047 |  0.1318 | 0.2475 |
| surprise_benchmark     | 1989 |  0.2632 | 0.2694 |

### 6.2 Aggregate walk-forward holdout (2026Q2, the only unbiased read)

The single number that matters for the research question:

* **Best non-surprise model:** `disclosure_heuristic`, r = 0.1643.
* **Best disclosure+historical model:** `combined_ridge`, r = 0.1548 —
  *below* disclosure alone.
* **Historical-only models:** r ≈ -0.03 — slightly *negative*, i.e. the
  historical-only models selected on 2026Q1 do not generalize and are
  marginally worse than constant.
* **Surprise benchmark:** r = 0.2632 — the strongest signal by a wide
  margin, but it is evaluation-only (post-event) and not a live feature.

**The historical feature store does not add genuine out-of-sample
predictive signal beyond the disclosure heuristic on the locked holdout.**

---

## 7. Ablation / permutation importance

Correlation drop when each family's columns are shuffled (higher = the
family contributed more to the fitted `combined_ridge` model's
correlation; near-zero or negative = the family carried no incremental
signal).

| Family               | Validation (2026Q1, decision-safe) | Holdout (2026Q2, post-hoc) |
| -------------------- | ---------------------------------: | -------------------------: |
| `disclosure`         |                          0.0677   |                   0.1870   |
| `historical_car1`    |                          0.0024   |                  -0.0048   |
| `historical_surprise`|                          0.0006   |                  -0.0040   |
| `historical_count`   |                         -0.0007   |                   0.0033   |

`disclosure` is the only family whose removal hurts. All three historical
families have drops indistinguishable from zero (or slightly negative,
meaning shuffling them *helps* — a clear sign of noise-fitting). This is
consistent across validation and holdout.

---

## 8. Why: the cold-start problem

The root cause is structural, not a modeling bug. The archive contains
only three sealed quarters, and `2025Q4` is the earliest. Because the
feature store uses **strictly-earlier same-ticker events only**, every
ticker in `2025Q4` has zero prior history:

```
2025Q4 distribution of number_of_prior_earnings_events:
  prior_n=0: 1849 rows   (100% of the training quarter)
```

Historical-feature coverage by quarter:

| Field                              | 2025Q4 | 2026Q1 | 2026Q2 |
| ---------------------------------- | -----: | -----: | -----: |
| `previous_car1`                    |   0.0% |  72.1% |  95.9% |
| `rolling_mean_car1`                |   0.0% |  72.1% |  95.9% |
| `rolling_car1_volatility`          |   0.0% |   0.0% |  70.8% |
| `previous_earnings_surprise`       |   0.0% |  72.1% |  94.0% |
| `rolling_mean_surprise`            |   0.0% |  72.1% |  94.0% |
| `historical_reaction_asymmetry`    |   0.0% |   0.0% |  36.4% |
| `number_of_previous_positive_surprises` | 100% |  100% |  100% |
| `number_of_prior_earnings_events`  |  100% |  100% |  100% |

Consequences:

1. **TRAIN has no historical signal to learn from.** Every imputed
   historical feature in 2025Q4 is the median of an empty set → 0.0, with
   the missing-indicator = 1. The only non-degenerate historical columns
   in TRAIN are the two counts, both identically 0. Ridge/forest fit on
   this produce constant predictions → undefined validation correlation.
2. **This is why the original `alpha=None` crash occurred.** When every
   Ridge candidate's validation correlation is `None` (undefined), the
   old `corr > best_corr` comparison (`-inf > -inf`) was always `False`,
   so `best_alpha` stayed `None`. Fixed in
   <ref_snippet file="/Users/zkhayyat/Projects/markets/src/explaining_markets/experiment.py" lines="454-469" />
   by initializing `best_model` on the first candidate. Regression-tested
   by `test_run_experiment_never_returns_none_hyperparameters`.
3. **VALIDATION (2026Q1) has partial coverage** (72%), but
   `rolling_car1_volatility` (needs ≥2 prior events) and
   `historical_reaction_asymmetry` (needs both positive and negative
   prior CAR1s) are still essentially empty. The historical-only forest
   picks up some signal here (r=0.1747) but it does not generalize.
4. **HOLDOUT (2026Q2) finally has rich coverage** (96% / 71% / 36%), but
   by then the models have already been selected on the sparse VALIDATION
   quarter and the selected configurations (depth=2, alpha=0.1/100) are
   not expressive enough to exploit it.

In short: **three quarters is not enough history for a per-ticker
walk-forward historical feature store to demonstrate out-of-sample
signal.** The infrastructure is correct; the data is the bottleneck.

---

## 9. Leakage checks

All passed (see `tests/test_experiment.py`):

* `assert_matrix_is_leakage_free` raises for every key in
  `FORBIDDEN_KEYS` (`car1`, `earnings_surprise`, `surprise`,
  `event_returns`, `baseline_predictions`, `y`, `predicted_percentile`).
* No feature name produced by any `MatrixBuilder` configuration
  (disclosure-only, historical-only, combined) is a forbidden field.
* The chronological split is strict: each `event_id` appears in exactly
  one quarter bucket; TRAIN/VALIDATION/HOLDOUT are pairwise disjoint.
* The real-archive sweep (`test_run_experiment_on_real_archive_has_no_leakage_and_clean_split`)
  confirms all of the above across all 6,286 records.
* `MatrixBuilder` fits imputation medians on TRAIN only and reuses them
  unchanged for VALIDATION/HOLDOUT — verified by
  `test_matrix_builder_fit_then_transform_uses_train_medians_only`.

---

## 10. Test suite

```
121 passed, 1 warning in 4.61s
```

100 pre-existing tests + 21 new tests in `tests/test_experiment.py`. The
single warning is a pre-existing Starlette/httpx deprecation in
`fastapi.testclient`, unrelated to this work.

---

## 11. Production isolation

`git status` after this work:

```
 M pyproject.toml          (research dependency group only)
 M uv.lock                 (auto-generated)
?? src/explaining_markets/experiment.py   (new, research-only)
?? tests/test_experiment.py               (new, research-only)
```

`predict.py`, `modal_app.py`, `model.py`, `backtest.py`,
`historical.py`, `features.py`, and `feature_store.py` are **unchanged**.
The `research` dependency group (`numpy`, `scikit-learn`) is not read by
`modal_app.py`'s deploy image — that image's dependencies are the
explicit `pip_install(...)` call in `modal_app.py`, which was not touched.

---

## 12. Conclusions

1. **The historical feature store does not add out-of-sample predictive
   signal on the locked 2026Q2 holdout.** Combined ridge (r=0.1548) is
   marginally *worse* than the disclosure heuristic alone (r=0.1643);
   historical-only models are slightly negative (r≈-0.03).
2. **The earnings-surprise benchmark (r=0.2632) remains the strongest
   signal by a wide margin**, but it is post-event and not a live
   feature. The competition's incremental-R² score is precisely the
   question of whether anything can beat surprise — and right now,
   nothing here does.
3. **The cause is data, not infrastructure.** The feature store, leakage
   guards, chronological split, and provenance are all correct (verified
   by the full test suite and the 6,286-record sweep). The bottleneck is
   that 2025Q4 — the only available training quarter — is a per-ticker
   cold-start quarter with zero prior history, so no historical signal
   can be learned in TRAIN.
4. **Ablation confirms disclosure is the only contributing feature
   family.** Shuffling any historical family does not reduce correlation
   (and sometimes slightly increases it), on both validation and holdout.

---

## 13. Recommended next experiment

1. **Acquire more sealed quarters (2025Q1–Q3).** This is the single
   highest-leverage action. With ≥5 quarters, the training set will
   contain tickers with real prior history, eliminating the cold-start
   problem. Re-run this exact experiment unchanged — the infrastructure
   already handles multi-quarter walk-forward.
2. **Add cross-sectional (ticker-level) features that do not require
   per-ticker history** — e.g. industry/sector momentum, market-cap
   bucket, or same-day-peer reaction. These are populated from the first
   quarter and avoid the cold-start gap entirely.
3. **Test whether disclosure + surprise (as a *live* feature, not just
   eval-only) beats surprise alone.** This is the actual competition
   question. It requires resolving whether `earnings_surprise` can ever
   be point-in-time safe at prediction time (see
   `docs/PREDICTION_TIME_INFORMATION_AUDIT.md` — currently classified
   `POST_EVENT`).
4. **If more quarters remain unavailable, switch to a leave-one-quarter-out
   cross-validation over the three available quarters** for *model-family*
   selection (not for the locked holdout), to reduce the variance from
   having a single sparse validation quarter. The locked 2026Q2 holdout
   protocol stays unchanged.
