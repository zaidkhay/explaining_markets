# Recent live model evaluation

## Current live performance

| Metric | Recent live | Stored validation |
| --- | ---: | ---: |
| N | 18 | 2390 |
| Spearman | 0.262 | 0.089 |
| Pearson | 0.290 | 0.101 |
| MAE | 0.274 | 0.248 |
| RMSE | 0.381 | 0.288 |
| Direction accuracy | 0.667 | n/a |

## Calibration/extremeness counterfactuals

These are retrospective diagnostics only. They are not automatically promoted. Note that uniform affine shrinkage does not change the official Delta-R2 objective; this section diagnoses absolute percentile error only.

| Shrink factor | Spearman | MAE | RMSE | Mean predicted extremeness |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | n/a | 0.283 | 0.305 | 0.000 |
| 0.25 | 0.262 | 0.250 | 0.293 | 0.075 |
| 0.50 | 0.262 | 0.228 | 0.304 | 0.151 |
| 0.75 | 0.262 | 0.233 | 0.335 | 0.226 |
| 1.00 | 0.262 | 0.274 | 0.381 | 0.301 |

## Failure-mode groups

| Group | N | Spearman | MAE | Signed error |
| --- | ---: | ---: | ---: | ---: |
| revenue_available | 4 | -0.800 | 0.320 | 0.104 |
| revenue_missing | 14 | 0.264 | 0.261 | 0.092 |
| eps_available | 3 | 1.000 | 0.100 | 0.030 |
| eps_missing | 15 | 0.095 | 0.309 | 0.108 |
| guidance_available | 0 | n/a | n/a | n/a |
| guidance_missing | 18 | 0.262 | 0.274 | 0.095 |
| provider_errors | 0 | n/a | n/a | n/a |
| no_provider_errors | 18 | 0.262 | 0.274 | 0.095 |
| high_model_relevance | 0 | n/a | n/a | n/a |
| low_model_relevance | 0 | n/a | n/a | n/a |
| high_interpretation_confidence | 0 | n/a | n/a | n/a |
| low_interpretation_confidence | 0 | n/a | n/a | n/a |
| extreme_predictions | 9 | 0.500 | 0.297 | 0.168 |
| non_extreme_predictions | 9 | 0.025 | 0.252 | 0.022 |

## Feature-family live signal

| Family | Mean contribution | Mean abs contribution | Spearman vs realized | Pearson vs realized |
| --- | ---: | ---: | ---: | ---: |

## Largest misses

| Ticker | Predicted | Realized | Abs error | Revenue available | Model relevance | Top features |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| AAP | 0.883 | 0.034 | 0.849 | True | n/a | n/a |
| WMT | 0.942 | 0.185 | 0.757 | False | n/a | n/a |
| FLUX | 0.705 | 0.023 | 0.682 | False | n/a | n/a |
| SCSC | 0.257 | 0.867 | 0.610 | False | n/a | n/a |
| DE | 0.459 | 0.818 | 0.360 | False | n/a | n/a |
| FLO | 0.053 | 0.382 | 0.329 | True | n/a | n/a |
| GRRR | 0.390 | 0.130 | 0.260 | False | n/a | n/a |
| NSSC | 0.705 | 0.451 | 0.254 | False | n/a | n/a |
| ROST | 0.915 | 0.706 | 0.209 | False | n/a | n/a |
| BJ | 0.946 | 0.751 | 0.195 | False | n/a | n/a |

## Evidence-based improvement hypotheses

1. **GUARDRAIL — Do not fit production coefficients directly to this live sample.**
   - Evidence: Only 18 realized live events are present; use them to form hypotheses, then test those hypotheses on chronological historical splits.
2. **HIGH — Backtest support-aware shrinkage toward 0.5.**
   - Evidence: Retrospective shrink_0.50 lowers recent MAE from 0.274 to 0.228. This is diagnostic only until reproduced out-of-sample.

## Promotion rule

Do not change production weights from this live sample alone. Implement each high-priority hypothesis as an ablation/candidate and require chronological historical validation improvement on the official Delta-R2 objective plus the existing live gate before promotion.
