# V3-lite visualization and audit dashboards

The visualization layer is intentionally diagnostic rather than decorative. It shows what the deployed model actually consumes, how the deterministic parser interprets each supplied disclosure fact, how external context arrived, the exact linear contribution of every deployed feature, and how raw scores are converted into submitted percentiles.

## 1. Inspect one prediction

Use a fresh signed competition `information_url` from `[LIVE_INPUT]` logs:

```bash
uv run python scripts/render_prediction_dashboard.py \
  --ticker WMT \
  --cutoff 2026-08-20T13:00:00Z \
  --event-id ea_WMT_Q2_2027 \
  --url 'https://...fresh signed information URL...'
```

Outputs:

```text
data/diagnostics/ea_WMT_Q2_2027__WMT.json
data/diagnostics/ea_WMT_Q2_2027__WMT.html
```

Open the HTML file in a browser.

### What the event dashboard shows

- supplied disclosure claims
- claim direction and deterministic parser matches
- **interpretation confidence** for each claim
- whether the deployed artifact actually consumes that claim's signal family
- available/missing V3 external-data families
- provider successes and errors
- exact value, z-score, coefficient, and raw-score contribution for every deployed feature
- raw score before calibration
- submitted percentile after empirical OOS calibration
- historical validation Spearman, Pearson, MAE, RMSE, and raw prediction dispersion

### Important confidence definition

`interpretation_confidence` is **not** a probability that the claim is true and is **not** a probability that the prediction is correct.

It measures how explicitly the deterministic parser can map the supplied text into its feature schema. For example:

- explicit realized-vs-consensus result with a numeric surprise: very high parser confidence
- explicit guidance raise/cut: high parser confidence
- quantitative forward-looking directional statement: moderately high parser confidence
- statement outside the deployed parser/model schema: low parser confidence

The competition disclosure itself is treated as supplied evidence; the current system does not independently fact-check each claim.

## 2. Disable fresh external calls

To inspect only the disclosure plus local cache:

```bash
uv run python scripts/render_prediction_dashboard.py \
  --ticker WMT \
  --cutoff 2026-08-20T13:00:00Z \
  --event-id ea_WMT_Q2_2027 \
  --url 'https://...' \
  --no-external
```

This is useful for comparing how much external context changes V3 features. Remember that the current production artifact is `fls_plus_revenue`, so some V3 families may be collected but not directly used by the deployed model.

## 3. Attach a realized outcome later

Once CAR1 and the quarter-relative realized percentile are known:

```bash
uv run python scripts/render_prediction_dashboard.py \
  --ticker WMT \
  --cutoff 2026-08-20T13:00:00Z \
  --event-id ea_WMT_Q2_2027 \
  --url 'https://...' \
  --realized-car1 -0.083 \
  --realized-percentile 0.07
```

The report will add realized CAR1, realized percentile, and absolute percentile error.

## 4. Evaluate many predictions

Create a CSV with at least:

```csv
ticker,predicted_percentile,realized_percentile,car1
AAA,0.80,0.90,0.05
BBB,0.20,0.10,-0.04
CCC,0.60,0.40,-0.01
```

Then run:

```bash
uv run python scripts/render_model_performance_dashboard.py \
  --input data/live_eval/2026-08-20.csv \
  --output data/diagnostics/performance_2026-08-20.html \
  --title 'Aug. 20 live prediction performance'
```

The performance dashboard graphs:

- predicted percentile vs realized percentile
- Spearman rank correlation
- Pearson correlation
- MAE and RMSE
- above/below-0.5 directional agreement
- largest absolute misses
- calibration by submitted-score bucket
- best and worst live calls

For the competition objective, prioritize **Spearman/ranking quality** over visual score spread.
