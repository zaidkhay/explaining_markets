# V3 Historical Enrichment

This workflow converts the archive-seed V3 matrix into a richer point-in-time historical matrix before research retraining.

## Data flow

```text
competition archive seed rows
        +
Alpha Vantage historical EARNINGS (cached by ticker)
        +
Alpha Vantage historical NEWS_SENTIMENT (cached in broad earnings-news windows)
        +
explicit adjusted daily price history (local CSV or entitled Alpha Vantage Daily Adjusted)
        ↓
point-in-time joins at each event cutoff
        ↓
news ranking + structured article reasoning
        ↓
event-level structured reasoning
        ↓
full V3 feature reassembly + point-in-time audit
        ↓
data/processed/v3_training_rows_enriched.jsonl.gz
        ↓
coverage gate
        ↓
research-only V3 retraining
```

## Safety rules

- Focal-event CAR1 remains label-only and is never used as a V3 feature.
- Historical EPS matching uses Alpha Vantage `EARNINGS` records near the competition event date; the reconstructed current-event earnings record is marked available exactly at the competition event cutoff, never before it.
- Historical news must have both `published_at <= cutoff` and `available_at <= cutoff`.
- Reasoners receive only the curated pre-cutoff packet and never receive CAR1 or the target percentile.
- Historical price features are populated only from explicitly adjusted daily closes. The pipeline does not substitute monthly data, compact recent data, or unadjusted split-distorted history for the daily V3 feature family.
- Vendor payloads are cached locally under `data/enrichment/v3/`, which is gitignored.
- Production promotion remains independent and fail-closed.

## Alpha Vantage call budgeting

The default CLI reserves 19 requests for historical earnings and 6 for historical news per run. Successful responses are cached, so repeated runs advance through uncached tickers/windows rather than paying for the same request again.

```bash
uv run python scripts/enrich_v3_training_rows.py
```

For a larger API allowance:

```bash
uv run python scripts/enrich_v3_training_rows.py \
  --earnings-api-calls 200 \
  --news-api-calls 50
```

To generate historical reasoning with OpenAI instead of the deterministic structured fallback:

```bash
uv run python scripts/enrich_v3_training_rows.py \
  --reasoning-mode openai
```

The deterministic mode is the default because it is reproducible, bounded, and does not depend on API quota.

## Historical prices

The local bulk CSV schema is:

```text
ticker,date,close,volume,available_at,source
AAPL,2021-01-04,129.41,143301900,2021-01-04T21:00:00+00:00,my_adjusted_source
```

Required columns are `ticker`, `date`, and `close`. `volume`, `available_at`, and `source` are optional. `close` must already be split/dividend adjusted according to the selected point-in-time-safe data policy.

Use it with:

```bash
uv run python scripts/enrich_v3_training_rows.py \
  --price-csv data/external/historical_prices.csv
```

If the Alpha Vantage key is entitled to the premium full Daily Adjusted endpoint, it can be tried with:

```bash
uv run python scripts/enrich_v3_training_rows.py --alpha-adjusted-prices
```

Unsupported/premium responses do not fabricate price features.

## Coverage gate and retraining

By default the enrichment CLI audits coverage and refuses automatic retraining until:

- EPS coverage >= 30%
- company-news coverage >= 20%
- reasoning coverage >= 20%

The thresholds are research gates, not production promotion criteria. Override them only for deliberate experiments.

```bash
uv run python scripts/enrich_v3_training_rows.py --retrain
```

When the coverage gate passes, this invokes the existing research-only `train_v3_model.py` on the enriched rows and still does not promote the production artifact.
