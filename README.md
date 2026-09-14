# Explaining Markets — V3-lite production

Production system for forecasting the next-day abnormal market reaction to corporate events as a within-quarter percentile.

## Current production path

```text
Explaining Markets webhook
        ↓
signed-event verification + immediate ACK
        ↓
fetch information_url facts
        ↓
point-in-time disclosure parsing + bounded live context
        ↓
V3 feature vector
        ↓
v3_lite_candidate.json
        ↓
out-of-sample percentile calibration
        ↓
competition submission
```

The deployed scorer is the operator-selected V3-lite candidate. The artifact remains `promoted=false` because the untouched-holdout promotion gate has not been satisfied; the operator override is explicit and auditable.

## Production files

- `modal_app.py` — webhook receiver, dedupe, async worker, diagnostics
- `predict.py` — live disclosure fetch and scoring entry point
- `src/explaining_markets/model_v3_lite.py` — V3-lite inference
- `src/explaining_markets/artifacts/v3_lite_candidate.json` — deployed fitted model
- `src/explaining_markets/features_v3.py` — frozen V3 feature assembly
- `src/explaining_markets/disclosure_results_v3.py` — focal-disclosure result parsing
- `src/explaining_markets/live_v3_context.py` — point-in-time live context
- `src/explaining_markets/point_in_time_audit_v3.py` — leakage guard
- `src/explaining_markets/reasoning/` — deterministic/OpenRouter reasoning
- `src/explaining_markets/providers/` — bounded live and historical providers

## Active retraining pipeline

Only the current V3-lite training path is retained:

```text
competition archive
    ↓
scripts/build_v3_training_rows.py
    ↓
scripts/enrich_v3_training_rows.py
    ↓
scripts/build_v3_lite_candidate.py
    ↓
scripts/verify_v3_lite_candidate.py
    ↓
scripts/audit_v3_lite_collapse.py
```

Bulk historical data and caches are intentionally gitignored.

## Setup

```bash
uv sync
cp .env.example .env
```

Required competition credentials:

```text
EM_API_KEY
EM_WEBHOOK_SECRET
```

Provider credentials are optional and fail closed when unavailable.

## Test

```bash
uv run pytest
```

The retained suite focuses on the live V3 path, point-in-time behavior, current retraining pipeline, webhook security, and provider failure handling.

## Verify the fitted candidate

```bash
uv run python scripts/verify_v3_lite_candidate.py
uv run python scripts/audit_v3_lite_collapse.py
```

## Deploy

```bash
uv run modal deploy modal_app.py
uv run modal run modal_app.py::check_production
```

The production diagnostic must report:

```text
status=PASS
model=v3_lite_operator_2026_08_19
live_gate_passed=True
```

Live feed diagnostic:

```bash
uv run modal run modal_app.py::check_v3_feed --ticker AAPL
```

## Live logs

```bash
uv run modal app logs explaining-markets-starter --since 2h --search LIVE_INPUT
uv run modal app logs explaining-markets-starter --since 2h --search V3_FEED
uv run modal app logs explaining-markets-starter --since 2h --search V3_LITE_MODEL
```

`LIVE_INPUT` should contain actual disclosure facts from `items[*].content`, never schema metadata such as event IDs or timestamps.

## Point-in-time rule

Every feature used for a focal event must have been legally knowable by that event's cutoff. Provider failures or unknown availability are represented as missing data rather than fabricated values. CAR1, realized returns, future prices, and post-cutoff news are never model inputs.

## Emergency rollback

The repository retains the validated V1 model only as an emergency rollback path:

```text
PRODUCTION_MODEL=v1
```

Re-deploy Modal after changing that environment setting.
