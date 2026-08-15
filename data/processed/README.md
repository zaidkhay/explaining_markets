# data/processed

Local, gitignored outputs derived from the competition archive and other
point-in-time research inputs.

V3 training now uses this directory for:

- `v3_training_rows.jsonl.gz` — validated V3 training rows. Each row contains
  the within-quarter CAR1 percentile label plus the frozen V3 feature vector.
- `v3_evaluation.json` — chronological ablations/model-selection metrics.
- `multi_signal_v3_research.json` — an explicitly **unpromoted** shadow linear
  artifact for local inference while the production promotion gate remains
  closed.

Build archive seed rows with:

```bash
uv run python scripts/build_v3_training_rows.py
```

Then train/evaluate a research V3 model with:

```bash
uv run python scripts/train_v3_model.py --archive-seed --run-tests
```

Archive seed rows deliberately do not fabricate historical EPS/revenue,
guidance, five-year prices, peers, news, or reasoning when those point-in-time
sources are unavailable. The coverage report makes those gaps explicit.
Production V3 serialization remains gated by the honest holdout, leakage,
distribution, live-feed, reasoning, latency, and test requirements in
`src/explaining_markets/v3_training.py`.

This directory is gitignored except for this README and `.gitkeep`; never
commit bulk historical rows or research artifacts.
