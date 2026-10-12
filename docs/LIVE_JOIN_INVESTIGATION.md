# Live join investigation (CLAUDE.md §7.3) — 2026-10-11

**Question.** Only 18 of 59 persisted live predictions joined to archive outcomes.
Were the missing 41 missing at random, or systematically?

**Answer.** Mostly timing, plus one real loader bug. With the archive now
published, **287 of 306** evidence rows join. No systematic selection was found.

Reproduce:

```bash
uv run modal volume get --force em-v3-data evidence data/live_eval
uv run python scripts/live_scorecard.py --refresh-archive
```

## Where every evidence row goes (306 rows, archives 2026Q3 + 2026Q4)

| Rows | Status | Cause | Bug? |
|---:|---|---|---|
| 287 | joined | prediction, CAR1 and surprise benchmark all present | — |
| 15 | no_surprise_benchmark | archive `surprise_status = missing_eps`; excluded from the benchmark regression | No — the organiser's regression cannot use them either |
| 4 | outcome_pending | 2026Q4 events from 2026-10-09; CAR1 not yet published | No — will join on the next run |
| (10) | recovered | earliest bundles (2026-08-17/18) store `submitted_percentile` and no `ticker` | **Yes** — the old loader required both fields and dropped them silently |

The original 18-of-59 figure was taken before most of those events' CAR1
outcomes were in the archive. The loader bug is fixed in
`live_scorecard.parse_evidence`, which `evaluate_recent_from_archive.py` now uses.

## Coverage: did every competition event get a prediction?

Archive events since the first evidence cutoff (2026-08-13): 419; with our
evidence: 306 (73%). By week:

- 2026-W32 (go-live week): 1 of 114 — the service was not yet live.
- 2026-W33 through 2026-W40: **100% every week** (305 of 305).

No ongoing loss of events. Missed events are confined to go-live week.

## Score of the joined sample (diagnostic; organiser scorer is authoritative)

All live predictions are frozen and were never used for fitting or selection.
The artifact's fit used 2025Q4 (coefficients, scaler) and 2026Q1 (calibration),
both disjoint from these events. Raw-score Delta R2 is **not available**: no
live bundle stores a raw score (see "Evidence schema" below), so only submitted
values are reported.

| Partition | Sample | n | R2 surprise | Delta R2 (submitted) | Null expectation | F | Permutation p |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026Q3 live | joined subset | 266 | 0.1315 | 0.01234 | 0.00330 | 3.79 | 0.052 |
| 2026Q3 | contest-style common sample, 2,009 of 2,275 events mean-imputed | 266 obs | 0.0837 | 0.00179 | 0.00040 | — | — |
| 2026Q4 live | joined subset | 21 | — | 0.0000008 | 0.0485 | 0.00 | 0.996 |

Reading:

- 2026Q3: 3.7x the null expectation with permutation p ≈ 0.05. Borderline
  evidence of incremental signal, not established skill.
- The contest-style number is lower mainly because 88% of the quarter's events
  are mean-imputed: the service went live in mid-August.
- 2026Q4: n = 21 is far too small to read (§2.4). It says nothing yet.

## Other findings from the evidence (no score impact now)

- **Evidence schema.** All 296 non-legacy bundles are the "summary" schema: no
  raw score, disclosure, feature values or diagnostics. Commit 817787e added
  those fields, but the deployment before 2026-10-11 predated it. The
  2026-10-11 deploy (fb16413) includes it, so bundles from the next event on
  are "full".
- **Company-history cache not mounted.** The `em-v3-data` volume holds only
  `evidence/`; `company_history.sqlite` was never uploaded. `company_history`
  was unavailable on all 296 events. The deployed feature that depends on it,
  `revenue_surprise_percentile_company`, only took the values 0 and 0.5 in
  training as well, so live inputs do not diverge from training. It does block
  any future history-based feature.
- **News provider stopped returning results.** `company_news` was available
  on 76% of earlier events and on 0% from 2026-10-07. News is not among the 39
  deployed features, so there is no score impact, but the provider (Alpha
  Vantage key or rate limit) needs checking.
- **0.7046 is the most common submitted value** (36 of 296). §9.1 records the
  same value from the constant-score bug. Here the overall spread is healthy
  (sd 0.31), so this is most likely the model's no-signal output for disclosures
  with nothing parsed. Full bundles will show the disclosure and confirm it.
  The `repeated_value` alarm watches for it clustering.
- **The largest errors are confident highs.** In the top-10 absolute errors,
  9 are submitted ≥ 0.88 with realized ≤ 0.12. This is consistent with §5's
  evidence that the CDF calibrator pushes scores to the extremes. It is a
  hypothesis for the calibration-removal experiment, not a conclusion, and
  must be tested on historical quarters, not on these live events (§7.1).
