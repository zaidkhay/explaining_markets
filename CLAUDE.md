# CLAUDE.md — Explaining Markets

Guardrails for any agent or human working in `zaidkhay/explaining_markets`.

Read Section 0 before doing anything. The rest explains *why* each rule exists, because a
rule without a reason gets rationalized away the first time it's inconvenient.

---

## What this project is

A production quantitative prediction system for the Explaining Markets competition. For each
corporate event (mostly earnings), predict the focal stock's next-day abnormal reaction as a
percentile in [0, 1]. The target is the CAR1 percentile relative to that quarter's competition
events.

**The scored objective is Delta R2**: incremental R-squared above an earnings-surprise
benchmark.

```
realized_CAR1_pct ~ b0 + b1 * our_prediction + b2 * surprise_pct

Delta R2 = R2(full) - R2(surprise-only)
```

Everything else — Spearman, Pearson, MAE, RMSE, direction accuracy — is a diagnostic. None of
them is the score. **Improving MAE is not the same as improving Delta R2 and frequently
trades against it.**

The one-line implication that governs most design decisions: *the model is only rewarded for
information the earnings-surprise benchmark does not already contain.*

---

## 0. Non-negotiables

Violating any of these can silently destroy the validity of months of work. Several are
irreversible.

1. **2026Q3's status has not yet been decided — do not let a Q3 number influence a
   specification choice until a human has explicitly recorded that decision.** Scoring
   already-submitted, frozen predictions against Q3 outcomes is legitimate: those predictions
   were fixed before the outcomes existed, so nothing leaks (this is what §7.3's 18-of-59 join
   diagnostic does). Using a Q3 result to choose between model specifications, hyperparameters,
   or features spends the holdout — that conversion has no undo. See §1.2.
2. **Never evaluate an artifact on a quarter it was fit on.** This bug has already happened once
   in `compare_v3_lite_official_score.py`. See §1.3.
3. **Never fit on the live events.** The recent live predictions are for attribution and failure
   diagnosis only. See §7.
4. **Never weaken or bypass the point-in-time audit to make something pass.** If
   `PointInTimeViolation` fires, the feature is wrong, not the audit. See §3.
5. **Never compare Delta R2 to zero.** It is non-negative by construction in-sample. Compare it
   to its null expectation. See §2.
6. **Never commit provider API keys** (Tiingo, Twelve Data, FMP, Finnhub, OpenRouter) or Modal
   credentials. Check diffs before every commit.
7. **Work commits directly to `main` — but a commit on `main` is not a promotion.** Do not
   create branches for ordinary work. Committing a candidate, a sweep, or an evaluation script
   does **not** deploy or bless it. The deployed artifact changes only by explicit operator
   sign-off, recorded with a `(promotion)` tag on the commit subject and a matching update to
   §8.1. Never force-push `main`. See §11.
8. **Never redeploy a Modal artifact without running `check_production` afterward.**
9. **Do not invent a prediction confidence.** The model produces no calibrated probability of
   correctness. See §10.

---

## 1. Data partitions and leakage

### 1.1 The partitions

| Quarter | Rows  | Status |
|---------|-------|--------|
| 2025Q4  | 1,849 | Training |
| 2026Q1  | 2,390 | Validation / model selection |
| 2026Q2  | 2,060 | Legacy chronological read — **NOT pristine** |
| 2026Q3  | —     | **Status undecided — see §1.2** |

Total historical: 6,299 event rows, ~2,607 unique tickers.

### 1.2 What each partition can and cannot be used for

- **2025Q4** — fit coefficients, fit scalers, fit calibrators. Anything.
- **2026Q1** — compare candidates, pick hyperparameters. **Its Delta R2 numbers are
  optimistically biased for anything selected on it.** A candidate chosen because it topped Q1
  cannot then cite its Q1 score as evidence of quality. That number is a selection statistic,
  not an estimate.
- **2026Q2** — chronological but already inspected during prior research. Useful as a
  directional sanity check. Not admissible as confirmatory evidence. Always label it
  "non-pristine" in any writeup.
- **2026Q3** — status not yet decided. The actual distinction is what the number is *used for*,
  not whether it's read:
  - Scoring already-submitted, frozen predictions against Q3 outcomes is legitimate. Those
    predictions were frozen before the outcomes existed, so nothing leaks — this is exactly
    what §7.3's join diagnostic and the worked example in §2.4 do.
  - Using a Q3 result to choose between model specifications, hyperparameters, or features
    spends the holdout. **A human must explicitly record that decision before any Q3 number is
    allowed to influence a specification choice.** That recording has not happened yet — until
    it does, treat Q3 as closed for selection purposes even though scoring frozen predictions
    against it is fine.

### 1.3 The leakage bug that already happened

`compare_v3_lite_official_score.py` originally evaluated the deployed artifact on Q1/Q2 even
though the artifact had been refit using those quarters. The reported "out-of-sample" numbers
were not out-of-sample. It was corrected by reconstructing candidate specifications from
2025Q4 only.

**The general failure pattern:** loading a pre-trained artifact and scoring it on historical
data feels like evaluation, but the artifact carries the fit inside it. The only safe pattern is
to *retrain the specification* on data strictly prior to the evaluation window.

Before adding any evaluation code, answer in a comment: *which rows touched the fit that
produced these coefficients, and is the evaluation window disjoint from them?*

### 1.4 Subtler leakage sources

- **Scaler fitting.** `training_mean` and `training_std` are fit parameters. Standardizing
  with statistics computed over train+val leaks.
- **Calibrator fitting.** `calibration_v1` is fit on Q1 validation predictions. Any evaluation
  that uses it on Q1 is partly in-sample. This is why Q1 raw and submitted numbers must be
  reported separately.
- **Quarter-relative percentiles.** The target is a rank *within the quarter*, so it
  mechanically depends on other events in the same quarter. That is fine for the target. It is
  fatal in a feature. Never construct a feature from any cross-sectional statistic computed
  over the evaluation quarter.
- **Ticker overlap.** ~2.4 events per ticker. Chronological splits handle time, not entity
  reuse. A ticker in both train and validation is acceptable here, but errors are clustered by
  ticker — do not treat n=2,390 as 2,390 independent observations for standard errors.
- **Universe construction.** If the set of events used to compute CAR1 percentiles differs
  from the competition's, the target is wrong in a way that will not raise an error.

---

## 2. Delta R2 — how to read the number

This section exists because the current live results are being over-read.

### 2.1 It is non-negative by construction

Adding a regressor to an OLS design matrix cannot reduce in-sample R-squared. So
`R2(full) >= R2(surprise)` always, for *any* prediction vector, including pure noise. A positive
Delta R2 is not evidence of anything on its own.

Corollary: **"100% of bootstrap samples had Delta R2 > 0" is a statement about arithmetic, not
about skill.** Do not report it as validation.

### 2.2 The null expectation

Under the null that predictions are pure noise, the expected increment is approximately:

```
E[Delta R2 | null] ~= (1 - R2_benchmark) / (n - k)

n = number of events, k = number of fitted parameters (3: intercept, prediction, surprise)
```

**Always compute and report this alongside any Delta R2.**

### 2.3 The significance test

```
F(1, n-k) = Delta_R2 / ((1 - R2_full) / (n - k))
```

### 2.4 Worked example — the 18 live events

```
n = 18, R2_surprise = 0.2905, R2_full = 0.2969, Delta R2 = 0.006369

Null expectation: (1 - 0.2905) / 15 = 0.0473
F(1,15)         : 0.006369 / (0.70314/15) = 0.136
p               : ~0.72
```

The observed increment is roughly **one seventh** of what a useless predictor would produce.
This sample is consistent with the predictions adding nothing beyond earnings surprise. It is
not positive evidence, and if anything sits on the unlucky side of the null.

Contrast with Q1, where `n = 2390` gives a null expectation of ~0.0003, so a raw Delta R2 of
0.00777 is `F ~= 25` — a real effect. **The historical quarters carry signal. The 18 live events
do not, yet.**

### 2.5 Bootstrap on small n

The reported live bootstrap gave a median (+0.0286) more than four times the point estimate
(+0.0064), with an upper bound of +0.32. That asymmetry is the signature of resampling inflating
in-sample fit through duplicated rows, not of a skewed sampling distribution for a real
parameter. Treat bootstrap CIs on n < ~100 as uninformative for this statistic.

If you want a usable small-sample reference, build a **permutation null**: shuffle the prediction
vector against the outcomes many times and compare the observed Delta R2 to that distribution.
That correctly accounts for the mechanical floor.

### 2.6 Affine invariance

The scoring regression fits `b1` freely, so Delta R2 is **invariant to any affine transform** of
the predictions. Consequences:

- Rescaling or shifting predictions changes MAE and RMSE but not Delta R2.
- A "shrink toward 0.5" transform that improves MAE from 0.274 to 0.228 improves **nothing**
  that is scored. Do not deploy it as an improvement.
- Monotone *non-linear* transforms (like the empirical CDF calibrator) are **not** invariant.
  They can and do change Delta R2, usually downward. See §5.

---

## 3. Point-in-time discipline

Every feature must be constructible from information legally available at the event's knowledge
cutoff. This is audited and the audit is a hard gate.

### 3.1 Rules

- Never use a record whose availability timestamp is later than the focal cutoff.
- If `point_in_time_audit_v3.py` raises, **fix the data path**. Do not add an exception, relax a
  comparison, widen a tolerance, or catch and continue.
- A prior real violation: `PointInTimeViolation: stock_prices is not available by focal cutoff`.
  That was a genuine bug caught by the audit working correctly.

### 3.2 Provider-specific traps

These are the ways look-ahead sneaks in through data that *feels* historical:

- **Adjusted price series.** Adjusted closes are retroactively restated for splits and
  dividends. A 5-year adjusted history pulled today encodes future corporate actions. Either use
  unadjusted prices with point-in-time adjustment, or confine adjusted series to features where
  the restatement cannot carry signal — and document which.
- **Restated fundamentals.** Providers serve the *current* version of financials. Revisions and
  restatements postdate the original release.
- **Consensus estimates.** Current consensus is not the consensus that stood at the cutoff.
  Consensus drifts and is sometimes backfilled. An "EPS surprise" computed against today's
  consensus is contaminated.
- **News timestamps.** Publication time vs. index time vs. update time are different. Use the
  earliest defensible one and verify the field actually means what you assume.
- **Backfill runs.** Historical enrichment fetches everything at once, today. Truncation to the
  cutoff must happen at feature construction, not be assumed from the fetch.

### 3.3 When adding any new provider or field

State explicitly, in code or comment:

1. What timestamp establishes availability?
2. Is that timestamp the original publication or a revision?
3. What does the audit compare it against?

---

## 4. Features and coverage

### 4.1 The deployed model is not the collected context

**COLLECTED CONTEXT != DEPLOYED MODEL INPUT.**

The V3 context gathers news, prices, company history, peers, reasoning, and EPS fields. The
production artifact uses **39 features** (~30 FLS + ~9 revenue/result). If a field is not in the
artifact's `feature_names`, it has **zero** coefficient and zero influence on the prediction.

Do not describe the production model as "using news" or "using peers" because the context
contains them. Check `src/explaining_markets/artifacts/v3_lite_candidate.json` before making
any claim about what the model uses.

### 4.2 Sparse coverage is the dominant practical constraint

Reported structured coverage on the historical set:

```
EPS coverage:     0.081
Revenue coverage: 0.299
```

Most events have most result-features missing.

**The trap:** coverage in training may differ substantially from coverage in live production.
Live events now supply ~10 real disclosure facts each after the fetch fix, so live parse rates
may be *higher* than the historical rates the coefficients were fit under. If so, the model is
operating in a regime it was not trained on, and the coefficients on those features are being
exercised far more often than during fitting.

**Before trusting any live-vs-historical comparison, measure live coverage and compare it to
training coverage.** A coverage shift is a distribution shift and invalidates naive comparison.

### 4.3 Missing-value encoding

How a missing feature is encoded matters enormously after standardization:

- Impute raw `0.0` → z = `(0 - mean)/std`, which is a *specific non-neutral value*, often
  extreme, and asserts "this is zero" rather than "this is unknown."
- Impute the training mean → z = 0, contributing nothing. This is usually what "missing" should
  mean.

Whenever a feature is added, verify which of these is happening. A missing-as-zero bug is
invisible in tests and shifts predictions systematically.

Prefer explicit `has_x` indicator features paired with mean-imputed values, so the model can
learn that absence is informative rather than conflating absence with a value.

### 4.4 Do not rebuild the benchmark

Delta R2 rewards only what earnings surprise does not already explain. A feature highly
correlated with earnings surprise contributes ~nothing to the score no matter how well it
predicts CAR1 on its own.

This is why the parser's growth-vs-surprise distinction is **intentional and must not be
"fixed"**:

```
"Revenue increased 9% year over year"   != "Revenue beat consensus by 9%"
```

Realized growth is not an expectations surprise. Treating it as one both misrepresents the
disclosure and pushes features toward the benchmark it needs to be orthogonal to.

**Useful diagnostic:** for any candidate feature, regress it on the surprise percentile and look
at the residual. If the feature is nearly spanned by surprise, it cannot help.

---

## 5. Calibration

### 5.1 Current state

Production applies `calibration_v1`: an empirical mid-rank CDF calibrator fit on out-of-sample
2026Q1 validation predictions from a 2025Q4-trained model. Raw scores are clipped to
`[0.05, 0.95]` before calibration.

### 5.2 The evidence against it

| Quarter | Raw Delta R2 | Submitted Delta R2 | Effect |
|---------|-------------|--------------------|--------|
| 2026Q1  | 0.007770    | 0.005728           | −0.002042 |
| 2026Q2  | 0.015303    | 0.011147           | −0.004156 |

The CDF transform reduced Delta R2 in both quarters examined.

### 5.3 Why this is expected, not surprising

The calibrator was introduced to spread out tightly clustered raw scores. **That clustering was
a symptom of the disclosure ingestion bug, not a property of the model.** With the bug fixed,
raw predictions already span roughly 0.04–0.94.

Meanwhile Delta R2 is affine-invariant (§2.6), so spreading scores out buys nothing that is
scored, while a monotone-nonlinear rank transform distorts whatever linear relationship the raw
score had with the realized percentile. The observed penalty is the mechanism working as
theory predicts.

### 5.4 Rules

- Report **raw and submitted Delta R2 separately**, always. Never collapse them into one number.
- Do not justify calibration by prediction spread, visual differentiation, or MAE.
- Removing calibration requires no refit — it is the cheapest available experiment. Run it
  before considering any model change.
- If calibration is removed, confirm the `[0.05, 0.95]` clip still applies and that submissions
  remain in valid range.
- Removing calibration from the **deployed** path is a promotion. It changes what production
  submits, so it takes a `(promotion)` commit and a §8.1 update. See §11.

---

## 6. Model selection and the winner's curse

### 6.1 Rank on raw, not submitted

The last sweep ranked 144 candidates by *submitted* Delta R2 — which bakes in the CDF transform
that §5 shows is harmful. The ordering inverts under raw:

```
                      raw ΔR2     submitted ΔR2
fls_plus_revenue      0.007770    0.005728      (current production spec)
fls_plus_reasoning    0.007188    0.006955      (sweep "winner")
```

"The reasoning candidate is best" is **conditional on keeping a transform the same evidence says
to drop.** Rank on raw.

### 6.2 Discount the top of any sweep

With 144 candidates evaluated on one quarter, the maximum observed Delta R2 is an order
statistic, not an unbiased estimate. The winner's score is inflated by selection.

Report at minimum:
- the number of candidates evaluated,
- the spread of scores across candidates,
- whether the winner's margin exceeds the run-to-run noise of the evaluation.

If the top ten candidates are within noise of each other, say so and prefer the simpler or more
robust specification rather than the nominal winner.

### 6.3 The current V1-vs-V3 evidence is weak

| Spec | Q1 submitted | Q1 raw | Q2 submitted | Q2 raw |
|------|-------------|--------|--------------|--------|
| V1 (elastic_net, α=0.005, l1=0.5) | 0.005360 | 0.007010 | 0.016096 | 0.021034 |
| V3 (fls_plus_revenue, constrained_ridge, α=100) | 0.005728 | 0.007770 | 0.011147 | 0.015303 |

Q1 is the selection quarter, so V3's +0.00037 edge there is selection-contaminated and inside
noise anyway. Q2 is non-pristine but was not used to select this spec, and V3 **loses** there by
−0.0049 submitted / −0.0057 raw.

Read together: weak evidence for V3, mild evidence against. Do not present V3 as an established
improvement over V1.

---

## 7. Live events

### 7.1 Diagnostic only

Live outcomes exist to identify failure modes and check them against historical hypotheses.
They are **not** training data. Never:

- refit coefficients on recent live events,
- tune hyperparameters to improve the live sample,
- drop a live event because it hurts the score.

### 7.2 Leave-one-out is not a model change

The LOO analysis found removing `ea_SCSC_Q4_2026` would improve Delta R2 by +0.0278 and removing
`ea_BJ_Q2_2026` would worsen it by −0.0056. **These are sensitivity diagnostics.** That a single
event out of 18 moves the statistic by four times the headline estimate is itself the finding:
the estimate is unstable. It is not license to exclude anything.

### 7.3 The join is a selection problem

Only 18 of 59 persisted predictions joined to archive outcomes. **Before drawing any conclusion
from the 18, characterize the missing 41.** Are they missing at random, or systematically —
later events not yet scored, tickers absent from the archive, event_id format mismatches,
failed predictions that never persisted cleanly? If the join drops events non-randomly, the 18
are a biased sample and even a large Delta R2 would not generalize.

This is a higher-priority investigation than any model change.

### 7.4 Evidence schema is not uniform

Older evidence bundles lack the richer schema (exact disclosure, full `feature_values`,
`raw_prediction`, diagnostics). **Do not assume any given evidence file has these fields**, and
do not assume the newest evidence-persistence code is deployed unless a deployment has been
confirmed *after* those commits.

---

## 8. Production and deployment

### 8.1 Current deployed state

```
model version : v3_lite_operator_2026_08_19
ablation      : fls_plus_revenue
model type    : constrained_ridge
alpha         : 100.0
features      : 39
calibration   : calibration_v1
promoted           : False
operator_override  : True
production_candidate : True
```

`promoted = False` is accurate and load-bearing. The normal untouched-holdout promotion gate was
**not** satisfied — no pristine holdout existed when this candidate was selected. This is an
operator-selected candidate, not a statistically promoted model.

**Do not flip `promoted` to True** without a genuine untouched-holdout evaluation. Do not
describe the current model as promoted or validated in any writeup.

**This block is the record of what is deployed.** It is updated only in a `(promotion)` commit,
in the same commit that changes the artifact. If this block and the mounted artifact disagree,
one of them is wrong and production is in an unknown state — stop and reconcile before doing
anything else. See §11.

### 8.2 Prediction path

```
webhook → verify signature → extract ticker/cutoff/information_url
→ fetch disclosure JSON → extract items[*].content
→ build point-in-time context → deterministic parsing → PIT audit
→ construct V3 feature vector → select the 39 artifact features
→ standardize → constrained ridge → clip [0.05, 0.95]
→ calibration_v1 → submit → persist evidence
```

Sign constraints in the constrained ridge (EPS surprise ≥ 0, is_eps_beat ≥ 0, is_eps_miss ≤ 0,
revenue surprise ≥ 0, is_revenue_beat ≥ 0, is_revenue_miss ≤ 0) encode economic priors. Removing
them to improve a fit is a red flag — they exist to prevent the model learning
sign-inverted relationships from sparse, noisy data.

### 8.3 Deployment discipline

- Only an explicitly selected artifact is mounted into Modal.
- Because work lands directly on `main` and Modal reads from the tracked tree, **committing and
  deploying are separated by convention, not by branch topology.** The `(promotion)` tag and
  the §8.1 block are the entire audit trail. Treat them as load-bearing infrastructure.
- Run `check_production` after every deploy and confirm it reports the expected model version,
  ablation, alpha, and feature count. Paste the result into the promotion commit body.
- Verify against the live gates: negative / neutral / positive disclosure cases, parser checks,
  PIT checks, full test suite.
- Baseline: **226 passed** (`uv run pytest`, full suite, as of the repo-hygiene cleanup). A drop
  in count without an explicit removal is a regression. Previously documented here as
  **215 + 7 = 222**; this section had not been updated as tests were added since.
- The webhook has a hard time limit. Any change that adds provider calls to the live path is a
  latency risk. Measure before deploying.
- **The system must always submit something.** A crash produces no submission. Failure paths
  should degrade to a neutral prediction and log loudly, never raise into the webhook handler.

---

## 9. Silent failure modes

These are the bugs that do not raise. They are the most dangerous class in this codebase — the
original constant-score failure ran in production undetected.

### 9.1 What already happened

- The fetcher read metadata instead of `items[*].content`, so the model received strings like
  `"1.0"`, `"ea_ADI_Q3_2026"`, a timestamp — instead of earnings facts.
- Result: `non_zero_fls_feature_count = 0`, every event identical, `fls_ridge_v1` emitting
  raw ≈ 0.4946 / submitted ≈ 0.4815 repeatedly.
- V3-lite then repeated it: raw ≈ 0.4947 / submitted ≈ 0.7046 for unrelated tickers.

Nothing errored. The pipeline was "healthy." Predictions were being submitted. They were
worthless.

### 9.2 Guards that should exist

- **Spread monitor.** Alert if the standard deviation of the last N predictions falls below a
  threshold. Identical predictions across unrelated tickers is the signature.
- **Non-zero feature count.** Alert on `non_zero_fls_feature_count = 0`, and on any event where
  the count is far from the historical distribution.
- **Disclosure sanity.** Assert the fetched facts are prose, not identifiers — minimum length,
  minimum count, rejection of pure version/ID/timestamp strings.
- **Provider failure visibility.** A 402 or rate-limit that returns empty rather than raising
  produces silently zeroed features. Log provider outcomes per event and treat "empty" as
  distinct from "absent."
- **Coverage drift.** Track live parse rates against training coverage (§4.2).

### 9.3 When touching the live path

Never assume ingestion works because the code runs. Verify against
`[LIVE_INPUT]` logs that the actual facts arrived, and against `[V3_FEED]` that context
coverage is non-degenerate.

### 9.4 Undeclared promotion is a silent failure mode

An artifact change that lands on `main` without a `(promotion)` tag looks exactly like any other
commit in `git log`. It deploys anyway. Months later there is no way to attribute a score
movement to it, because nothing in the history marks the date production changed. This belongs
in the same category as the constant-score bug: nothing errors, everything looks healthy, and
the record is wrong.

---

## 10. Terminology — do not conflate

| Term | Meaning |
|------|---------|
| **Raw model score** | Linear model output before calibration |
| **Submitted percentile** | Value actually sent, after calibration |
| **Realized percentile** | Quarter-relative percentile of realized CAR1 |
| **Earnings-surprise percentile** | The benchmark regressor |
| **Delta R2** | Incremental R² beyond the benchmark |
| **Interpretation confidence** | Confidence the parser *understood* a claim — **not** the probability the claim is true |
| **Prediction confidence** | **Does not exist.** No calibrated correctness probability is produced. Do not invent, display, or imply one. |
| **Commit** | A change to the repository. Says nothing about what is deployed. |
| **Promotion** | A change to the deployed artifact. Requires operator sign-off, a `(promotion)`-tagged commit, and a §8.1 update. |
| **`promoted = True`** | A *stronger* claim than promotion: that an untouched-holdout gate was passed. Currently False and should stay False. |

Note the deliberate asymmetry: a `(promotion)` commit means "production changed." It does
**not** mean the `promoted` flag may be flipped. Every deploy is a promotion; almost no deploy
earns `promoted = True`.

---

## 11. Git and repo conventions

### 11.1 Branching and pushing

- **All work commits directly to `main`.** No feature branches, no PRs for routine work.
- **Do not create branches unless there is a stated, specific reason** — an experimental line
  that must not touch the tree Modal reads, or a change genuinely too large to land atomically.
  "Being careful" is not a reason. An unmerged branch is invisible work that silently drifts
  from `main`, and this repo has one operator, so there is no review to gate on.
- **Push to `origin main` after each logical commit.** Do not accumulate unpushed local history
  — to everyone but you, an unpushed commit and uncommitted work are the same thing.
- **Commits must be atomic and independently revertible.** `main` is the only line of history,
  so `git revert` is the only rollback mechanism. A commit that cannot be reverted cleanly on
  its own is too big.
- **Never force-push `main`. Do not rewrite published history.** To undo, commit a revert.
- **One concern per commit.** Do not mix an evaluation-methodology fix with a model change — it
  becomes impossible to attribute a score movement, and impossible to revert one without the
  other.

### 11.2 A commit is not a promotion

Since Modal mounts from the tracked artifact, "on `main`" and "in production" are nearly the
same thing. The `(promotion)` tag is the only thing separating them in the record.

- **Default: no tag.** A normal commit — including one that adds a candidate, runs a sweep,
  writes an evaluation script, or scores something — **does not change what production
  submits.** Most commits are this.
- **A commit that changes the deployed artifact must end its subject line with `(promotion)`.**
  This covers: swapping the mounted artifact, changing ablation / model type / alpha / feature
  set, changing or removing the calibration path, and any change to the live prediction path
  in §8.2 that alters submitted values.
- **A `(promotion)` commit does nothing else.** No refactors, no drive-by fixes, no unrelated
  file moves. If it needs a prerequisite change, that is a separate untagged commit first.
- **A `(promotion)` commit must update §8.1** in the same commit, so the document and the
  deployed state move together.
- **The tag is reserved.** Do not use it decoratively, do not use it on a commit that "prepares
  for" a promotion, and do not use it because a change feels important. It means exactly one
  thing: what production submits is now different.

A promotion commit body must state:

1. Artifact identity — ablation, model type, alpha, feature count, calibration.
2. What it replaces.
3. The evidence, with partition labels, raw and submitted Delta R2 separately, `n`, and the
   null expectation (§13).
4. The operator who signed off.
5. The state of the `promoted` flag and why (see §8.1 — it stays False absent a real holdout).
6. Confirmation `check_production` ran after deploy, with its reported values.

### 11.3 Staging discipline

- **Never `git add .` or `git add -A`.** Stage explicit paths only — a blanket add is how
  unrelated or unreviewed files (data dumps, stray gitlinks, scratch output) end up committed.
  This matters more now that there is no branch or PR between staging and deployed state.
- Run `git status` and read `git diff` (or `git diff --cached`) before every commit. Confirm
  exactly what's staged matches what you intend.
- Read git's warnings rather than letting them scroll past — a gitlink notice, a CRLF warning,
  a "not tracking" message is often the only signal something is about to go in wrong.
- Never commit API keys, `.env` files, Modal tokens, or downloaded evidence bundles.
- Never commit large data files (`data/`, archives, evidence dumps).
- Especially careful with: `artifacts/v3_lite_candidate.json`, the calibration path, anything
  under the webhook handler. A commit touching these is almost certainly a promotion — if you
  are about to commit one of these without the `(promotion)` tag, stop and check why.
- `scripts/sweep_v3_lite_official_candidates.py` is **read-only** by design. Keep it that way.

### 11.4 Commit messages

State what changed and what it does *not* change. The "does not change" line is not boilerplate
— it is the assertion that this commit is not a promotion.

Routine commit:

```
Rank sweep candidates by raw Delta R2 instead of submitted

Submitted ΔR2 includes the CDF transform, which reduces ΔR2 on both
Q1 (-0.0020) and Q2 (-0.0042). Ranking on it selects for compatibility
with a transform we have evidence against.

Does not change the deployed artifact or calibration path.
```

Promotion commit:

```
Swap deployed artifact to fls_plus_reasoning, alpha=100 (promotion)

Replaces: fls_plus_revenue / constrained_ridge / alpha=100 / 39 features
With:     fls_plus_reasoning / constrained_ridge / alpha=100 / 41 features
Calibration: unchanged (calibration_v1)

Evidence: 2026Q1 (selection quarter — contaminated, see §1.2)
  raw ΔR2 0.007188 / submitted ΔR2 0.006955, n=2390,
  null expectation (1-0.29)/2387 = 0.00030
  Ranked 3rd of 144 candidates on raw; top ten within noise.

promoted remains False — no untouched holdout exists (§8.1).
Operator sign-off: Zaid, 2026-08-28.
check_production: v3_lite_operator_2026_08_28 / fls_plus_reasoning /
  alpha 100.0 / 41 features — matches intended.

§8.1 updated in this commit.
```

---

## 12. Commands

```bash
# Tests
uv run pytest

# Modal logs
uv run modal app logs explaining-markets-starter --since 2h --search LIVE_INPUT
uv run modal app logs explaining-markets-starter --since 2h --search V3_FEED
uv run modal app logs explaining-markets-starter --since 2h --search V3_LITE_MODEL
uv run modal app logs explaining-markets-starter --since 2h --search V3_PREDICT

# Evidence
mkdir -p data/live_eval/evidence
uv run modal volume get em-v3-data evidence data/live_eval/evidence

# Evaluation
uv run python scripts/evaluate_recent_from_archive.py
uv run python scripts/analyze_recent_live_results.py
uv run python scripts/compare_v3_lite_official_score.py
uv run python scripts/sweep_v3_lite_official_candidates.py

# Prediction attribution
uv run python scripts/inspect_v3_prediction.py
uv run python scripts/render_prediction_dashboard.py

# Promotion history — every time production changed
git log --oneline --grep='(promotion)' --fixed-strings
```

Log categories: `[LIVE_INPUT]` (what production received), `[V3_FEED]` (context coverage),
`[V3_LITE_MODEL]` (version/ablation/raw/submitted/calibration), `[V3_PREDICT]` (final details).

---

## 13. Reporting standards

Any result stated in a commit, comment, or summary must include:

1. **Raw and submitted Delta R2 separately.** Never one number.
2. **n**, and the null expectation `(1 - R2_benchmark)/(n - k)`.
3. **Which partition** produced it, and whether that partition is pristine, selection-used, or
   legacy.
4. **What touched the fit** that produced the evaluated coefficients.
5. For sweeps: **how many candidates** were evaluated.

Language discipline:

- "Delta R2 was positive" — meaningless without §13.2. Do not write it.
- "Out-of-sample" — only if the evaluation window is disjoint from everything that touched the
  fit, including scalers and calibrators.
- "Validated" / "promoted" — only if a genuine untouched holdout was used. Currently nothing is.
  Note that a `(promotion)`-tagged commit does not make anything "validated"; it records a
  deployment, not a passed gate (§10).
- "The model uses X" — only if X is in the artifact's `feature_names`.
- "Deployed" / "in production" — only if it landed in a `(promotion)` commit and
  `check_production` confirmed it.

Do not report an improvement without stating what it was measured against and what else moved.

---

## 14. Pre-flight checklist

Before proposing any change, answer:

- [ ] Does this use a 2026Q3 number to choose between specifications, hyperparameters, or
      features (rather than just scoring already-frozen predictions against it)? → **stop**,
      see §0.1 / §1.2
- [ ] Does it fit anything on live events? → **stop**
- [ ] Does it evaluate an artifact on data that touched its fit? → **stop**
- [ ] Does it weaken the point-in-time audit? → **stop**
- [ ] Does it add a feature correlated with earnings surprise? → likely worthless for Delta R2
- [ ] Is the reported gain larger than `(1 - R2_bench)/(n - k)`?
- [ ] Are raw and submitted Delta R2 reported separately?
- [ ] How many candidates were compared to find this?
- [ ] Does it touch the live path? → latency, ingestion verification, always-submit guarantee
- [ ] Does it change what `check_production` validates?
- [ ] **Does it change what production submits?** → it is a promotion: tag the subject line
      `(promotion)`, update §8.1, include the §11.2 body fields, run `check_production` after.
- [ ] **If it is not a promotion, does the commit message say so explicitly?**
- [ ] Is this landing on `main` — and if a branch is being created instead, what is the stated
      reason? → default is no branch (§11.1)
- [ ] Is everything staged by explicit path, with `git diff --cached` read? → no `git add .`
- [ ] Do the 226 tests still pass?

When uncertain, produce the analysis and the caveat. **Do not resolve ambiguity in the direction
that makes a result look better.** The purpose of this document is to prevent a system that
scores well on its own diagnostics and poorly on the competition.