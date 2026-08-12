# Prediction-Time Information Audit

**Status:** Research/audit only. `predict.py`, `modal_app.py`, `model.py`, and the
production submission pipeline were **not modified** in the course of this
document. All conclusions below are traced to specific files, live API
responses, or direct inspection of the 6,287 real historical records
downloaded in `data/historical/` — no assumptions were substituted for
evidence. Where evidence was insufficient, the item is marked `UNKNOWN`
rather than assumed safe, per the task's explicit conservatism requirement.

**Evidence sources used:**
- `predict.py`, `src/explaining_markets/{historical,features,model,backtest}.py` (current implementation)
- `src/explaining_markets/{config,client,event_utils,webhook_verification}.py`, `modal_app.py` (plumbing, read-only)
- `examples/src/examples/{client,schemas,archive,scoring,summary}.py`, `examples/notebooks/*.ipynb`, `examples/data/README.md`
- The live OpenAPI spec at `https://explainingmarkets.ai/openapi.json` (fetched fresh for this audit — `WebhookPayload`, `CalendarEvent`, `DisclosureBundle`, `ArchiveFile` schemas, checked for **full property lists**, not just required fields)
- Direct field-level inspection of all 6,287 records across the three downloaded sealed-quarter archive files (`data/historical/EARNINGS_RELEASE_{2025Q4,2026Q1,2026Q2}.jsonl.gz`)
- `data/historical/PROVENANCE.json`, `HISTORICAL_DATA_INVESTIGATION.md`/`competition.md` (prior investigation docs, used as leads, not as unverified fact)

---

## 1. Executive Summary

The live prediction pipeline (`predict.py`) currently has access to exactly
**one substantive information source at prediction time**: the event's own
`information_url` → `DisclosureBundle.items[].content` (the ten fact
sentences), plus trivial identity/scheduling metadata (`event_id`, `ticker`,
`event_type`, `event_datetime`, `prediction_deadline`). This is confirmed
directly against the live OpenAPI spec's `WebhookPayload` schema — it has
**exactly eight properties**, and no more.

Two consequential, evidence-backed findings emerged that were not previously
established with this precision:

1. **`knowledge_cutoff` is confirmed ABSENT from the live webhook payload.**
   The full `WebhookPayload.properties` list (fetched fresh from the live
   OpenAPI spec for this audit) is `id, event_id, event_type,
   timing_category, event_datetime, focal_assets, information_url,
   prediction_deadline` — no `knowledge_cutoff` field, required or optional.
   It **is** present on `GET /events` (`CalendarEvent.knowledge_cutoff`,
   required) and on every one of the 6,287 real archive records inspected.
   `predict.py` today has no way to read its own event's cutoff without an
   additional `GET /events` lookup it does not currently make.
2. **The historical archive's derived fields (`car1`, `earnings_surprise`)
   are computed in a single, late, batch process (`returns_computed_at`),
   and the disclosure facts on backfilled archive records are generated with
   a median 137-day (min 10, max 274-day) lag after `event_datetime`.** This
   confirms, from real data rather than inference, that these fields are
   unambiguously post-event and that the archive's `generated_at` timestamp
   on old records reflects *backfill* timing, not original live-delivery
   timing (the live path, bound by the platform's own 5-minute prediction
   window, must generate the disclosure far faster — see §2).

Of the (see §9 for exact count) candidate information sources audited: a
small number are safely usable **today** without any further work (the
disclosure text and trivial event metadata); a meaningful second group
(same-ticker prior-event outcomes) is **conditionally safe** but requires
either an explicit point-in-time timestamp check or an external data source,
because our own `/archive` access lags real time by roughly a quarter (see
§4); the realized outcome fields and everything derived from them
cross-sectionally (percentile ranks, `baseline_predictions`) are **never**
safe as live features; and a large group of externally-sourceable
information (market prices, analyst estimates, sector data, filings) is
simply **not present anywhere in the two repositories** and would need to be
independently obtained and timestamp-validated before any use.

---

## 2. Temporal Model of an Event

Reconstructed directly from the real archive data (not assumed), using the
`data/historical/EARNINGS_RELEASE_2025Q4.jsonl.gz` record `ea_PEP_Q3_2025`
as the concrete evidence trail:

```
knowledge_cutoff            2025-10-08T20:00:00Z   (CalendarEvent / archive field; CONFIRMED
                                                     absent from live WebhookPayload — see §1)
        │
        ▼
[window_start_date]         2025-10-08              (CAR1 return-window open)
        │
event_datetime               2025-10-09T10:00:00Z   (the earnings call / release itself)
        │
[window_end_date]           2025-10-09              (CAR1 return-window close)
        │
        ▼  (market absorbs the news; r_i, r_m observed over the window)
        │
        ▼  --- EVERYTHING BELOW THIS LINE IS POST-EVENT, REALIZED, DERIVED ---
        │
metrics.earnings_surprise    computed from reported-vs-consensus, same batch
   .surprise / .price_date   as event_returns (no separate pre-event timestamp
                              exists anywhere in the schema — see §5)
        │
event_returns.<T>.car1       = r_i - r_m over [window_start_date, window_end_date]
   (r_i, r_m, n_obs)         (formula CONFIRMED by direct arithmetic check: 
                              0.04227888... - (-0.00346584...) = 0.04574472...,
                              matches the record's own car1 to full precision)
        │
returns_computed_at          2026-07-10T22:18:51Z   (batch computation, ~9 months
                              after event_datetime for this record)
        │
disclosure.generated_at      2026-07-10T22:18:51Z   (archive backfill timestamp —
                              NOT the live-delivery generation time; median lag
                              across all 6,287 records is 137 days, min 10, max 274)
        │
        ▼
Quarter sealed (built_at)    2026-07-23T13:31:25Z   (ArchiveFile.built_at)
        │
        ▼
We download it               2026-08-11             (this session)
```

**Critical interpretive note, evidence-based, not assumed:** the 10–274 day
`disclosure.generated_at` lag applies to how the *archive* was backfilled for
already-realized historical quarters. It must **not** be read as "the
platform takes months to produce a disclosure" — the starter's own
documentation (`README.md`, `docs/advanced.md`) establishes a **5-minute**
prediction window that opens at ACK, meaning the live `information_url` must
resolve to a usable `DisclosureBundle` within minutes of the event firing.
The archive's `generated_at` values are an artifact of the backfill batch
process, not evidence about live-path latency. Where this document says
"UNKNOWN" about a live timestamp, it is because we have not independently
observed a real live delivery — not because the batch-lag numbers above
apply to it.

---

## 3. Complete Candidate-Source Inventory (Task 1)

Every item requested in Task 1, audited individually. "Current event" means
the field's value *for the event actually being predicted*; "prior event"
means the same field on an **earlier** event for the same ticker (relevant to
§5).

### 3.1 Webhook / live envelope fields

| # | Candidate | Where it comes from | In historical data? | In live webhook? | Externally obtainable? |
|---|---|---|---|---|---|
| 1 | `id`, `event_type`, `timing_category`, `event_datetime`, `focal_assets`, `prediction_deadline` | `WebhookPayload` (OpenAPI, confirmed full property list) | `event_type`, `event_datetime`, `focal_assets` present as archive fields; `id`/`prediction_deadline` are webhook-only, not archived | **Yes** — these are exactly the webhook's required fields | N/A, platform-native |
| 2 | `information_url` | `WebhookPayload` | Not archived (archive inlines `disclosure` directly instead of a URL) | **Yes** | N/A |
| 3 | `knowledge_cutoff` (on the webhook itself) | — | — | **CONFIRMED ABSENT** from `WebhookPayload.properties` (fetched fresh; 8 properties total, none named `knowledge_cutoff`) | N/A |
| 4 | `knowledge_cutoff` (via a separate `GET /events` call keyed by `event_id`) | `CalendarEvent.knowledge_cutoff` (required field) | N/A (different endpoint) | Available via an **additional API call that `predict.py` does not currently make** | N/A |
| 5 | `knowledge_cutoff` (as recorded in the archive) | Archive record top-level field | **Present on all 6,287/6,287 records inspected** | N/A (archive-only) | N/A |

### 3.2 Disclosure (`information_url` → `DisclosureBundle`)

| # | Candidate | Where it comes from | In historical data? | In live webhook path? |
|---|---|---|---|---|
| 6 | `DisclosureBundle.schema_version`, `.event_id` (envelope identity) | `DisclosureBundle` schema | Present (`schema_version`, `event_id` on all 6,287 records) | Yes, same schema applies to the live GET |
| 7 | `DisclosureBundle.generated_at` (live) | Same schema | Present, but reflects **archive backfill time**, not live time (see §2) — do not reuse the archive's lag statistics for the live path | Present in schema; actual live-path value **not independently observed** in this audit |
| 8 | `disclosure.items[].content` — the fact sentences | `DisclosureItem.content` (`kind="facts"`, `source="earnings_call"`) | Present on 6,282/6,287 records (5 records carry no facts item at all) | Yes — this is what `features.py::extract_features` already consumes |
| 9 | `disclosure.items[].kind` / `.source` / `.media_type` | Same | Present | Yes |
| 10 | Guidance text embedded inside a fact sentence (e.g. "Full-year revenue guidance was raised to...") | Same content field, not a separate structured field | Present as free text within `content`, confirmed by direct inspection of multiple sample facts | Yes, same as #8 — it is a subset of the disclosure text, not a distinct source |

### 3.3 Historical archive as a bulk capability

| # | Candidate | Notes |
|---|---|---|
| 11 | `GET /archive` bulk access (generic) | Confirmed working (this session). Returns only **sealed** quarters; unsealed quarters (e.g. `2026Q3` at time of retrieval) are explicitly excluded from our download. **Sealing lag observed:** `2026Q2` (`event_datetime_max` = 2026-06-30) was `built_at` 2026-08-11 — roughly six weeks after quarter-end. This lag is the central constraint on using "the most recent quarter" for live features (§4). |

### 3.4 Realized/derived fields on the CURRENT event (the one being predicted)

| # | Candidate | Confirmed present (archive) | Live availability |
|---|---|---|---|
| 12 | `event_returns.<ticker>.car1` | 6,286/6,287 (1 record has `return_status` other than `ok`) | **Never** — this is the value being predicted |
| 13 | `event_returns.<ticker>.{r_i, r_m, window_start_date, window_end_date, n_obs}` | Same coverage as #12 | **Never** — these are the raw legs of the outcome itself |
| 14 | `metrics.earnings_surprise.surprise` | 6,286/6,287 (`surprise_status="ok"`: 6,145; `"missing_eps"`: 141; other: 1) | **Never** — requires the actual reported result, which doesn't exist until after the event |
| 15 | `metrics.earnings_surprise.price_date` | Same coverage as #14 | **Never** — metadata of a post-event computation |
| 16 | `baseline_predictions` (two reference LLM agents' own predictions for **this same event**) | 6,075/6,287 | **Never**, and not merely "post-event" — these are other agents' *predictions*, not facts, disclosed only retrospectively for benchmarking. Using them would also be circular even if they were somehow available live. |
| 17 | `returns_computed_at`, `status`, `return_status`, `surprise_status` | Present on all records | **Never** as features — these are provenance/validity markers that only exist once the outcome has been computed; their mere presence signals "this event is already resolved," which is itself post-event information |

### 3.5 Prior events for the same ticker (a different, earlier event)

| # | Candidate | Availability discussion |
|---|---|---|
| 18 | Prior event's `car1` (same ticker, `event_datetime` strictly before the current event) | The **value** was realized and, in principle, publicly derivable (it's a market-adjusted return, computable from public price data) once that prior event's own return window closed — normally 1 trading day after that prior event. **But our own copy of it, via `/archive`, is only available once that prior event's quarter is sealed — and sealing lags roughly six weeks past quarter-end** (§3.3). This is a hard practical constraint: a prior event from *earlier in the current quarter* is very likely **not yet in our downloaded archive** at the time we'd want to use it live. |
| 19 | Prior event's `earnings_surprise.surprise` | Same availability profile and same sealing-lag constraint as #18. |
| 20 | Prior event's `baseline_predictions` | Same sealing-lag constraint as #18/#19, **plus** the same "other agents' predictions, not facts" objection as #16 — discouraged even where technically obtainable. |
| 21 | `GET /events` calendar — a prior event's mere **existence** (`event_id`, `event_type`, `event_datetime`, `knowledge_cutoff`, `focal_assets` — no outcome fields at all) | This endpoint's own description states both "forward-looking scheduled" and "realized" events appear — and it carries none of the outcome/derived fields, only identity/scheduling. This means **counting** or **listing** a ticker's prior events is plausibly available in near-real-time, without the archive's sealing lag, because no outcome data is involved. This is *not independently confirmed* for the specific case of "does `/events` retain arbitrarily old realized entries," so it is marked conditionally in §6, not unconditionally safe. |

### 3.6 External / not present in either repository

None of the following appear anywhere in `src/explaining_markets/`,
`examples/src/examples/`, any notebook, any test fixture, or any of the
6,287 real archive records — confirmed by targeted `grep` across both
repositories in this and prior audit sessions, and by the full key-inventory
dump of every field actually present in the real archive (§ field inventory
below has exactly 13 top-level keys, 7 `event_returns` leg keys, 1 `metrics`
key, 3 `earnings_surprise` keys, 4 `disclosure` keys, 8 `disclosure.items[]`
keys — nothing sector/price/estimate/filing-related among any of them).

| # | Candidate | Present in EM data? | Externally obtainable? |
|---|---|---|---|
| 22 | Raw market prices for the focal ticker | No | Yes, from a market-data vendor |
| 23 | Market/index returns (benchmark leg) | No (only the already-netted `r_m` inside a *realized* record, never as a standalone live series) | Yes, from a market-data vendor |
| 24 | Volatility (any window) | No | Derivable from #22 if sourced |
| 25 | Sector/industry classification | No | Yes, e.g. GICS mapping from a reference-data vendor |
| 26 | Individual analyst estimates | No | Yes, from an estimates vendor — but timestamp reliability varies by vendor |
| 27 | Earnings consensus (aggregated) | No | Yes, same caveat as #26 |
| 28 | Structured guidance history (separate from the disclosure text) | No | Yes, would need to be built from filings/press releases over time |
| 29 | Company filings (10-K/10-Q/8-K, e.g. via SEC EDGAR) | No | Yes — EDGAR acceptance timestamps are an authoritative public record, among the more reliably timestamped external sources available |
| 30 | Raw earnings-call transcripts (full text, any source) | **No — explicitly not distributed** (`examples/data/README.md`: "Real transcripts are licensed and are not distributed with this repo") | Possibly, via a licensed transcript provider — but see §6 for a rules-interpretation caveat this document cannot resolve |
| 31 | Any other external/public source (news wires, social sentiment, etc.) | No | Case-by-case; publication-timestamp reliability varies enormously by source and must be individually verified, never assumed |

**Field-inventory evidence for §3.6's "confirmed absent" claim** (direct dump
across all 6,287 real records, this audit session):

```
top-level keys:            event_id, event_type, timing_category, event_datetime,
                            knowledge_cutoff, focal_assets, status, return_status,
                            event_returns, returns_computed_at, metrics,
                            baseline_predictions, disclosure
event_returns.<T> keys:    window_start_date, r_m, car1, n_obs, return_status,
                            r_i, window_end_date
metrics keys:               earnings_surprise
earnings_surprise keys:     surprise, surprise_status, price_date
disclosure keys:            schema_version, event_id, generated_at, items
disclosure.items[] keys:    id, kind, source, media_type, content, url, bytes, sha256
```

No sector, price, estimate, consensus, or filing field exists anywhere in
this list.

---

## 4. Leakage Audit — the Most Dangerous Variables (Task 3)

For each variable, the exact temporal sequence position (from §2) and why
it is or is not usable — not merely asserted, but placed on the timeline.

### `event_returns.<ticker>.car1` (current event)
**Position on the timeline:** computed *after* `window_end_date` (the day of
`event_datetime`), then batch-recorded at `returns_computed_at` — confirmed
9 months after the event for the sampled record, and by construction always
strictly after the event's own return window closes.
**Why prohibited as a feature:** it is arithmetically the quantity being
predicted (`car1` → within-quarter percentile rank → the competition's
scored target). Confirmed by direct recomputation: `r_i - r_m` for the
sample record equals `car1` to full floating-point precision. There is no
sequence of events by which this value could be known before `event_datetime`
plus one trading day, let alone before `knowledge_cutoff`.
**Classification: `POST_EVENT`.**

### `event_returns` (the full sub-object: `r_i`, `r_m`, window dates, `n_obs`)
**Position:** same as `car1` — these are `car1`'s own input legs, computed
over the same post-event window.
**Why prohibited:** even though `r_i`/`r_m` are not literally `car1`, they
are its arithmetic components; exposing them to a model is equivalent to
exposing `car1` itself (an adversarial or even an entirely innocent linear
model could reconstruct `car1` exactly from `r_i - r_m`).
**Classification: `POST_EVENT`.**

### `metrics.earnings_surprise` (current event)
**Position:** requires the actual *reported* result, which does not exist
until the earnings release happens at `event_datetime` — by definition
after `knowledge_cutoff`. It is recorded in the same post-hoc batch as
`event_returns` (`returns_computed_at`), and **no separate, earlier
timestamp for the underlying consensus estimate exists anywhere in the
schema** (confirmed — `earnings_surprise` has exactly three keys: `surprise`,
`surprise_status`, `price_date`; none of them date a pre-event consensus).
**Why prohibited as a live feature for the current event:** unconditionally
post-event — the reported half of "surprise" cannot exist before the event.
**Classification: `POST_EVENT`** for the current event. (For a *prior*
event of the same ticker, see §3.5/§6 — a different, conditional case.)

### `baseline_predictions`
**Position:** these are the two reference LLM agents' own *predictions* for
the identical event being scored — logically parallel to, not upstream of,
our own prediction. They are only disclosed once the archive backfills the
quarter, long after scoring.
**Why prohibited, beyond "post-event":** even setting timing aside, using
another agent's prediction for the *same event* as an input to your own
prediction for that *same event* is circular by construction, not merely
late. **Classification: `POST_EVENT`, and additionally flagged `NEVER USE`
even in principle, independent of timing, for the current event.** For a
prior event of the same ticker, it becomes merely `HISTORICAL_ONLY` and
discouraged (§3.5, item 20) rather than logically circular, but is still
gated by the same sealing lag as everything else in §3.5.

### `returns_computed_at` / `disclosure.generated_at` (on archive records)
**Position:** these are *provenance timestamps*, not information content.
**Why they matter for the audit, not as leakage risks themselves:** they are
the direct evidence this document uses to establish that `car1` and
`earnings_surprise` are late-computed (§2) — their role here is diagnostic,
not as candidate features. Using either timestamp *as a feature value*
would be nonsensical (it doesn't describe the company, only our own
data pipeline's processing schedule), and in any case, both timestamps only
exist on already-realized records. **Classification: `POST_EVENT`** (not
applicable as a feature; included here only because Task 3 named them
explicitly).

### Quarter cross-sectional percentile ranks (`y` / `surprise_pct`, from `backtest.py`/`examples.scoring`)
**Position:** these require the **entire quarter's** realized outcomes —
strictly the most "future" quantity in this whole audit, since a single
event's percentile rank cannot even be computed until every other event in
the same quarter has also resolved (`percentile_ranks()` requires the full
cross-section). This is worse than ordinary post-event leakage: it requires
information from *other, potentially later* events in the same quarter, not
just the current event's own outcome.
**Classification: `POST_EVENT`**, and the single most severe leakage vector
in this entire inventory if ever mistaken for a feature.

---

## 5. Historical-Company Feature Audit (Task 4)

For each requested rolling/aggregate feature, the exact non-leaking
construction rule for event `E` (ticker `t`, `event_datetime = T`,
`knowledge_cutoff = C`):

> **General rule, applies to every feature below:** only use events for the
> **same ticker** with `event_datetime < T` (strictly before — never
> including `E` itself), **and** whose own outcome (`car1`/`earnings_surprise`)
> must itself already have been realized *and available to us* by `T` — not
> merely chronologically earlier. "Available to us" is the binding
> constraint in practice, because of the archive's sealing lag (§3.3/§3.5):
> a prior event from earlier in the *same, still-open* quarter is very
> likely not yet in our downloaded archive.

| Feature | Exact rule |
|---|---|
| `previous_car1` | `car1` from the single most recent prior event for ticker `t` with `event_datetime < T`. **Never** the current event. |
| `rolling_mean_car1(N)` | Mean of `car1` over the `N` most recent prior events for `t` with `event_datetime < T`. Undefined (not zero-filled) if fewer than `N` qualifying prior events exist. |
| `rolling_car1_volatility(N)` | Sample standard deviation of the same `N`-event window as `rolling_mean_car1`. Same "undefined if insufficient history" rule. |
| `previous_earnings_surprise` | `earnings_surprise.surprise` from the most recent prior event for `t` with `event_datetime < T` **and** `surprise_status == "ok"` on that prior record — skip (do not substitute) prior events with `"missing_eps"` or other non-`"ok"` status rather than treating them as zero. |
| `rolling_mean_surprise(N)` | Mean `surprise` over the `N` most recent qualifying (`surprise_status=="ok"`) prior events. |
| `number_of_previous_positive_surprises(N)` | Count of prior qualifying events (as above) within the `N`-event or time-bounded window where `surprise > 0`. |
| `historical_reaction_asymmetry` | E.g. `mean(car1 | car1 > 0 over prior events) - abs(mean(car1 | car1 < 0 over prior events))`, computed only over prior, already-resolved events for `t`. |
| `number_of_prior_earnings_events` | Count of **any** prior calendar entries for `t` with `event_datetime < T`, **regardless of whether their outcome is known to us yet** — this one does **not** require `car1`/`surprise` at all, only event *existence*, which is plausibly obtainable from `GET /events` without the archive's sealing lag (§3.5, item 21). This is the one feature in this family with a materially different (better) availability profile — flagged explicitly so it is not lumped in with the others. |

**Can any of these use an event whose outcome was not yet publicly known at
`E`'s knowledge_cutoff?** No — by the general rule above, every feature in
this family excludes any prior event whose own `car1`/`surprise` had not yet
been realized and made available to us by `T`. In our **current, practical**
setup, "made available to us" is bottlenecked by `/archive`'s sealing lag,
which is stricter than "realized" — so the correct, conservative
implementation is to check availability against **our own downloaded
archive's `built_at`/coverage**, not merely against the prior event's
`event_datetime`, until/unless a lower-latency source (independent market
data, or `/events`-only existence counts) is substituted.

---

## 6. Feature Eligibility Table (Task 5)

| Feature | Source | Historical availability | Live availability | Required timestamp | Point-in-time safe? | Classification | Leakage risk | Implementation difficulty | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Webhook envelope (`event_type`, `event_datetime`, `focal_assets`, `prediction_deadline`, `id`) | `WebhookPayload` | Partial (archive lacks `id`/`prediction_deadline`) | Yes | None needed — identity/scheduling only | Yes | **SAFE_LIVE** | None | Trivial | Already used by `predict.py` |
| `information_url` | `WebhookPayload` | N/A | Yes | None | Yes | **SAFE_LIVE** | None | Trivial | Access mechanism, not content |
| `knowledge_cutoff` (on the webhook itself) | — | — | **Confirmed absent** | — | N/A | **UNKNOWN** | N/A | N/A | Full property list checked; not present, required or optional |
| `knowledge_cutoff` (via `GET /events`) | `CalendarEvent` | N/A | Yes, but requires an extra call `predict.py` doesn't make today | None (it IS the cutoff) | Yes | **SAFE_LIVE** (once implemented) | None | Low (one more GET call) | Addressable gap, not a blocker |
| `knowledge_cutoff` (archived) | Archive record | 6,287/6,287 | N/A | N/A | N/A | **HISTORICAL_ONLY** | None | N/A | Audit/backtest metadata only |
| `disclosure` envelope identity (`schema_version`, `event_id`) | `DisclosureBundle` | Present | Yes | None | Yes | **SAFE_LIVE** | None | Trivial | |
| `disclosure.generated_at` (live) | `DisclosureBundle` | Present but reflects backfill, not live timing | Present in schema; live value not independently observed | Would need to be `< knowledge_cutoff`? Unclear if even relevant, since content is exempted anyway | Not established | **UNKNOWN** | Low (metadata, not content) | N/A | Do not reuse archive lag stats for live path |
| `disclosure.items[].content` (facts) | `DisclosureBundle` | 6,282/6,287 | Yes | None — explicitly exempted from the cutoff by competition rules ("describes the event itself") | Yes, by rule | **SAFE_LIVE** | None (by rule) | Already implemented | Core input to `features.py` today |
| Guidance text within facts | Same | Same | Same | Same | Yes | **SAFE_LIVE** | None | Already implemented (implicit, via keyword match) | Subset of the above, not separate |
| Bulk `/archive` access | `GET /archive` | This IS the source | No (sealed-quarter lag) | N/A | N/A | **HISTORICAL_ONLY** | None (used only for backtesting) | Already implemented (`historical.py`) | ~6-week sealing lag confirmed |
| `event_returns.car1` (current event) | Archive | 6,286/6,287 | Never | N/A | No | **POST_EVENT** | Severe (is the label) | N/A | Never a feature |
| `event_returns.{r_i,r_m,window_*}` (current event) | Archive | Same coverage | Never | N/A | No | **POST_EVENT** | Severe (reconstructs the label) | N/A | Never a feature |
| `earnings_surprise.surprise` (current event) | Archive | 6,286/6,287 | Never | N/A | No | **POST_EVENT** | High | N/A | Requires the actual reported result |
| `earnings_surprise.price_date` (current event) | Archive | Same | Never | N/A | No | **POST_EVENT** | Low (metadata) | N/A | Provenance of a post-event calc |
| `baseline_predictions` (current event) | Archive | 6,075/6,287 | Never | N/A | No | **POST_EVENT / NEVER USE** | Severe + circular | N/A | Other agents' predictions for the same event |
| `returns_computed_at`/`status`/`*_status` flags (current event) | Archive | All records | Never | N/A | No | **POST_EVENT** | Moderate (signals resolution) | N/A | Provenance only |
| Prior event's `car1` (same ticker) | Archive | Yes, subject to sealing lag | Conditional | `event_datetime(prior) < T` AND archive coverage includes it | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate (needs availability check against our download, not just chronology) | See §5 |
| Prior event's `earnings_surprise` (same ticker) | Archive | Same | Conditional | Same, plus `surprise_status=="ok"` on that record | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate | See §5 |
| Prior event's `baseline_predictions` (same ticker) | Archive | Same | Conditional but discouraged | Same as above | Conditionally, but not recommended | **HISTORICAL_ONLY** | Moderate (other agents' output) | Low, but low value | Discouraged even where technically available |
| Prior event existence/count (`GET /events`) | `CalendarEvent` | N/A (different endpoint) | Plausibly yes, not independently confirmed for arbitrarily old entries | `event_datetime(prior) < T` only — no outcome needed | Yes, if confirmed | **SAFE_LIVE** (pending confirmation) | Low | Low | Best-availability item in the rolling-feature family |
| `previous_car1` | Derived (§5) | Buildable from archive | Same constraint as "prior event's car1" | Same | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate | |
| `rolling_mean_car1(N)` | Derived (§5) | Buildable | Same | Same | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate | |
| `rolling_car1_volatility(N)` | Derived (§5) | Buildable | Same | Same | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate | |
| `previous_earnings_surprise` | Derived (§5) | Buildable | Same | Same, plus status filter | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate | |
| `rolling_mean_surprise(N)` | Derived (§5) | Buildable | Same | Same | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate | |
| `number_of_previous_positive_surprises(N)` | Derived (§5) | Buildable | Same | Same | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate | |
| `historical_reaction_asymmetry` | Derived (§5) | Buildable | Same | Same | Conditionally | **SAFE_IF_TIMESTAMPED** | Moderate | Moderate-high | |
| `number_of_prior_earnings_events` | Derived (§5) | Buildable | Better profile (existence only, no outcome) | `event_datetime(prior) < T` only | Yes, if `/events` confirmed | **SAFE_LIVE** (pending confirmation) | Low | Low | |
| Raw market prices | External | No | No | Must prove `price_timestamp < event_datetime` | Not established | **UNKNOWN** | Depends on source, generally low once dated | Moderate | Not present in either repo |
| Market/index returns | External | No | No | Same | Not established | **UNKNOWN** | Low once dated | Moderate | Not present |
| Volatility (derived) | External | No | No | Same, derived from prices | Not established | **UNKNOWN** | Low once dated | Moderate | Not present |
| Sector/industry classification | External | No | No | Low ambiguity (rarely changes) but still needs a snapshot date | Not established | **UNKNOWN** | Low | Low-moderate | Not present |
| Individual analyst estimates | External | No | No | Must prove `estimate_timestamp <= knowledge_cutoff` | Not established | **UNKNOWN** | High if unverified | High | Not present; hardest to timestamp reliably |
| Earnings consensus (aggregated) | External | No | No | Same as above | Not established | **UNKNOWN** | High if unverified | High | Not present |
| Structured guidance history | External | No | No | Must prove pre-cutoff publication | Not established | **UNKNOWN** | Moderate | Moderate-high | Not present |
| Company filings (e.g. SEC EDGAR) | External | No | No | EDGAR acceptance timestamp (authoritative once sourced) | Conditionally, once sourced correctly | **UNKNOWN** (not yet sourced) | Low once dated correctly | Moderate | Best-quality timestamp among external sources, but not sourced yet |
| Raw earnings-call transcripts | External / not distributed by EM | No (synthetic sample only) | No | Depends on provider; also a rules-interpretation question | Not established | **UNKNOWN** | Rules ambiguity + timing | High | See §6 caveat below |
| Any other external/public source | External | No | No | Must prove publication predates cutoff, case by case | Not established | **UNKNOWN** | Variable, often high | Variable | Never assume; verify per source |
| Quarter percentile rank `y` (current event) | `backtest.py`/`examples.scoring` | Computed offline from archive | Never | N/A — requires the whole quarter | No | **POST_EVENT** | Most severe in this document | N/A | Requires other, potentially later events too |
| Quarter percentile rank `surprise_pct` | Same | Computed offline | Never | N/A | No | **POST_EVENT** | Severe | N/A | Benchmark only, never a feature |

**Rules-interpretation caveat on raw transcripts:** the FAQ/docs establish
that "the event materials delivered by the platform... describe the event
itself and are of course fair game" — this document cannot establish
whether an **externally, independently sourced** full transcript of the
same call (faster or more complete than the platform's own facts) falls
under the same exemption, or whether the exemption is specific to
platform-delivered materials only. This is marked `UNKNOWN` rather than
assumed either way, and is flagged as an open question in §8.

---

## 7. Final Live Feature Contract (Task 6)

**SAFE TO USE NOW:**
- The event's own disclosure fact sentences (`disclosure.items[].content`, `kind="facts"`) — including any guidance/quantitative language embedded within them
- `event_type`, `ticker`/`focal_assets`, `event_datetime` (as identity/scheduling context, not as a proxy for outcome)
- `information_url`, `prediction_deadline` (operational, not signal)

**SAFE ONLY AFTER TIMESTAMP VALIDATION:**
- Prior-event `car1` and `earnings_surprise` for the same ticker — safe only if the prior event's `event_datetime` is strictly before the current event **and** that prior event's outcome is confirmed to already be in our own archive coverage (not merely chronologically earlier — see the sealing-lag constraint in §3.3/§5)
- All eight rolling/aggregate historical-company features in §5, with the same underlying constraint
- `knowledge_cutoff` via a `GET /events` lookup (safe once implemented, since it's a required field of that response) — currently not implemented in `predict.py`
- Any externally sourced market price, index/benchmark return, volatility, sector classification, analyst estimate, consensus, guidance history, or filing data — safe only with an explicit, source-specific, provable `timestamp <= knowledge_cutoff` (or `< event_datetime` for market data specifically) check per record, never assumed

**NOT CURRENTLY AVAILABLE (not a "never," just not present today):**
- Prior-event outcomes for events still in an unsealed quarter (practically, this includes essentially the entire *current* quarter for most of its duration, given the observed ~6-week sealing lag)
- Any market price, index return, volatility, sector, analyst-estimate, consensus, guidance-history, or filing dataset — none exist in either repository today
- `number_of_prior_earnings_events` and other existence-only counts pending confirmation that `GET /events` actually retains old realized entries

**NEVER USE:**
- `event_returns.car1` and its components (`r_i`, `r_m`) for the current event
- `metrics.earnings_surprise` for the current event
- `baseline_predictions` for the current event (post-event AND circular)
- `returns_computed_at`, `status`, `return_status`, `surprise_status` as feature values (they only exist once the event is already resolved)
- Any quarter-level cross-sectional percentile rank (`y`, `surprise_pct`) as a feature — these require the whole quarter's outcomes, the single most severe leakage vector identified in this audit

---

## 8. Open Questions / Blockers

1. **Does `GET /events` retain old realized entries indefinitely, or only a rolling window?** This determines whether "prior event existence/count" is genuinely `SAFE_LIVE` or needs the same archive-dependent caveat as outcome-bearing features. Not resolved by this audit — requires a live call with a wide historical `start_date`/`end_date` window and inspection of what comes back.
2. **What is the live-path latency for `disclosure.generated_at`?** The archive's 10–274 day lag is confirmed to be a backfill artifact, but no live delivery was inspected in this audit to establish the real number. Low risk either way, since content is rules-exempted regardless of generation time, but worth confirming for completeness.
3. **Does the "event materials... are fair game" exemption extend to externally, independently sourced material about the same event** (e.g., a licensed full transcript, or a competitor's own faster-than-platform disclosure), or is it specific to platform-delivered materials? Unresolved; recommend a direct question to `contact@explainingmarkets.ai` before relying on any externally sourced same-event material.
4. **Exact sealing cadence** — we observed one data point (`2026Q2`: quarter ended 2026-06-30, sealed/built 2026-08-11, ~6 weeks). Whether this is representative (vs. a one-off) is not established from a single observation.
5. **Multi-asset events** — all 6,287 records currently on file are single-focal-asset events. The prior-event rolling features in §5 are defined per-ticker and should generalize cleanly if multi-asset events appear, but this has not been tested against real multi-asset data because none exists yet.

---

## 9. Recommended Next Research Step

Before writing any feature-extraction code beyond what already exists:

1. Resolve Open Question #1 (`GET /events` retention window) — it is the cheapest possible check (one more live API call, no new infrastructure) and determines whether an entire feature sub-family (`number_of_prior_earnings_events`, and possibly a fast existence-only "has this ticker reported recently" signal) is `SAFE_LIVE` today or gated behind the archive's sealing lag like everything else in that family.
2. If confirmed available, treat that as the **first** new live feature to prototype (lowest risk, per this audit), before attempting any `SAFE_IF_TIMESTAMPED` feature that requires building a point-in-time-availability check against our own archive download state.
3. Do **not** attempt to source any external dataset (market prices, analyst estimates, filings) until the fully-internal `SAFE_IF_TIMESTAMPED` features (§5) have been implemented, backtested, and shown to add signal beyond the existing surprise-only benchmark — external sourcing carries materially higher implementation cost and timestamp-verification risk (per §6) and should not be the first move.

No model or `predict.py` changes are recommended or were made as part of this audit.

---

## Files inspected vs. modified

**Files inspected (read-only) in this audit:**
`predict.py`; `modal_app.py`; `src/explaining_markets/{__init__,config,client,event_utils,webhook_verification,historical,features,model,backtest}.py`; `examples/src/examples/{__init__,client,config,schemas,archive,scoring,summary,frames,plotting}.py`; `examples/notebooks/*.ipynb`; `examples/data/README.md`; `examples/.env.example`; `data/historical/PROVENANCE.json`; `HISTORICAL_DATA_INVESTIGATION.md`/`competition.md`; the live OpenAPI spec at `https://explainingmarkets.ai/openapi.json`; and all 6,287 records across `data/historical/EARNINGS_RELEASE_{2025Q4,2026Q1,2026Q2}.jsonl.gz` (full field-inventory scan, not sampling).

**Files created:** `docs/PREDICTION_TIME_INFORMATION_AUDIT.md` (this document) only.

**Files modified:** none. `predict.py`, `modal_app.py`, `model.py`, and the production submission pipeline are unchanged from their state at the end of the prior session.
