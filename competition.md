# Competition Context — Explaining Markets

**Purpose of this document:** a single, comprehensive reference so another AI
agent (or human) can understand the competition, this repository's execution
architecture, the sibling `examples/` repository, the available data, the
scoring mechanism, and current implementation state — without re-reading the
whole codebase from scratch.

This document is descriptive only. No strategy code was changed to produce it.
`predict.py`, `modal_app.py`, and the `src/explaining_markets/` package are
untouched.

---

## 0. Repository layout (as inspected)

```
markets/                              ← this repo (the "starter-modal" deployment)
├── README.md
├── predict.py                        ← the only file competitors edit
├── modal_app.py                      ← Modal/FastAPI deployment ("plumbing")
├── pyproject.toml / uv.lock
├── .env.example / .env (gitignored, credentials configured — see §11)
├── docs/
│   ├── advanced.md
│   └── images/submission_flow.png    ← "How a submission works" diagram
├── src/explaining_markets/
│   ├── __init__.py
│   ├── config.py
│   ├── client.py
│   ├── event_utils.py
│   └── webhook_verification.py       ← vendored HMAC verifier
├── tests/
│   ├── test_predict.py
│   ├── test_modal_app.py
│   ├── test_webhook_verification.py
│   └── test_vectors.json
└── examples/                          ← cloned sibling repo, NOT part of the
                                          deployed starter; read-only research
                                          / documentation aid
    ├── README.md
    ├── data/
    │   ├── README.md
    │   ├── sample/                    (events_sample.json, archive_*.jsonl.gz,
    │   │                                transcript_sample.json, summary_sample.json)
    │   └── archive/ (gitignored cache for real downloaded data)
    ├── notebooks/
    │   ├── 00_api_quickstart.ipynb
    │   ├── 01_historical_archive.ipynb
    │   └── 02_earnings_call_facts.ipynb
    ├── scripts/download_archive.py
    ├── src/examples/
    │   ├── __init__.py, client.py, config.py, schemas.py, frames.py,
    │   │   plotting.py, archive.py, scoring.py, summary.py
    └── tests/ (offline, mocked HTTP + bundled sample data)
```

`markets/examples` is a full clone of the separate, official
`explaining-markets/examples` repository (per its own README/CI badges), living
inside this project's directory as a sibling for reference. It is **not**
imported by `predict.py` or `modal_app.py` and is **not** part of the deployed
image (`modal_app.py`'s `add_local_python_source` only bundles
`explaining_markets` and `predict`). It exists purely to give research access to
the competition's API client, historical archive tooling, and exact scoring
implementation.

---

## 1. What the competition predicts (confirmed facts)

- **Confirmed** (Source: `README.md`, `predict.py` docstring, `examples/src/examples/scoring.py` module docstring): For each competition **event** (currently always `EARNINGS_RELEASE`), for each **focal asset** (a ticker), the competitor must submit a single float `predicted_percentile ∈ [0, 1]`.
- **Confirmed**: `predicted_percentile` is a prediction of where the asset's **next-day abnormal (market-adjusted) return** will rank, in `[0, 1]`, **cross-sectionally across all of the quarter's event outcomes** — not a percentile within the asset's own history. `0` = the quarter's most negative reaction, `0.5` = median, `1` = most positive.
  Source: `predict.py` docstring (lines 41-54), `README.md` lines 137-144.
- **Confirmed**: The realized target the platform scores against is `car1` — a one-day cumulative abnormal return — percentile-ranked within the scoring period (a quarter). Source: `examples/src/examples/scoring.py` lines 1-16, 145-166 (`add_percentiles`).
- **Confirmed**: A **focal asset** is `{"identifier_type": "TICKER", "identifier_value": "<TICKER>"}`. Source: `event_utils.py` docstring, `examples/src/examples/schemas.py::AssetIdentifier`.
- **Confirmed**: An **event** currently is always `event_type == "EARNINGS_RELEASE"` (with a synthetic `"TEST"` type for portal self-tests). `event_type` is documented as an **open string set** — new types can appear. Source: `examples/src/examples/schemas.py` docstring, `examples/README.md`, `examples/notebooks/00_api_quickstart.ipynb` cell 2.
- **Confirmed**: `timing_category` is `"SCHEDULED"` or `"UNSCHEDULED"`. Source: `CalendarEvent.timing_category` docstring in `examples/src/examples/schemas.py`.

---

## 2. Full event payload shape (confirmed, reconstructed from both repos)

The **webhook-delivered** event (what `predict(event)` receives), per
`src/explaining_markets/event_utils.py` docstring and `predict.py`:

```jsonc
{
  "id": "<matches Webhook-Id header; idempotency key>",
  "event_id": "<uuid>",
  "event_type": "EARNINGS_RELEASE",              // or "TEST"
  "timing_category": "SCHEDULED",
  "event_datetime": "2026-01-15T21:00:00Z",
  "focal_assets": [
    {"identifier_type": "TICKER", "identifier_value": "AAPL"}
  ],
  "information_url": "https://...signed...",      // short-lived signed URL → event summary JSON
  "prediction_deadline": "2026-01-15T21:05:00Z"
}
```

The **calendar** (`GET /events`) event, per `examples/src/examples/schemas.py::CalendarEvent`,
additionally documents a `knowledge_cutoff` field:

```jsonc
{
  "event_id": "...",
  "event_type": "EARNINGS_RELEASE",
  "timing_category": "SCHEDULED",
  "event_datetime": "2026-01-27T21:00:00Z",
  "knowledge_cutoff": "2026-01-26T21:00:00Z",
  "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "MSFT"}]
}
```
(Confirmed shape, source: `examples/data/sample/events_sample.json`, matches `CalendarEvent`.)

The **realized/archive** event record (what `GET /archive` delivers, and what
`predict.py`'s `information_url` fetch conceptually resolves to a summary of)
carries a `disclosure` block:

```jsonc
{
  "event_id": "a1000000-...-000001",
  "event_type": "EARNINGS_RELEASE",
  "timing_category": "SCHEDULED",
  "event_datetime": "2025-07-31T21:00:00Z",
  "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "AAPL"}],
  "disclosure": {
    "schema_version": "1.0",
    "generated_at": "2025-07-31T21:00:00Z",
    "items": [
      {
        "id": "facts-1",
        "kind": "facts",
        "source": "earnings_call",
        "content": [ "Revenue $94.0B, up 5% YoY.", "..." ]
      }
    ]
  }
}
```
(Confirmed shape, source: `examples/data/sample/archive_EARNINGS_RELEASE_2025Q3.jsonl.gz`,
`examples/src/examples/schemas.py::Disclosure`/`DisclosureItem`.)

The **real, scored** archive record additionally carries (per the scoring
module's expectations, though not present in the tiny bundled sample):
`event_returns` (per-ticker `car1`, `return_status`), `metrics.earnings_surprise`
(`surprise`, `surprise_status`), and `baseline_predictions` (reference LLM
baselines' own predicted percentiles, keyed
`"gemini/ea-explain-contemp-summary"` / `"openai/ea-explain-contemp-summary"`).
**Confirmed to exist** as fields the production scorer reads (source:
`examples/src/examples/scoring.py::event_asset_rows`, and its test fixtures in
`examples/tests/test_scoring.py::_record`), but **UNCONFIRMED** in exact
real-world schema/completeness because the bundled sample archive file does not
include them (see §7).

---

## 3. Knowledge cutoffs

- **Confirmed**: `GET /events` calendar entries carry `knowledge_cutoff`, an
  ISO-8601 UTC datetime. Per `README.md`: *"your agent must not use any
  information from after that instant... The event payload delivered to your
  webhook describes the event itself and is fair game."* Source: `README.md`
  lines 157-164.
- **Confirmed**: The webhook-delivered `event` payload (§2, top) does **not**
  itself carry a `knowledge_cutoff` field in the documented shape (`event_utils.py`
  docstring) — only the calendar (`GET /events`) response schema explicitly
  models it (`CalendarEvent.knowledge_cutoff`). **Potential concern (not a
  confirmed bug):** if the webhook delivery omits `knowledge_cutoff`, an agent
  wanting to programmatically self-enforce the cutoff would need to cross-reference
  the calendar API by `event_id`, since `predict.py` cannot read a field that
  isn't there. **UNCONFIRMED** whether the live webhook payload actually includes
  `knowledge_cutoff` (the vendored docstring may simply be incomplete) — this
  should be verified against a real delivery or the API's OpenAPI docs
  (`/docs/api`, referenced in `examples/data/README.md`) before relying on its
  absence.
- **Confirmed**: The event's own `information_url` disclosure (earnings-call
  facts) is explicitly "fair game" regardless of cutoff, since it *describes the
  event itself* — this is the mechanism by which the competition supplies
  information about the event without violating the cutoff (the facts are
  extracted only from the transcript of that exact call, nothing else — see §8).

---

## 4. Current runtime architecture (this repo, `starter-modal`)

### 4.1 High-level flow

```
Competition platform (upstream, not controlled by this repo)
        │  detects/schedules an event
        ▼
Signed webhook POST → Modal public URL (root path)
        │
        ▼
modal_app.py :: web() → competition_webhook(request)
   1. raw_body = await request.body()             (RAW BYTES — never request.json())
   2. verify_webhook(raw_body, headers, secret)    [webhook_verification.py]
        │  failure → 401
        ▼
   3. _claim_aio(webhook_id)  [modal.Dict "seen_webhooks"]
        │  already in_flight/done → return 200, no further work
        ▼
   4. log_deadline(event)     [event_utils.py — best-effort print]
        ▼
   5. return 200  ← the ACK (must happen within 20s of platform's POST)
        ▼
   6. predict_and_submit.spawn.aio(event, webhook_id)   ← fire-and-forget,
                                                            separate Modal container
        ▼
modal_app.py :: predict_and_submit(event, webhook_id)   (own container, timeout=600s)
   7. is_test(event) → neutral_predictions(event)  [TEST events]
      else           → predict(event)               [predict.py — the strategy]
        ▼
   8. submit_predictions(event_id, predictions, Config.from_env())  [client.py]
        │  POST {EM_API_BASE_URL}/predictions, header X-API-Key
        ▼
   9. on success → _release(webhook_id, submitted=True)  → marks "done" (permanent)
      on exception → _release(webhook_id, submitted=False) → drops claim (retry later)
        ▼
Competition platform scores the submission later (offline, per-quarter)
```

### 4.2 Two clocks (confirmed, `docs/advanced.md`, `README.md`)

| Clock | Budget | Starts | Miss it and… |
|---|---|---|---|
| Delivery ACK | **20 s** | platform POSTs to your webhook | delivery retried up to 5×/~30min; 5 consecutive failures email admins; ~50 disables the webhook |
| Prediction window | **5 min** | you ACK 200 | prediction tagged late, dropped at scoring |

Note: `docs/images/submission_flow.png` labels the ACK budget "10s" — this
contradicts the README/advanced-docs' "20s" figure everywhere else. Treat 20s as
the authoritative value (it appears in three separate textual sources vs one
diagram label) and flag the diagram as outdated/inconsistent.

### 4.3 Idempotency (confirmed, `docs/advanced.md`, `modal_app.py`)

- Dedup key: `Webhook-Id` header == `event["id"]`, stable across retries.
- States in `modal.Dict("em-webhook-dedupe")`: **absent** (never seen / last attempt raised → run), **in_flight** (claimed, being worked → skip duplicate), **done** (API accepted the prediction → permanent skip).
- Claim uses `put(..., skip_if_exists=True)` — atomic, race-safe across containers.
- Marking "done" only happens **after** a successful submit — a failed prediction is never falsely marked handled.

### 4.4 Submission API behavior (confirmed, `README.md`, `client.py`)

- `POST {EM_API_BASE_URL}/predictions` with `X-API-Key` header, body
  `{"event_id": ..., "predictions": [{"identifier_value", "predicted_percentile"}, ...]}`.
- Always accepts well-formed predictions with `201`; late, duplicate, and
  pre-broadcast submissions are **tagged for scoring purposes, not rejected**.
- **Only the first submission per event is scored** — re-POSTing the same
  event_id is accepted but does not overwrite it. Get it right the first time.
- TEST events: accepted, but **never scored**.

---

## 5. `predict.py` deep analysis

### 5.1 What it receives / must return

- Input: the verified webhook `event` dict (§2).
- Output contract: `list[dict]`, one entry per focal asset:
  `{"identifier_value": str, "predicted_percentile": float in [0,1]}`.

### 5.2 Current baseline implementation, step by step

1. `httpx.get(event["information_url"], timeout=15.0)` → fetches the event's
   summary JSON (`summary_json`). `raise_for_status()` — any non-2xx propagates.
2. For **each** focal asset (comment notes: today every event carries exactly
   one asset; if that changes, run concurrently rather than raising the
   timeout), calls `_ask_llm(summary=summary_json, ticker=..., event_type=...)`
   **serially**.
3. `_ask_llm`:
   - If `OPENAI_API_KEY` is unset: logs a one-shot warning, returns `0.5`
     (the "round-trip works without burning credits" baseline).
   - Else lazily builds a module-global `OpenAI(timeout=120.0, max_retries=1)`
     client (picks `OPENAI_API_KEY` up from env implicitly).
   - Extracts `summary.get("summary")` (a text string) if present, else
     `json.dumps(summary)`; truncates to 8000 chars.
   - Calls `client.chat.completions.parse(model=openai_model(), messages=[system, user], response_format=Prediction)`
     — OpenAI's **structured outputs** decoding mode.
   - `Prediction` is a Pydantic model: `predicted_percentile: float = Field(ge=0.0, le=1.0)` — the `[0,1]` bound is enforced by the JSON schema the SDK builds from this model, not by manual clamping.
   - `SYSTEM_PROMPT` (verbatim, `predict.py` lines 94-112) encodes calibration
     discipline: ~25%/50%/25% up/neutral/down base rates, discourage
     overconfidence (reserve >0.80/<0.20 for unambiguous evidence, hard bounds
     0.90/0.10, tone should move the number ≤~0.03).
   - `user_prompt` interpolates `event_type`, `ticker`, `summary_text`, and a
     5-point weighing checklist (quantitative surprise → guidance → strategic
     shifts → tone → risks).
   - If the model "refuses" (`parsed is None`), falls back to `0.5`.
4. Timeout budgeting (explicit comment): worst case `15 + (120×2) + 15 = 270s`
   against the 300s (5-minute) prediction window — ~30s slack. **This budget
   assumes exactly one focal asset**; multiple assets processed serially would
   blow the budget (acknowledged in-code, not currently guarded against).
5. Failure handling: no try/except inside `predict.py` — any exception (HTTP
   error, OpenAI error after its 1 retry, or a `ValidationError` from the
   structured-output schema not matching) propagates to
   `predict_and_submit`, which logs it and releases the idempotency claim so a
   redelivery can retry (there is otherwise **no retry on `predict()` itself**).

### 5.3 Strategy vs. plumbing split

- **Strategy** (intended to be replaced/improved): the entire body of `predict()`
  and `_ask_llm`, `SYSTEM_PROMPT`, the `Prediction` schema, the "what data to
  fetch and how to reason about it" logic.
- **Plumbing** (not meant to change): webhook receipt, verification, ACK/spawn
  split, idempotency, submission POST — all outside `predict.py`.

---

## 6. API surface — consolidated across both repos

Base URLs (confirmed, `config.py` in both repos):
- Starter (`src/explaining_markets/config.py`): production default
  `https://api.explainingmarkets.ai/v1`.
- Examples (`examples/src/examples/config.py`): beta default
  `https://api-beta.explainingmarkets.ai/v1` (both overridable via
  `EM_API_BASE_URL`). The examples README says beta is "the stage open to
  invite-code holders" — implying production and beta may be gated
  differently. **UNCONFIRMED** exactly how these two stages diverge in data
  completeness/availability.

Authentication: `X-API-Key` header, from `EM_API_KEY`. Confirmed in both
`client.py` implementations.

| Endpoint | Method | Purpose | Confirmed in |
|---|---|---|---|
| `/events` | GET | Calendar — scheduled/realized events, optional `start_date`/`end_date` window | `examples/src/examples/client.py::events()` |
| `/health` | GET | Rolling 24h webhook + submission counters for your submission | `examples/src/examples/client.py::health()` |
| `/archive` | GET | Historical archive manifest (list of downloadable files w/ signed URLs) | `examples/src/examples/client.py::archive_manifest()` |
| `/archive/{event_type}/{quarter}` | GET | Refresh one file's expired signed URL | `examples/src/examples/client.py::archive_file()` |
| `/webhook/test` | POST | Fire a synthetic TEST delivery at your registered webhook (202 = enqueued) | `examples/src/examples/client.py::send_test_event()` |
| `/predictions` | POST | Submit predictions for one event (`X-API-Key`; 201 on acceptance) | `src/explaining_markets/client.py::submit_predictions()` |
| `information_url` (per-event, signed) | GET | Fetch one event's summary/disclosure JSON | `predict.py` |
| `/docs/api` | — | OpenAPI reference (mentioned, not fetched here) | `examples/data/README.md` |

Response schemas (Pydantic, lenient `extra="allow"` — future fields pass
through): `CalendarEvent`, `AssetIdentifier`, `DisclosureItem`, `Disclosure`,
`ArchiveFile`, `ArchiveManifest`, `SubmissionHealth` — all in
`examples/src/examples/schemas.py` (see §2 for shapes). No pagination
mechanism is present in any modeled endpoint (`/events` and `/archive` both
return full lists with no cursor/page params). No documented rate limits
found in either repo — **UNCONFIRMED**.

Error handling in both clients: any non-2xx raises (`ApiError` in examples,
`PredictionSubmissionError` in the starter) with the response body/status
embedded; the examples client special-cases hint text for 401/403/404.
`submit_predictions` has **no retry logic** — a single POST attempt.

---

## 7. Historical data (the archive)

### 7.1 What exists and where

- **Confirmed**: `GET /archive` returns an `ArchiveManifest` — a list of
  `ArchiveFile` entries, one per `(event_type, quarter)`, each a gzip-JSONL file
  behind a short-lived CloudFront-signed URL. Fields: `event_type`, `quarter`,
  `key`, `url`, `events` (count), `bytes`, `sealed` (bool), `url_expires_at`,
  `event_datetime_min`/`max`.
- **Confirmed**: Downloaded/cached locally via
  `examples.archive.download_archive()` into `data/archive/` (gitignored) or a
  custom `--dest`; `scripts/download_archive.py` is the headless CLI
  equivalent, filterable by `--event-type` and `--since <quarter>`.
- **Confirmed**: Loaded into a pandas DataFrame via
  `examples.archive.load_archive()`, which reads every `*.jsonl.gz` in a
  directory and adds `event_type`/`quarter` provenance columns from the
  filename (never overwriting a field the record already has).
- **Confirmed** ("sealed" flag exists on `ArchiveFile`) implies the platform
  marks some historical quarters as finalized/immutable — **UNCONFIRMED**
  precisely what "sealed" guarantees operationally (e.g., whether unsealed
  quarters can still have their content revised).

### 7.2 Schema — bundled sample vs. real archive

The bundled sample (`data/sample/archive_EARNINGS_RELEASE_2025Q3.jsonl.gz`, 5
records) contains only: `event_id`, `event_type`, `timing_category`,
`event_datetime`, `focal_assets`, `disclosure` (with `facts`/`earnings_call`
items). It explicitly does **not** include `event_returns`, `metrics`, or
`baseline_predictions` — confirmed both by direct inspection and by
`examples/data/README.md`: *"The record shape mirrors the documented event
payload with an inlined disclosure (facts). It is a stand-in for exposition —
the real archive is served by the API and its lines may include additional
fields. Treat the API as the source of truth."*

The **real** archive record schema, per what `examples/src/examples/scoring.py`
and its tests (`examples/tests/test_scoring.py::_record`) expect to read,
includes additionally:

```jsonc
{
  "event_returns": {
    "AAPL": {"car1": 0.05, "return_status": "ok"}
  },
  "metrics": {
    "earnings_surprise": {"surprise": 0.01, "surprise_status": "ok"}
  },
  "baseline_predictions": {
    "gemini/ea-explain-contemp-summary": {"AAPL": 0.7},
    "openai/ea-explain-contemp-summary": {"AAPL": 0.6}
  }
}
```

- `car1` = the **realized one-day cumulative abnormal (market-adjusted) return**
  for that ticker for that event — this is literally the ground truth the
  competition scores predictions against (percentile-ranked into `y`).
- `earnings_surprise.surprise` = a numeric earnings-surprise metric with a
  status flag (`"ok"` vs presumably `"unavailable"`), percentile-ranked
  separately as the naive benchmark regressor.
- `baseline_predictions` = the competition's own **two reference LLM
  baselines'** predicted percentiles per ticker (Gemini 2.5 Flash-Lite and
  GPT-5 nano, per `BASELINE_LABELS`), included in the archive presumably so
  competitors can benchmark against them offline.

**This schema is UNCONFIRMED for the tiny bundled sample** (it lacks these
fields entirely) but **confirmed to be expected/consumed** by the production
scoring port and its test fixtures. `examples/notebooks/01_historical_archive.ipynb`
section 5 explicitly gates on `{"event_returns", "metrics",
"baseline_predictions"} <= set(df.columns)` before attempting the real
regression reproduction — i.e., the notebook itself acknowledges this data
is only present when the **live** archive is downloaded, not in the sample.

### 7.3 Look-ahead bias considerations

- The archive is **historical/realized** data — by construction every record in
  it is for an event whose knowledge cutoff has already passed and whose
  `car1` outcome is already known. Using it for **training a model on past
  events** is safe as training data as long as, at inference time for a *new*
  event, you only ever use the model's fitted parameters (not future
  post-cutoff information about that new event).
- **Potential look-ahead risk to be aware of when building a training set**:
  the archive's `disclosure.items[].content` "facts" for a given event are
  themselves generated *after* the event occurred (they summarize the actual
  earnings call, which happens at `event_datetime`). This is fine to use as a
  training feature/target-adjacent covariate for **that same historical event**
  (there's no leakage within one event: facts and returns both post-date the
  cutoff and are used together retrospectively). But it would be a **leakage
  bug** to use one event's post-event facts/returns as an input feature when
  predicting a *different, contemporaneous* event — not applicable here since
  each event's facts only describe itself.
- The **cross-sectional percentile ranking** (`percentile_ranks`) is computed
  **within a period** (quarter) over the *whole* set of that period's outcomes.
  If you use archived data to build a rank-based feature (e.g., "what
  percentile would this surprise value be within its historical quarter"),
  you must restrict the ranking universe to that specific quarter's known
  outcomes — exactly as `outcomes_frame`/`add_percentiles` already do — or you
  introduce look-ahead by ranking against data that wasn't available yet in a
  live setting. Note also this ranking is *retrospective by construction* for
  archive data (the whole point of the archive is analysis after the fact);
  it cannot be used to compute a **live** prediction's percentile in real time,
  because the current quarter's cross-section isn't complete until scoring
  time — this is the fundamental reason competitors submit `predicted_percentile`
  and the platform computes the realized percentile *itself*, later.
- `baseline_predictions` in the archive are themselves other **submissions'**
  predicted percentiles for historical events — using them as a feature for
  your own live model would be an unusual (not necessarily prohibited, but
  worth flagging) form of relying on a competitor baseline's forecast rather
  than the underlying data; there's no evidence this is disallowed by the
  rules, but it's not "primary" signal.

### 7.4 Constructing a training set (conceptual only — not implemented)

Conceptually, from `load_archive(...)` + `examples.scoring`:
1. For each historical quarter, build the per-`(event, asset)` outcome frame
   via `outcomes_frame(records)` → columns `event_id`, `identifier_value`,
   `car1`, `surprise`, and each baseline's predicted percentile.
2. Apply `add_percentiles(frame)` to get `y` (realized abnormal-return
   percentile rank within the quarter) and `surprise_pct`.
3. For each event, pull its `disclosure` "facts" (`facts_from_disclosure`) as
   textual features, plus any other archive fields, plus `surprise_pct` as a
   numeric feature.
4. Train a model to predict `y` (or something correlated with it) from
   pre-cutoff-available features (facts text + surprise + any other
   quantitative fields the archive exposes), being careful that ranking-based
   features are computed within-quarter only over already-realized outcomes.
5. At **live inference time** (inside `predict.py`), you cannot compute `y`
   directly (the current quarter's cross-section is incomplete) — you instead
   predict a percentile *estimate*, which is exactly `predicted_percentile`,
   using the model fitted on historical quarters' `(features → y)` pairs.

No such model is implemented in either repo; this is purely the transform
`examples.scoring` already provides for **offline analysis and backtesting**.

---

## 8. Earnings-call / textual information — deep dive

### 8.1 What competitors actually receive

Per `predict.py`, the live webhook event carries `information_url`, a
short-lived signed URL to a "summary JSON." Per `examples/src/examples/summary.py`
and `examples/src/examples/schemas.py::DisclosureItem`, the **archive's**
equivalent field is `disclosure.items[]` with an entry
`kind="facts"`/`source="earnings_call"` whose `content` is a list of **exactly
ten sentences** ("facts") distilled from the actual earnings-call transcript.

**Important nuance (UNCONFIRMED exact equivalence):** `predict.py` fetches
`information_url` and treats the response as `{"summary": "<text>"}` or falls
back to `json.dumps(the whole object)`. The examples repo's `disclosure` object
instead has a `content: list[str]` of ten facts, not a single `summary` string.
It is **not fully confirmed from source alone** that these are byte-identical
representations of the same underlying artifact — they are very likely the
same underlying "ten facts" content, but `predict.py`'s code defensively
handles both a `summary` key and an arbitrary JSON blob, suggesting the author
was not 100% certain of the exact `information_url` response shape either.
**Action item:** confirm the real `information_url` response shape against a
live test event before optimizing prompt engineering around it.

### 8.2 How the facts are produced (documented in `examples/src/examples/summary.py`)

- Model: `claude-opus-4-8` (constant `SUMMARY_MODEL`, tracks production, may be
  stale for older disclosures — the artifact's own metadata records which model
  actually produced it).
- Request: a **plain** Anthropic Messages API call — `system` + one `user`
  message, **adaptive extended thinking at `effort="high"`**, `max_tokens=16000`.
  **No tools, no `response_format`/JSON schema, no `temperature`.** The
  ten-fact JSON structure is requested purely via prompt text (`SUMMARY_USER_TEMPLATE`)
  and validated after the fact.
- Input to the model: `format_transcript(transcript)` — renders **only** the
  transcript's own header title plus every speaker/text component as markdown.
  **No ticker, no event date, no fiscal period, no consensus estimates, no
  market data are injected.** No truncation/chunking — the entire call goes
  into one message.
- Output: exactly 10 sentences, each "a single sentence capturing a specific,
  quantified insight," synthesized (not verbatim), drawn only from the
  transcript.
- Parsing: `recover_facts()` (lenient — what production runs) applies only
  safe repairs (strip fences, close a missing `]`, trim extra facts to 10);
  anything else is an unrecoverable failure, recorded as `facts: null` with a
  `parse_note` (`"strict"`, `"repaired_close"`, `"trimmed_from_N"`,
  `"unparseable"`, `"no_facts_list"`, `"empty_fact"`, `"too_few:N"`).
- Because thinking is adaptive/nondeterministic, **the same transcript can
  produce different facts on different runs** — a source of irreducible noise
  in the disclosure content itself.

### 8.3 What this means for prediction strategy

- The "facts" available to `predict.py` are a **quantified, investor-framed
  distillation of the call and nothing else** — no market data, no analyst
  estimates, no cross-company context. Anything beyond the call's own content
  must be sourced by the competitor separately (subject to the knowledge
  cutoff).
- Facts are ordered by the model's own sense of relevance (not guaranteed
  ranked by magnitude of market impact).
- Because facts are synthesized, they may already implicitly encode some
  "surprise" judgment (e.g., "ahead of guidance," "below consensus") — worth
  exploiting as a feature (e.g., counting positive vs negative framing terms,
  or few-shot-prompting an LLM to rate each fact's likely market impact).
- `facts_from_disclosure(record)` is the utility to extract them from an
  archive record; returns `[]` if absent (e.g., not-yet-occurred scheduled
  event, or a call whose summarization failed).

---

## 9. Scoring — exact mechanics (source: `examples/src/examples/scoring.py`, extensively docstring-documented; corroborated by `examples/tests/test_scoring.py`)

### 9.1 The unit of scoring

- Scoring happens **within a period** — a calendar quarter, or the official
  contest "Scoring Period."
- For each realized `(event, focal asset)` outcome with a `car1`:
  - `y` = percentile rank of `car1` **across the period's cross-section**
    (`percentile_ranks`, ties share the average rank; single value → 0.5;
    empty → `[]`).
  - `surprise_pct` = percentile rank of the earnings surprise, ranked only
    over rows with a valid (`surprise_status == "ok"`) surprise.

### 9.2 The regression (per submission, per period)

```
y = alpha + beta1 * predicted_percentile + beta2 * surprise_percentile
```

- `r_squared` = fit's R² (how much of the realized ranking the model
  explains).
- `r_squared_surprise` = the same-sample univariate fit on `surprise_pct`
  alone (naive benchmark).
- `delta_r_squared` = `r_squared - r_squared_surprise` — **the key metric**:
  how much a submission's predictions explain **beyond** the naive
  earnings-surprise benchmark.

### 9.3 Two properties that materially affect strategy (explicit in the module docstring)

1. **Score is invariant to any positive affine remap of your predictions.**
   `beta1` enters squared in the `delta_r_squared` identity
   (`delta_r_squared == beta1**2 * s11_2 / s_yy`), so `p → 1-p`, or rescaling
   toward the middle, leaves R², delta, and ranking **bit-for-bit identical**.
   **This measures explanatory power / ranking correlation, not calibration.**
   Neither the direction nor the spread of your raw numbers is scored directly
   — only how much of the realized ranking they track (confirmed by
   `test_reversed_predictions_score_identically`,
   `test_positive_rescaling_leaves_the_score_unchanged`).
2. **Constant predictions earn exactly zero.** A submission that always
   predicts the same value has no regressor variance; `beta1` is
   unidentified/forced to 0; `delta_r_squared` is exactly `0.0`. Still ranked
   (at the bottom), not disqualified (confirmed by
   `test_constant_predictions_earn_nothing_but_stay_ranked`, including a
   floating-point-robust `DEGENERATE_SD = 1e-6` threshold so this isn't fooled
   by rounding).

### 9.4 Contest-specific imputation rules (Official Rules §5, per docstring)

- Every submission is evaluated on the **same common set of valid scored
  events** ("Scored Events" — every row with both `y` and `surprise_pct`
  defined).
- First compute the arithmetic mean of the submission's own **timely, valid**
  predictions on that common set (before any filling).
- Every Scored Event the submission did **not** validly predict is
  mean-imputed with that submission-specific mean; refit the same
  two-regressor OLS over the **full** common set → metrics suffixed
  `_imputed` (`r_squared_imputed`, `delta_r_squared_imputed` — **the contest
  ranking metric**).
- A submission with **no** timely valid prediction on any Scored Event has an
  undefined mean → **not eligible** for imputed metrics or prizes.
- Tie-break: exact metric tie → higher pre-imputation coverage (`n_obs`) wins;
  any remaining exact tie splits the combined prize equally, **no random
  tie-break**.

### 9.5 Implication for strategy

- **Calibration does not matter for the score** (only rank/explanatory
  correlation does) — but note the README still frames `predicted_percentile`
  as a genuine percentile prediction, and other downstream consumers (or
  future scoring changes) might care about calibration; don't over-index on
  exploiting the affine-invariance property as "the" strategy without
  confirming this remains true for the official contest rules version you're
  bound by.
- **Never submit a constant value for all events** — it guarantees a
  zero contribution (worse than any noisy but variable signal, on the ranking
  metric).
- **Coverage matters as a tie-break and, more importantly, for imputation** —
  missing predictions get filled with your own mean, which dilutes your signal
  toward the benchmark; predicting on **every** Scored Event you can (even
  with a mediocre model) is likely better than skipping some.
- The competition explicitly benchmarks you against the earnings-surprise
  ranking and two named reference LLM baselines' percentiles
  (`BASELINE_LABELS`) — these baselines' outputs are visible in archive data
  and give a natural performance floor/reference point.

---

## 10. Data availability vs. prediction-time availability

| Data | Exists? | Source | Available at prediction time? | Potential use |
|---|---|---|---|---|
| Event metadata (`event_id`, `event_type`, `focal_assets`, `event_datetime`, `prediction_deadline`) | Yes | Webhook payload (`event_utils.py`) | Yes — delivered directly | Identify the event/asset; scheduling |
| `knowledge_cutoff` | Confirmed on `GET /events` calendar; UNCONFIRMED on webhook payload | `examples/src/examples/schemas.py::CalendarEvent` | Possibly not in the webhook payload itself — may require a calendar lookup by `event_id` | Enforce the cutoff rule programmatically |
| Event summary/"facts" (`information_url` / `disclosure.items[]`) | Yes | Webhook `information_url`; archive `disclosure` | Yes — this **is** what `predict.py` fetches | Primary textual signal for the model |
| Historical events (archive) | Yes | `GET /archive` → gzip-JSONL, cached via `download_archive` | Yes, but only for **past** events (by definition — used for training/backtesting, not the live event itself) | Train/calibrate a model offline |
| Earnings-call full transcripts | **No** — not distributed; only the 10-fact distillation is exposed | N/A (licensed; `examples/data/README.md` explicitly says real transcripts aren't shipped) | No | N/A — only the facts, not the raw transcript, are ever available |
| Realized market/return data (`car1`) | Yes, but only for **historical/scored** archive events | Archive `event_returns` field (UNCONFIRMED present in bundled sample; confirmed expected by scoring code) | No for the live event (that's the outcome being predicted) — Yes for historical events | Training target; backtesting |
| Earnings surprise (`metrics.earnings_surprise`) | Yes, historically | Archive `metrics.earnings_surprise` | No for live event's *realized* surprise status ahead of the call — analyst **expectations** (consensus) might be independently obtainable from other sources subject to the cutoff | Naive benchmark regressor; possible feature if you can source consensus estimates yourself pre-cutoff |
| Reference baseline predictions (`baseline_predictions`) | Yes, historically | Archive `baseline_predictions` | No for live/current events (only released for scored/historical periods) | Benchmark comparison; NOT a live-time feature |
| Analyst expectations / consensus | Not provided by either repo | External (not this API) | Only if sourced yourself, subject to knowledge cutoff | Could feed a surprise-magnitude estimate |
| Guidance changes | Only implicitly, inside the 10 facts (e.g., "guidance raised to $X from $Y") | Disclosure facts | Yes, if present in the facts | Directional signal for the LLM prompt |
| Fundamentals (balance sheet, historical financials) | Not provided by either repo | External | Only if sourced yourself, subject to cutoff | Contextual feature, e.g. valuation-adjusted reaction |

---

## 11. Current project state (known-good, as of this audit)

- **Tests**: the user has reported `uv run pytest` passes with **17 passed, 1
  warning** in the starter repo (`markets/`). This was not independently
  re-executed in this session because `uv` was not available on this shell's
  `PATH`; treat the 17/1 figure as user-reported ground truth pending
  re-verification with `uv` available. Source tests inspected and consistent
  with that count: `tests/test_predict.py` (1), `tests/test_modal_app.py` (7),
  `tests/test_webhook_verification.py` (9) = 17 total, matching.
- **Modal deployment**: user has reported `modal deploy modal_app.py`
  successfully creates the `predict_and_submit` function, the `web` ASGI app,
  and a public Modal URL. Consistent with `modal_app.py`'s structure
  (`@app.function` × 2, one wrapped in `@modal.asgi_app(label="explaining-markets")`).
- **Credentials** (values themselves never captured in this document):
  - `EM_API_KEY=<configured>`
  - `EM_WEBHOOK_SECRET=<configured>`
  - `OPENAI_API_KEY=<not configured>` — meaning `predict.py` currently runs in
    the **0.5 fallback baseline mode**, not making real LLM calls, per
    `.env` inspection (`OPENAI_API_KEY=` is blank).
  - `OPENAI_MODEL` / `EM_API_BASE_URL` — left at defaults (commented out in
    `.env`), i.e. `gpt-5.4-nano` / production base URL respectively, though
    since `OPENAI_API_KEY` is unset the model choice is currently moot.
- **Git**: repo has two commits (`Initial`, `fixed env`); `.env`/`.env.example`
  are gitignored in `markets/` (per its own `.gitignore`); the `examples/`
  directory is currently untracked (`??`) in `git status` — i.e., it has not
  yet been added/committed to this repo's git history. It is a nested git
  clone (has its own `.git`), so it should likely remain untracked or be
  handled as a submodule/gitignored path rather than accidentally committed
  wholesale — flagging this for the user's awareness, not fixing it here.

---

## 12. Known Unknowns

- Exact live event schedule / cadence (how many events per quarter, per
  event-type distribution) — not observable from either repo; only bundled
  sample data (13 calendar events, 5 archive records) was available.
- Exact real archive schema completeness (whether `event_returns`, `metrics`,
  `baseline_predictions` are present on every record or only for
  fully-scored/sealed quarters) — UNCONFIRMED, since the bundled sample lacks
  them entirely.
- Exact scoring implementation on the **live platform** — `examples/src/examples/scoring.py`
  is described as "an exact semantic port" and its numbers are asserted to
  match the leaderboard, but this repo cannot independently verify that claim
  against the actual production scorer's source.
- Whether the webhook-delivered `event` payload includes `knowledge_cutoff`
  (only the calendar API schema documents it explicitly).
- Exact `information_url` response shape (`{"summary": "..."}` vs a `facts`
  list vs something else) — `predict.py` defensively handles ambiguity here,
  suggesting even the starter's author wasn't fully certain.
- API access permission differences between the beta stage
  (`api-beta.explainingmarkets.ai`) and production
  (`api.explainingmarkets.ai`) — the examples repo defaults to beta ("open to
  invite-code holders"), the starter defaults to production; UNCONFIRMED how
  these two stages' data/availability differ.
- Whether additional external data sources/tools are permitted beyond the
  knowledge cutoff constraint — README says "no restrictions on data sources,
  models, or tools" subject to the cutoff, but exact enforcement/audit
  mechanics for the cutoff are unknown.
- Full asset universe (which tickers/companies are eligible focal assets) —
  not documented; only example tickers seen (AAPL, MSFT, GOOGL, TSLA, AMZN,
  BA, META, IBM, CVX, AMD, NVDA, QCOM in the sample calendar).
- API rate limits — not documented in either repo.
- Whether `event_type` values beyond `EARNINGS_RELEASE`/`TEST` currently exist
  in production (the schema is explicitly open-ended "for future-proofing,"
  but none other were observed).
- Exact behavior/format of `POST /webhook/test`'s enqueued delivery beyond
  "sends `event_type=TEST`" (e.g., whether it always carries a fixed
  `focal_assets` list).

---

## 13. Potential Strategy Directions (research directions only — nothing implemented)

1. **Earnings-surprise-aware LLM prompting** (extends the current baseline)
   - Data required: the 10 disclosure facts (already fetched) + possibly a
     self-sourced consensus/expectation figure.
   - Availability: facts are available now; consensus estimates are not
     provided by the API and would need an external, cutoff-respecting source.
   - Signal captured: magnitude/direction of the earnings **surprise**
     specifically, since the score explicitly benchmarks against
     `surprise_pct` — beating that benchmark is the entire point of
     `delta_r_squared`.
   - Leakage risk: low, if the consensus source is dated strictly before the
     event's `knowledge_cutoff`.
   - Difficulty: moderate — requires sourcing/parsing external consensus data
     and blending it into the existing `_ask_llm` prompt.

2. **Historical analog / calibration model trained on the archive**
   - Data required: `load_archive()` output across many quarters, with
     `event_returns.car1`, `metrics.earnings_surprise`, `disclosure` facts.
   - Availability: confirmed accessible via `GET /archive` with a live
     `EM_API_KEY`; UNCONFIRMED completeness of `event_returns`/`metrics` fields
     in practice.
   - Signal captured: an empirically calibrated mapping from
     (surprise magnitude, fact sentiment, sector, etc.) → historical
     `y` percentile distribution, potentially outperforming a zero-shot LLM
     call.
   - Leakage risk: moderate — must rank-features strictly within-period as
     `examples.scoring` does; must not use a future quarter's cross-section to
     rank a past quarter's outcome.
   - Difficulty: moderate-high — requires assembling a proper backtesting
     pipeline (already half-built for you in `examples.scoring` /
     `examples.archive`), plus a model choice (even simple OLS/logit on
     surprise + fact-derived features could beat the naive benchmark).

3. **Textual sentiment / fact-classification features**
   - Data required: the 10 disclosure facts.
   - Availability: available now, both live and archived.
   - Signal captured: a numeric feature vector (e.g., count of
     positive/negative-framed facts, presence of "raised guidance," "missed
     consensus," "record," "below expectations" phrasing) fed into a
     lightweight classifier/regressor, potentially cheaper and more
     interpretable than a single LLM call.
   - Leakage risk: low (facts are event-specific, already fair game).
   - Difficulty: low-moderate.

4. **Event-type-specific models**
   - Data required: `event_type` (currently only `EARNINGS_RELEASE` observed).
   - Availability: available now.
   - Signal captured: none additional today, since only one event type
     exists — but architecture should anticipate new event types (e.g., FOMC,
     CPI) given the explicitly open-ended `event_type` schema, and the reward
     of building type-specific prompts/models once new types appear.
   - Leakage risk: none.
   - Difficulty: low now, but requires future maintenance as new types arrive.

5. **Sector-relative / cross-sectional adjustment**
   - Data required: the current quarter's *other* events' focal assets and
     sectors, which are unknown at prediction time (that's the whole
     cross-section being ranked against) — but sector membership itself
     (independent of returns) could be sourced externally.
   - Availability: sector data not provided by the API; must self-source.
   - Signal captured: whether this asset's reaction is expected to be
     stronger/weaker than typical for its sector this quarter.
   - Leakage risk: low if sector data is static/objective (e.g., GICS
     classification), not derived from other competitors' realized outcomes.
   - Difficulty: moderate — requires an external sector-mapping data source.

6. **Ensembling the two published reference baselines**
   - Data required: `baseline_predictions` (`gemini/ea-explain-contemp-summary`,
     `openai/ea-explain-contemp-summary`) — but only available in **historical
     archive** data, not live events.
   - Availability: historical only — cannot be used live (the platform does
     not surface other submissions' live predictions to you).
   - Signal captured: historically, how well each baseline correlates with
     realized outcomes, useful for **benchmarking your own model's historical
     performance**, not as a live feature.
   - Leakage risk: N/A for live use since it's simply unavailable at
     inference time; using it in offline backtesting is fine.
   - Difficulty: low (already exposed via `BASELINE_LABELS` /
     `event_asset_rows`).

7. **Calibration-aware ensembling despite the score's affine invariance**
   - Data required: none beyond your own model.
   - Availability: n/a.
   - Signal captured: since the score is invariant to affine remaps of your
     predictions, the practical lesson is to **maximize correlation/rank
     accuracy** rather than spend effort on calibrating the numeric scale —
     effort is better spent on getting the *relative ordering* of
     predictions right across events than on getting individual numbers
     "properly percentile-shaped."
   - Leakage risk: none — this is a modeling-priorities insight, not a data
     source.
   - Difficulty: n/a — informs how you should prioritize modeling effort
     (rank correlation > calibration).

8. **Concurrency-aware asset processing (engineering, not modeling)**
   - Not a modeling direction but flagged because `predict.py`'s own comments
     note that if events start carrying multiple focal assets, the current
     serial per-asset LLM calls could exceed the 5-minute prediction window;
     any strategy upgrade should also parallelize `_ask_llm` calls per asset
     (e.g., via `asyncio.gather` or a thread pool) to stay safe.

---

## 14. AI Handoff Summary

**What are we building?** A Modal-deployed webhook receiver + prediction agent
for the Explaining Markets competition: it verifies signed event webhooks,
ACKs within 20 seconds, then (within a 5-minute window) predicts a
cross-sectional percentile rank of each focal asset's next-day abnormal return
and submits it back to the competition API. `predict.py` is the only file
meant to contain the actual prediction strategy.

**How does it receive events?** The competition platform POSTs an
HMAC-SHA256-signed (Standard-Webhooks style) JSON payload to the Modal-hosted
FastAPI route (`POST /`, aliased at `/competition/webhook`). `modal_app.py`
verifies the raw request bytes via `verify_webhook()`, checks/claims
idempotency via a `modal.Dict`, ACKs 200, then spawns `predict_and_submit` in
its own container.

**What exactly must `predict()` produce?** `list[{"identifier_value": str,
"predicted_percentile": float in [0,1]}]`, one entry per `event["focal_assets"]`.
`predicted_percentile` is a **cross-sectional rank** of the asset's next-day
abnormal (market-adjusted) return across all of the quarter's scored event
outcomes (0 = worst, 0.5 = median, 1 = best) — not a percentile within the
asset's own history.

**What information can it access?** The verified webhook `event` (metadata +
`information_url` pointing to the event's summary/disclosure — a
model-generated, ten-sentence, transcript-only distillation with no injected
market data, ticker, date, or estimates); the historical archive
(`GET /archive`) for offline backtesting/model-fitting; the calendar
(`GET /events`, which does document a `knowledge_cutoff` field); and, subject
to the cutoff, any external data source of the competitor's choosing.

**What information must it avoid?** Anything dated after the event's
`knowledge_cutoff`. That field is confirmed present on the calendar API
response but **not confirmed present on the webhook-delivered event payload
itself** — worth verifying against a real delivery before assuming it's
absent or present.

**How are predictions submitted?** `POST {EM_API_BASE_URL}/predictions` with
`X-API-Key`, body `{"event_id", "predictions": [...]}`. Only the first
submission per event is scored; TEST events are accepted but never scored;
late/duplicate/pre-broadcast submissions are tagged, not rejected.

**What historical data is available?** A per-quarter, per-event-type
gzip-JSONL archive (`GET /archive`) with realized events, their disclosure
facts, and — for scored quarters — `event_returns.car1` (realized outcome),
`metrics.earnings_surprise` (naive benchmark), and `baseline_predictions`
(two named reference LLM baselines). The bundled sample in this repo's
`examples/data/sample/` is intentionally minimal and lacks the returns/scoring
fields — treat the live API as ground truth.

**What APIs/data sources are available?** The Explaining Markets competition
API (`/events`, `/health`, `/archive`, `/archive/{type}/{quarter}`,
`/webhook/test`, `/predictions`); OpenAI (for the current baseline's LLM
call); Anthropic (only used by the `examples` repo to *demonstrate* how the
platform's own disclosure facts are generated — not needed for the
competitor's own strategy).

**What does the current baseline do?** If `OPENAI_API_KEY` is unset (current
state in this repo's `.env`), it submits a flat `0.5` for every asset — a
neutral placeholder that exercises the full pipeline without any real
signal. If a key is set, it fetches the event's disclosure summary and asks
`gpt-5.4-nano` (or configured model) for a single calibrated percentile via
structured-output decoding, guided by a hand-written system prompt with
explicit calibration discipline.

**What are the most promising next research directions?** (1) exploiting the
score's explicit surprise-benchmark structure by better estimating/using
earnings surprise magnitude; (2) building an offline backtest/calibration
model on the historical archive using the already-ported `examples.scoring`
transform; (3) turning the ten disclosure facts into structured
sentiment/magnitude features rather than relying solely on a single zero-shot
LLM call; (4) remembering that the score rewards **rank correlation, not
calibration** — a submission's affine scale/direction doesn't matter, only
how much it explains of the realized cross-sectional ranking beyond the
surprise benchmark; (5) never submitting constant predictions, since they
score exactly zero.

**What are the biggest unknowns?** Real archive schema completeness for
`event_returns`/`metrics`/`baseline_predictions`; whether the webhook payload
itself carries `knowledge_cutoff`; the exact `information_url` response
shape; the beta-vs-production API stage differences; the true event calendar
scale/asset universe; and whether the production scoring code is truly
byte-identical to the ported `examples.scoring` module (strongly implied but
not independently verifiable from these two repos alone).
