# Company history cache

Offline SQLite cache of per-ticker daily adjusted prices and historical
earnings events, read at live-prediction time by
`explaining_markets.data_providers.cache.CompanyHistoryCache`.

- Default file: `company_history.sqlite` (gitignored — never commit bulk data).
- Schema and point-in-time rules: see the module docstring in
  `src/explaining_markets/data_providers/cache.py`.
- Populate it OFFLINE from a real vendor implementing
  `MarketDataProvider` / `EarningsDataProvider`
  (`src/explaining_markets/data_providers/protocols.py`). No vendor
  credentials are configured in this repository yet, so this cache is empty
  by default and the live model treats price/vendor-earnings features as
  missing (explicit availability indicators, neutral imputation).
- Every row carries `source`, `available_at`, and `retrieved_at` provenance;
  records whose availability cannot be established before an event's cutoff
  are excluded (fail closed).
