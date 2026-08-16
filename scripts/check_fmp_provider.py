#!/usr/bin/env python3
"""Make one bounded FMP historical-EOD smoke test without printing secrets."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from explaining_markets.historical_v3_enrichment import DEFAULT_CACHE_DIR, DiskJsonCache
from explaining_markets.providers.free_historical import FmpHistoricalClient, fmp_price_records


def main() -> int:
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    print("=== FMP CONFIG ===")
    print(f"FMP_API_KEY: {'PRESENT' if api_key else 'MISSING'}")
    if not api_key:
        return 2

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=5 * 366 + 45)
    cache = DiskJsonCache(DEFAULT_CACHE_DIR)
    client = FmpHistoricalClient(api_key, cache=cache, max_api_calls=1, timeout=15.0)
    try:
        payload = client.prices_payload("AAPL", start=start, end=now)
        rows = fmp_price_records(payload, "AAPL", retrieved_at=now)
        eligible = [row for row in rows if row.eligible(now)]
        print("\n=== FMP HISTORICAL EOD ===")
        print(f"raw rows: {len(payload)}")
        print(f"normalized rows: {len(rows)}")
        print(f"eligible rows: {len(eligible)}")
        print(f"first: {rows[0].value_timestamp.date() if rows else None}")
        print(f"last: {rows[-1].value_timestamp.date() if rows else None}")
        print(f"source: {rows[-1].source if rows else None}")
        print(f"5y_feature_eligible: {len(eligible) >= 1261}")
        print(f"api_calls: {client.api_calls}")
        print(f"cache_hits: {client.cache_hits}")
        ok = len(rows) > 0
    except Exception as exc:
        print(f"\nFMP FAIL: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        client.close()

    print("\n=== FINAL ===")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
