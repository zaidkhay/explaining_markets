#!/usr/bin/env python3
"""Bounded Twelve Data historical-price entitlement smoke test.

Uses five symbols (five API credits) and never prints the API key. The chosen
symbols mix a mega-cap with names that were problematic or lower coverage in
other free providers.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from explaining_markets.historical_v3_enrichment import DEFAULT_CACHE_DIR, DiskJsonCache
from explaining_markets.providers.free_historical import (
    TwelveDataHistoricalClient,
    twelve_data_price_records,
)

SYMBOLS = ("AAPL", "BLK", "SOTK", "BYRN", "NEOG")


def main() -> int:
    load_dotenv()
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    print("=== TWELVE DATA CONFIG ===")
    print(f"TWELVE_DATA_API_KEY: {'PRESENT' if api_key else 'MISSING'}")
    print(f"symbols: {list(SYMBOLS)}")
    if not api_key:
        return 2

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=5 * 366 + 45)
    cache = DiskJsonCache(DEFAULT_CACHE_DIR)
    client = TwelveDataHistoricalClient(
        api_key,
        cache=cache,
        max_api_calls=len(SYMBOLS),
        timeout=20.0,
    )

    passed = 0
    five_year = 0
    try:
        for symbol in SYMBOLS:
            print(f"\n=== {symbol} ===")
            try:
                payload = client.prices_payload(symbol, start=start, end=now)
                rows = twelve_data_price_records(payload, symbol, retrieved_at=now)
                eligible = [row for row in rows if row.eligible(now)]
                is_5y = len(eligible) >= 1261
                print(f"raw rows: {len(payload)}")
                print(f"normalized rows: {len(rows)}")
                print(f"first: {rows[0].value_timestamp.date() if rows else None}")
                print(f"last: {rows[-1].value_timestamp.date() if rows else None}")
                print(f"source: {rows[-1].source if rows else None}")
                print(f"5y_feature_eligible: {is_5y}")
                if rows:
                    passed += 1
                if is_5y:
                    five_year += 1
            except Exception as exc:
                print(f"FAIL: {type(exc).__name__}: {exc}")
    finally:
        client.close()

    print("\n=== SUMMARY ===")
    print(f"symbols_with_prices: {passed}/{len(SYMBOLS)}")
    print(f"symbols_with_5y_history: {five_year}/{len(SYMBOLS)}")
    print(f"api_calls: {client.api_calls}")
    print(f"cache_hits: {client.cache_hits}")
    print(f"provider_blocked_reason: {client.unavailable_reason}")
    print("\n=== FINAL ===")
    ok = passed == len(SYMBOLS)
    print("PASS" if ok else "PARTIAL/FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
