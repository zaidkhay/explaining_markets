#!/usr/bin/env python3
"""Make one bounded live smoke test against Tiingo, Finnhub and OpenRouter.

No credentials are printed. The script uses at most one Tiingo price request,
two Finnhub requests, and one OpenRouter request (unless responses are cached).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from explaining_markets.historical_v3_enrichment import DEFAULT_CACHE_DIR, DiskJsonCache
from explaining_markets.providers.free_historical import (
    FinnhubHistoricalClient,
    TiingoHistoricalClient,
    finnhub_news_records,
    tiingo_price_records,
)
from explaining_markets.reasoning.openrouter_client import openrouter_api_key, openrouter_model, structured_json


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def main() -> int:
    load_dotenv()
    tiingo_key = _env("TINGO_API", "TIINGO_API_KEY", "TIINGO_API")
    finnhub_key = _env("FINNHUB_API_KEY", "FINNHUBB_API")
    or_key = openrouter_api_key()

    print("=== CONFIG ===")
    print(f"Tiingo: {'PRESENT' if tiingo_key else 'MISSING'}")
    print(f"Finnhub: {'PRESENT' if finnhub_key else 'MISSING'}")
    print(f"OpenRouter: {'PRESENT' if or_key else 'MISSING'}")
    print(f"OpenRouter model: {openrouter_model()}")

    cache = DiskJsonCache(DEFAULT_CACHE_DIR)
    now = datetime.now(timezone.utc)
    ok = True

    if tiingo_key:
        client = TiingoHistoricalClient(tiingo_key, cache=cache, max_api_calls=1, timeout=15.0)
        try:
            start = now - timedelta(days=5 * 366 + 45)
            payload = client.prices_payload("AAPL", start=start, end=now)
            rows = tiingo_price_records(payload, "AAPL", retrieved_at=now)
            eligible = [row for row in rows if row.available_at <= now]
            print("\n=== TIINGO EOD ===")
            print(f"adjusted rows: {len(rows)}")
            print(f"eligible rows: {len(eligible)}")
            print(f"first: {rows[0].value_timestamp.date() if rows else None}")
            print(f"last: {rows[-1].value_timestamp.date() if rows else None}")
            print(f"source: {rows[-1].source if rows else None}")
            ok = ok and len(rows) >= 1000
        except Exception as exc:
            print(f"\nTIINGO FAIL: {type(exc).__name__}: {exc}")
            ok = False
        finally:
            client.close()
    else:
        ok = False

    if finnhub_key:
        client = FinnhubHistoricalClient(finnhub_key, cache=cache, max_api_calls=2, timeout=15.0)
        try:
            earnings = client.earnings_payload("AAPL")
            news_start = now - timedelta(days=30)
            raw_news = client.company_news_payload("AAPL", start=news_start, end=now)
            news = finnhub_news_records(raw_news, "AAPL", retrieved_at=now)
            print("\n=== FINNHUB ===")
            print(f"earnings rows: {len(earnings)}")
            print(f"earnings periods: {[row.get('period') for row in earnings[:4]]}")
            print(f"30d company news rows: {len(news)}")
            print(f"latest headline: {news[-1].headline[:160] if news else None}")
            ok = ok and len(earnings) > 0 and len(news) > 0
        except Exception as exc:
            print(f"\nFINNHUB FAIL: {type(exc).__name__}: {exc}")
            ok = False
        finally:
            client.close()
    else:
        ok = False

    if or_key:
        try:
            result = structured_json(
                schema_name="market_reasoning_smoke",
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "direction": {"type": "number", "minimum": -1, "maximum": 1},
                        "materiality": {"type": "number", "minimum": 0, "maximum": 1},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string", "maxLength": 180},
                    },
                    "required": ["direction", "materiality", "confidence", "rationale"],
                },
                system_prompt=(
                    "Use only the supplied synthetic headline. Return bounded market-reaction features; "
                    "do not use outside facts and do not predict a competition percentile."
                ),
                user_payload={"headline": "Synthetic company beats EPS expectations and raises guidance."},
            )
            print("\n=== OPENROUTER ===")
            print(f"structured output: {result}")
            ok = ok and all(name in result for name in ("direction", "materiality", "confidence", "rationale"))
        except Exception as exc:
            print(f"\nOPENROUTER FAIL: {type(exc).__name__}: {exc}")
            ok = False
    else:
        ok = False

    print("\n=== FINAL ===")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
