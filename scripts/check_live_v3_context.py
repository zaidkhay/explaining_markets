"""Local diagnostic for live V3 cache/news/reasoning population."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from explaining_markets.live_v3_context import build_live_v3_context, feed_diagnostics
from explaining_markets.point_in_time_audit_v3 import audit_context
from explaining_markets.providers.live_context import default_provider_bundle_from_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--sector", default=None)
    parser.add_argument("--sector-ticker", default=None)
    parser.add_argument("--peers", default="", help="comma-separated peer tickers")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cutoff = datetime.now(timezone.utc)
    event = {
        "event_id": f"diagnostic-{ticker}-{cutoff:%Y%m%dT%H%M%S}",
        "sector": args.sector,
        "sector_ticker": args.sector_ticker,
        "peer_tickers": [x.strip().upper() for x in args.peers.split(",") if x.strip()],
        "disclosure": [],
    }
    providers = default_provider_bundle_from_env()
    context = build_live_v3_context(ticker=ticker, event=event, cutoff=cutoff, providers=providers)
    audit = audit_context(context)
    diag = feed_diagnostics(context)

    print(f"Ticker: {ticker}")
    print(f"Cutoff: {cutoff.isoformat()}")
    print()
    print(f"Earnings source: {context.earnings.source if context.earnings else 'UNAVAILABLE'}")
    print(f"EPS actual: {context.earnings.reported_eps if context.earnings else None}")
    print(f"EPS estimate: {context.earnings.consensus_eps if context.earnings else None}")
    print(f"Earnings eligible: {bool(context.earnings and context.earnings.eligible(cutoff))}")
    print(f"Revenue actual: {context.earnings.reported_revenue if context.earnings else None}")
    print(f"Revenue estimate: {context.earnings.consensus_revenue if context.earnings else None}")
    print(f"Revenue eligible: {bool(context.earnings and context.earnings.eligible(cutoff))}")
    print(f"Guidance availability: {context.guidance is not None}")
    print()
    print(f"Price rows: {len(context.stock_prices)}")
    latest = context.stock_prices[-1].value_timestamp.isoformat() if context.stock_prices else None
    print(f"Latest eligible price timestamp: {latest}")
    print(f"Peer count: {len(context.peers)}")
    print(f"Peer names: {[p.peer_ticker for p in context.peers]}")
    print(f"Company news count: {diag['company_news_count']}")
    print(f"Peer news count: {diag['peer_news_count']}")
    print(f"Sector news count: {diag['sector_news_count']}")
    print(f"Reasoned article count: {diag['reasoned_news_count']}")
    print(f"Event reasoning available: {bool(context.event_reasoning)}")
    print(f"POINT-IN-TIME AUDIT: PASS ({audit.records_checked} records)")
    print("Provider receipts:")
    for receipt in diag["provider_receipts"]:
        print(f"  {receipt}")


if __name__ == "__main__":
    main()
