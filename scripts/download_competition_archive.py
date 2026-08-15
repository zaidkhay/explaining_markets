"""Download sealed competition archive files for offline training.

Requires EM_API_KEY. Credentials are loaded from `.env` when present and are
never logged or persisted. By default all sealed event types are downloaded
for the requested quarters so within-quarter CAR1 percentile labels are ranked
against the complete competition cross-section. Feature-row construction may
still filter to a narrower event type such as EARNINGS_RELEASE.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from explaining_markets.config import DEFAULT_API_BASE_URL

DEFAULT_QUARTERS = ("2025Q4", "2026Q1", "2026Q2")


def download(
    base_url: str,
    destination: str | Path,
    *,
    quarters: tuple[str, ...] = DEFAULT_QUARTERS,
    event_types: tuple[str, ...] = (),
) -> list[Path]:
    load_dotenv()
    api_key = os.environ.get("EM_API_KEY")
    if not api_key:
        raise RuntimeError("EM_API_KEY is required; add it to .env")
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"X-API-Key": api_key}
    base = base_url.rstrip("/")
    requested_quarters = set(quarters)
    requested_types = set(event_types)

    with httpx.Client(headers=headers, follow_redirects=True, timeout=120.0) as client:
        response = client.get(f"{base}/archive")
        response.raise_for_status()
        manifest = response.json()
        files = manifest.get("files") if isinstance(manifest, dict) else manifest
        if not isinstance(files, list):
            raise RuntimeError("archive manifest has unexpected shape")

        selected = [
            item for item in files
            if item.get("quarter") in requested_quarters
            and item.get("sealed") is not False
            and (not requested_types or item.get("event_type") in requested_types)
        ]
        found_quarters = {item.get("quarter") for item in selected}
        missing_quarters = requested_quarters - found_quarters
        if missing_quarters:
            raise RuntimeError(f"sealed archive missing required quarters: {sorted(missing_quarters)}")
        if requested_types:
            found_types = {item.get("event_type") for item in selected}
            missing_types = requested_types - found_types
            if missing_types:
                raise RuntimeError(f"sealed archive missing required event types: {sorted(missing_types)}")

        paths: list[Path] = []
        for item in sorted(selected, key=lambda value: (value["quarter"], value.get("event_type", ""))):
            quarter = str(item["quarter"])
            event_type = str(item.get("event_type") or "UNKNOWN")
            url = item.get("url")
            if not url:
                fresh = client.get(f"{base}/archive/{event_type}/{quarter}")
                fresh.raise_for_status()
                url = fresh.json()["url"]
            target = dest / f"{event_type}_{quarter}.jsonl.gz"
            data = client.get(url)
            if data.status_code in {403, 404}:
                fresh = client.get(f"{base}/archive/{event_type}/{quarter}")
                fresh.raise_for_status()
                data = client.get(fresh.json()["url"])
            data.raise_for_status()
            target.write_bytes(data.content)
            paths.append(target)
            print(f"downloaded {event_type} {quarter}: {len(data.content):,} bytes")
        return paths


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL),
    )
    parser.add_argument("--dest", default="data/historical")
    parser.add_argument(
        "--quarters",
        default=",".join(DEFAULT_QUARTERS),
        help="comma-separated sealed quarters to download",
    )
    parser.add_argument(
        "--event-types",
        default="",
        help="optional comma-separated event types; default downloads all sealed event types",
    )
    args = parser.parse_args()
    quarters = tuple(value.strip() for value in args.quarters.split(",") if value.strip())
    event_types = tuple(value.strip() for value in args.event_types.split(",") if value.strip())
    paths = download(args.base_url, args.dest, quarters=quarters, event_types=event_types)
    print(f"downloaded {len(paths)} sealed archive files")


if __name__ == "__main__":
    main()
