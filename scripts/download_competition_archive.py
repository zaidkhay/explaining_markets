"""Download selected sealed competition archive files for offline training.

Requires EM_API_KEY. No credentials are logged or persisted. The downloader
accepts either production or beta API base URLs and stores only gzip JSONL
files under the requested local directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

QUARTERS = {"2025Q4", "2026Q1", "2026Q2"}
EVENT_TYPE = "EARNINGS_RELEASE"


def download(base_url: str, destination: str | Path) -> list[Path]:
    api_key = os.environ.get("EM_API_KEY")
    if not api_key:
        raise RuntimeError("EM_API_KEY is required")
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"X-API-Key": api_key}
    base = base_url.rstrip("/")

    with httpx.Client(headers=headers, follow_redirects=True, timeout=120.0) as client:
        response = client.get(f"{base}/archive")
        response.raise_for_status()
        manifest = response.json()
        files = manifest.get("files") if isinstance(manifest, dict) else manifest
        if not isinstance(files, list):
            raise RuntimeError("archive manifest has unexpected shape")

        selected = [
            item for item in files
            if item.get("event_type") == EVENT_TYPE
            and item.get("quarter") in QUARTERS
            and item.get("sealed") is not False
        ]
        found = {item.get("quarter") for item in selected}
        missing = QUARTERS - found
        if missing:
            raise RuntimeError(f"sealed archive missing required quarters: {sorted(missing)}")

        paths: list[Path] = []
        for item in sorted(selected, key=lambda x: x["quarter"]):
            quarter = item["quarter"]
            url = item.get("url")
            if not url:
                fresh = client.get(f"{base}/archive/{EVENT_TYPE}/{quarter}")
                fresh.raise_for_status()
                url = fresh.json()["url"]
            target = dest / f"{EVENT_TYPE}_{quarter}.jsonl.gz"
            data = client.get(url)
            if data.status_code in {403, 404}:
                fresh = client.get(f"{base}/archive/{EVENT_TYPE}/{quarter}")
                fresh.raise_for_status()
                data = client.get(fresh.json()["url"])
            data.raise_for_status()
            target.write_bytes(data.content)
            paths.append(target)
            print(f"downloaded {quarter}: {len(data.content):,} bytes")
        return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("EM_API_BASE_URL", "https://api-beta.explainingmarkets.ai/v1"))
    parser.add_argument("--dest", default="data/historical")
    args = parser.parse_args()
    paths = download(args.base_url, args.dest)
    print(f"downloaded {len(paths)} sealed quarter files")


if __name__ == "__main__":
    main()
