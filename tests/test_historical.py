"""Historical event loading: parsing, tolerance, and non-fatal empty-dir behavior."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from explaining_markets.historical import (
    HistoricalEvent,
    labeled_events,
    load_historical_events,
    read_jsonl_gz,
)


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _record(event_id: str, ticker: str, *, car1=None, surprise=None, surprise_ok=True) -> dict:
    rec: dict = {
        "event_id": event_id,
        "event_type": "EARNINGS_RELEASE",
        "event_datetime": "2025-07-31T21:00:00Z",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
        "disclosure": {"items": [{"kind": "facts", "content": ["Revenue beat.", "Guidance raised."]}]},
    }
    if car1 is not None:
        rec["event_returns"] = {ticker: {"car1": car1, "return_status": "ok"}}
    if surprise is not None:
        rec["metrics"] = {
            "earnings_surprise": {
                "surprise": surprise,
                "surprise_status": "ok" if surprise_ok else "unavailable",
            }
        }
    return rec


def test_missing_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_historical_events(tmp_path / "does_not_exist") == []


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_historical_events(tmp_path) == []


def test_loads_records_and_derives_quarter_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "EARNINGS_RELEASE_2025Q3.jsonl.gz"
    _write_jsonl_gz(path, [_record("e1", "AAPL", car1=0.05, surprise=0.01)])

    events = load_historical_events(tmp_path)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, HistoricalEvent)
    assert event.event_id == "e1"
    assert event.ticker == "AAPL"
    assert event.event_type == "EARNINGS_RELEASE"
    assert event.quarter == "2025Q3"
    assert event.car1 == 0.05
    assert event.earnings_surprise == 0.01
    assert event.disclosure == ["Revenue beat.", "Guidance raised."]


def test_multiple_focal_assets_become_multiple_events(tmp_path: Path) -> None:
    record = _record("e1", "AAPL", car1=0.05)
    record["focal_assets"].append({"identifier_type": "TICKER", "identifier_value": "MSFT"})
    record["event_returns"]["MSFT"] = {"car1": -0.02, "return_status": "ok"}
    path = tmp_path / "EARNINGS_RELEASE_2025Q3.jsonl.gz"
    _write_jsonl_gz(path, [record])

    events = load_historical_events(tmp_path)
    tickers = {e.ticker for e in events}
    assert tickers == {"AAPL", "MSFT"}
    car1_by_ticker = {e.ticker: e.car1 for e in events}
    assert car1_by_ticker == {"AAPL": 0.05, "MSFT": -0.02}


def test_missing_car1_and_surprise_are_none(tmp_path: Path) -> None:
    path = tmp_path / "EARNINGS_RELEASE_2025Q3.jsonl.gz"
    _write_jsonl_gz(path, [_record("e1", "AAPL")])  # no returns/metrics at all

    (event,) = load_historical_events(tmp_path)
    assert event.car1 is None
    assert event.earnings_surprise is None


def test_surprise_status_not_ok_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "EARNINGS_RELEASE_2025Q3.jsonl.gz"
    _write_jsonl_gz(path, [_record("e1", "AAPL", car1=0.01, surprise=0.02, surprise_ok=False)])

    (event,) = load_historical_events(tmp_path)
    assert event.car1 == 0.01
    assert event.earnings_surprise is None


def test_labeled_events_filters_on_car1_presence(tmp_path: Path) -> None:
    path = tmp_path / "EARNINGS_RELEASE_2025Q3.jsonl.gz"
    _write_jsonl_gz(
        path,
        [_record("e1", "AAPL", car1=0.01), _record("e2", "MSFT")],  # e2 has no car1
    )

    events = load_historical_events(tmp_path)
    assert len(events) == 2
    labeled = labeled_events(events)
    assert len(labeled) == 1
    assert labeled[0].event_id == "e1"


def test_malformed_file_is_skipped_not_raised(tmp_path: Path) -> None:
    good = tmp_path / "EARNINGS_RELEASE_2025Q3.jsonl.gz"
    _write_jsonl_gz(good, [_record("e1", "AAPL", car1=0.01)])
    bad = tmp_path / "EARNINGS_RELEASE_2025Q4.jsonl.gz"
    with gzip.open(bad, "wt", encoding="utf-8") as fh:
        fh.write("not valid json\n")

    events = load_historical_events(tmp_path)
    assert len(events) == 1
    assert events[0].event_id == "e1"


def test_read_jsonl_gz_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")
    assert list(read_jsonl_gz(path)) == [{"a": 1}, {"a": 2}]
