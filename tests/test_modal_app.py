"""The ASGI app guards the webhook route and ACKs before predicting.

Modal wraps `web` and `predict_and_submit` in `modal.Function` objects;
`get_raw_f()` unwraps them so the handler logic can be exercised offline,
without a Modal connection. `.spawn()` is patched out — what matters here is
that the route ACKs and hands the work off, not that Modal dispatches it.
"""

from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

import modal_app

SECRET = "whsec_" + base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("=")


def _signed(payload: dict) -> tuple[bytes, dict]:
    raw_body = json.dumps(payload).encode()
    ts = int(time.time())
    key = base64.urlsafe_b64decode(SECRET[len("whsec_"):] + "==")
    signed = f"{payload['id']}.{ts}.".encode() + raw_body
    signature = "v1," + base64.b64encode(hmac.new(key, signed, sha256).digest()).decode()
    return raw_body, {
        "webhook-id": payload["id"],
        "webhook-timestamp": str(ts),
        "webhook-signature": signature,
    }


def _test_event(webhook_id: str = "evt_test_abc") -> dict:
    return {
        "id": webhook_id,
        "event_id": "test_abc",
        "event_type": "TEST",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "TEST"}],
        "prediction_deadline": "2099-01-01T00:00:00+00:00",
    }


class _Put:
    """Callable with an `.aio` attribute, mirroring modal.Dict.put."""

    def __init__(self, store):
        self.store = store

    def __call__(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self.store:
            return False
        self.store[key] = value
        return True

    async def aio(self, key, value, *, skip_if_exists=False):
        return self(key, value, skip_if_exists=skip_if_exists)


class FakeDict:
    """Stand-in for modal.Dict with the same atomic-claim semantics."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.put = _Put(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value

    def pop(self, key, default=None):
        return self.data.pop(key, default)


class _Spawn:
    """Callable with an `.aio` attribute, mirroring modal.Function.spawn."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)

    async def aio(self, *args, **kwargs):
        return self(*args, **kwargs)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EM_API_KEY", "test-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(modal_app, "seen_webhooks", FakeDict())
    return TestClient(modal_app.web.get_raw_f()())


@pytest.fixture
def spawned(monkeypatch):
    """Record spawn() calls instead of dispatching them to Modal."""
    spawn = _Spawn()
    monkeypatch.setattr(modal_app.predict_and_submit, "spawn", spawn)
    return spawn.calls


def test_health_ok(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_unsigned_post_is_rejected(client, spawned) -> None:
    for path in ("/", "/competition/webhook"):
        assert client.post(path, content=b"{}").status_code == 401
    assert spawned == []


def test_signed_event_acks_and_spawns_the_work(client, spawned) -> None:
    raw_body, headers = _signed(_test_event())
    resp = client.post("/", content=raw_body, headers=headers)
    assert resp.status_code == 200
    # The route did not predict inline — it handed off exactly one job.
    assert len(spawned) == 1
    event, webhook_id, slot = spawned[0]
    assert event["event_id"] == "test_abc"
    assert webhook_id == "evt_test_abc"
    assert slot is None
    assert modal_app.seen_webhooks.data["evt_test_abc"] == "in_flight"


def test_duplicate_while_in_flight_is_skipped(client, spawned) -> None:
    raw_body, headers = _signed(_test_event())
    assert client.post("/", content=raw_body, headers=headers).status_code == 200
    assert client.post("/", content=raw_body, headers=headers).status_code == 200
    assert len(spawned) == 1  # the redelivery spawned nothing


def test_route_uses_the_async_modal_interfaces() -> None:
    """Regression guard: blocking Modal calls inside the `async def` route stall
    the event loop -- the exact failure ACKing first is meant to remove. Modal
    only warns about this at runtime, so pin it here."""
    import inspect

    src = inspect.getsource(modal_app.web.get_raw_f())
    assert "await _claim_aio(" in src
    assert "await predict_and_submit.spawn.aio(" in src
    assert "seen_webhooks.put(" not in src


def test_claim_release_state_machine(monkeypatch) -> None:
    """in_flight -> done on success; in_flight -> absent on failure."""
    fake = FakeDict()
    monkeypatch.setattr(modal_app, "seen_webhooks", fake)

    assert modal_app._claim("evt_1") is True
    assert modal_app._claim("evt_1") is False  # already in flight
    modal_app._release("evt_1", True)
    assert fake.data["evt_1"] == "done"
    assert modal_app._claim("evt_1") is False  # done is permanent

    assert modal_app._claim("evt_2") is True
    modal_app._release("evt_2", False)
    assert "evt_2" not in fake.data  # released, so a redelivery can retry
    assert modal_app._claim("evt_2") is True


def test_worker_submits_and_marks_done(monkeypatch) -> None:
    """The spawned function submits a neutral prediction for a TEST event and
    marks the claim done; a raising submit releases it instead."""
    monkeypatch.setenv("EM_API_KEY", "test-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", SECRET)
    fake = FakeDict()
    monkeypatch.setattr(modal_app, "seen_webhooks", fake)

    submitted: list[dict] = []
    import explaining_markets.client as client_mod

    monkeypatch.setattr(
        client_mod,
        "submit_predictions",
        lambda *, event_id, predictions, config: submitted.append(
            {"event_id": event_id, "predictions": predictions}
        ),
    )

    worker = modal_app.predict_and_submit.get_raw_f()
    fake.put("evt_test_abc", "in_flight")
    worker(_test_event(), "evt_test_abc")

    assert submitted == [
        {
            "event_id": "test_abc",
            "predictions": [{"identifier_value": "TEST", "predicted_percentile": 0.5}],
        }
    ]
    assert fake.data["evt_test_abc"] == "done"

    def broken(**kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(client_mod, "submit_predictions", broken)
    fake.put("evt_test_def", "in_flight")
    worker(_test_event("evt_test_def"), "evt_test_def")
    assert "evt_test_def" not in fake.data


# --- Additional submission slots (/s/2 .. /s/5) -------------------------------

SLOT_SECRET = "whsec_" + base64.urlsafe_b64encode(b"s" * 32).decode().rstrip("=")


def _signed_with(secret: str, payload: dict) -> tuple[bytes, dict]:
    raw_body = json.dumps(payload).encode()
    ts = int(time.time())
    key = base64.urlsafe_b64decode(secret[len("whsec_"):] + "==")
    signed = f"{payload['id']}.{ts}.".encode() + raw_body
    signature = "v1," + base64.b64encode(hmac.new(key, signed, sha256).digest()).decode()
    return raw_body, {
        "webhook-id": payload["id"],
        "webhook-timestamp": str(ts),
        "webhook-signature": signature,
    }


@pytest.fixture
def slot_client(monkeypatch, client):
    for slot in (2, 3):
        monkeypatch.setenv(f"EM_API_KEY_S{slot}", f"key-s{slot}")
        monkeypatch.setenv(f"EM_WEBHOOK_SECRET_S{slot}", SLOT_SECRET)
    for slot in (4, 5):
        monkeypatch.delenv(f"EM_API_KEY_S{slot}", raising=False)
        monkeypatch.delenv(f"EM_WEBHOOK_SECRET_S{slot}", raising=False)
    return client


def test_unconfigured_or_unknown_slot_is_404(slot_client, spawned) -> None:
    for path in ("/s/1", "/s/4", "/s/5", "/s/6", "/s/0"):
        assert slot_client.post(path, content=b"{}").status_code == 404, path
    assert spawned == []


def test_slot_with_only_one_credential_is_404(slot_client, spawned, monkeypatch) -> None:
    monkeypatch.setenv("EM_API_KEY_S4", "key-s4")
    assert slot_client.post("/s/4", content=b"{}").status_code == 404
    assert spawned == []


def test_unsigned_post_to_configured_slot_is_401(slot_client, spawned) -> None:
    assert slot_client.post("/s/2", content=b"{}").status_code == 401
    assert spawned == []


def test_slot_rejects_the_legacy_secret(slot_client, spawned) -> None:
    raw_body, headers = _signed(_test_event())
    assert slot_client.post("/s/2", content=raw_body, headers=headers).status_code == 401
    assert spawned == []


def test_legacy_route_rejects_a_slot_secret(slot_client, spawned) -> None:
    raw_body, headers = _signed_with(SLOT_SECRET, _test_event())
    assert slot_client.post("/", content=raw_body, headers=headers).status_code == 401
    assert spawned == []


def test_signed_slot_event_acks_and_spawns_with_its_slot(slot_client, spawned) -> None:
    raw_body, headers = _signed_with(SLOT_SECRET, _test_event())
    assert slot_client.post("/s/3", content=raw_body, headers=headers).status_code == 200
    assert len(spawned) == 1
    event, webhook_id, slot = spawned[0]
    assert event["event_id"] == "test_abc"
    assert webhook_id == "s3:evt_test_abc"
    assert slot == 3


def test_slot_claims_are_independent(slot_client, spawned) -> None:
    """The same delivery id on two slots (or legacy) is two submissions, not a
    duplicate -- each must spawn its own job. Redelivery to one slot is still
    deduped."""
    legacy_body, legacy_headers = _signed(_test_event())
    slot_body, slot_headers = _signed_with(SLOT_SECRET, _test_event())
    assert slot_client.post("/", content=legacy_body, headers=legacy_headers).status_code == 200
    assert slot_client.post("/s/2", content=slot_body, headers=slot_headers).status_code == 200
    assert slot_client.post("/s/3", content=slot_body, headers=slot_headers).status_code == 200
    assert slot_client.post("/s/3", content=slot_body, headers=slot_headers).status_code == 200
    assert [call[1:] for call in spawned] == [
        ("evt_test_abc", None),
        ("s2:evt_test_abc", 2),
        ("s3:evt_test_abc", 3),
    ]


def test_slot_worker_submits_with_that_slots_api_key(monkeypatch) -> None:
    monkeypatch.setenv("EM_API_KEY", "legacy-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("EM_API_KEY_S5", "key-s5")
    monkeypatch.setenv("EM_WEBHOOK_SECRET_S5", SLOT_SECRET)
    fake = FakeDict()
    monkeypatch.setattr(modal_app, "seen_webhooks", fake)

    used_keys: list[str] = []
    import explaining_markets.client as client_mod

    monkeypatch.setattr(
        client_mod,
        "submit_predictions",
        lambda *, event_id, predictions, config: used_keys.append(config.api_key),
    )
    worker = modal_app.predict_and_submit.get_raw_f()
    fake.put("s5:evt_test_abc", "in_flight")
    worker(_test_event(), "s5:evt_test_abc", 5)
    worker(_test_event("evt_legacy"), "evt_legacy")
    assert used_keys == ["key-s5", "legacy-key"]
    assert fake.data["s5:evt_test_abc"] == "done"


# --- Always submit (neutral fallback) ------------------------------------------

def _live_event(webhook_id: str = "evt_live") -> dict:
    return {
        "id": webhook_id,
        "event_id": "live_1",
        "event_type": "EARNINGS_RELEASE",
        "focal_assets": [
            {"identifier_type": "TICKER", "identifier_value": "AAA"},
            {"identifier_type": "TICKER", "identifier_value": "BBB"},
        ],
    }


def _capture_submissions(monkeypatch) -> list[dict]:
    monkeypatch.setenv("EM_API_KEY", "test-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(modal_app, "seen_webhooks", FakeDict())
    submitted: list[dict] = []
    import explaining_markets.client as client_mod

    monkeypatch.setattr(
        client_mod,
        "submit_predictions",
        lambda *, event_id, predictions, config: submitted.append(
            {"event_id": event_id, "predictions": predictions}
        ),
    )
    return submitted


def test_model_failure_submits_neutral_and_marks_done(monkeypatch) -> None:
    submitted = _capture_submissions(monkeypatch)
    import predict as predict_mod

    def boom(event):
        raise RuntimeError("artifact failed to load")

    monkeypatch.setattr(predict_mod, "predict", boom)
    modal_app.seen_webhooks.put("evt_live", "in_flight")
    modal_app.predict_and_submit.get_raw_f()(_live_event(), "evt_live")

    assert submitted == [
        {
            "event_id": "live_1",
            "predictions": [
                {"identifier_value": "AAA", "predicted_percentile": 0.5},
                {"identifier_value": "BBB", "predicted_percentile": 0.5},
            ],
        }
    ]
    assert modal_app.seen_webhooks.data["evt_live"] == "done"


def test_model_success_submits_model_output_unchanged(monkeypatch) -> None:
    submitted = _capture_submissions(monkeypatch)
    import predict as predict_mod

    out = [{"identifier_value": "AAA", "predicted_percentile": 0.81}]
    monkeypatch.setattr(predict_mod, "predict", lambda event: out)
    modal_app.predict_and_submit.get_raw_f()(_live_event(), "evt_live")
    assert submitted == [{"event_id": "live_1", "predictions": out}]


def test_worker_retries_a_transient_submit_failure(monkeypatch) -> None:
    monkeypatch.setenv("EM_API_KEY", "test-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(modal_app, "seen_webhooks", FakeDict())
    import httpx

    import explaining_markets.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    attempts: list[int] = []

    def flaky(*, event_id, predictions, config):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("slow")
        return {}

    monkeypatch.setattr(client_mod, "submit_predictions", flaky)
    modal_app.seen_webhooks.put("evt_test_abc", "in_flight")
    modal_app.predict_and_submit.get_raw_f()(_test_event(), "evt_test_abc")
    assert len(attempts) == 2
    assert modal_app.seen_webhooks.data["evt_test_abc"] == "done"
