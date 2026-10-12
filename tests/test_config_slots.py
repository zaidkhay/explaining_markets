"""Per-slot credentials for the additional submissions served by one deployment."""

from __future__ import annotations

import pytest

from explaining_markets.config import EXTRA_SLOTS, Config, configured_slots


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setenv("EM_API_KEY", "legacy-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", "whsec_legacy")
    for slot in EXTRA_SLOTS:
        monkeypatch.delenv(f"EM_API_KEY_S{slot}", raising=False)
        monkeypatch.delenv(f"EM_WEBHOOK_SECRET_S{slot}", raising=False)


def test_default_slot_is_the_original_submission() -> None:
    cfg = Config.from_env()
    assert (cfg.api_key, cfg.webhook_secret) == ("legacy-key", "whsec_legacy")


def test_slot_reads_its_own_suffixed_vars(monkeypatch) -> None:
    monkeypatch.setenv("EM_API_KEY_S3", "key-s3")
    monkeypatch.setenv("EM_WEBHOOK_SECRET_S3", "whsec_s3")
    cfg = Config.from_env(3)
    assert (cfg.api_key, cfg.webhook_secret) == ("key-s3", "whsec_s3")


def test_missing_slot_credentials_raise_naming_the_var() -> None:
    with pytest.raises(RuntimeError, match="EM_API_KEY_S2"):
        Config.from_env(2)


@pytest.mark.parametrize("slot", [0, 1, 6])
def test_unknown_slot_is_rejected(slot) -> None:
    with pytest.raises(ValueError):
        Config.from_env(slot)


def test_configured_slots_requires_both_credentials(monkeypatch) -> None:
    assert configured_slots() == []
    monkeypatch.setenv("EM_API_KEY_S2", "k")
    assert configured_slots() == []
    monkeypatch.setenv("EM_WEBHOOK_SECRET_S2", "whsec_x")
    monkeypatch.setenv("EM_API_KEY_S5", "k")
    monkeypatch.setenv("EM_WEBHOOK_SECRET_S5", "whsec_y")
    assert configured_slots() == [2, 5]


def test_empty_values_count_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("EM_API_KEY_S4", "")
    monkeypatch.setenv("EM_WEBHOOK_SECRET_S4", "")
    assert configured_slots() == []
