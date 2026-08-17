"""Persistent record of symbols a provider will never serve.

Twelve Data's free/basic universe does not include every ticker in the
competition archive (ADRs, some OTC lines, delisted shells, index proxies).
Without a durable memory of those rejections, every enrichment run re-requests
them and burns the daily API budget on deterministic failures.

This cache is intentionally *not* a failure cache: only classified permanent
rejections are stored (see ``retry_policy.classify_provider_message``).
Transient timeouts never land here.

Layout
------
One JSON document per provider under the enrichment cache root::

    data/enrichment/v3/twelve_data_unsupported.json

with structured metadata per symbol rather than a bare list, so a human can
audit *why* a symbol was skipped and when it was last attempted::

    {
      "provider": "twelve_data",
      "version": 1,
      "symbols": {
        "XYZ": {
          "ticker": "XYZ",
          "provider": "twelve_data",
          "first_seen_at": "2026-08-16T12:00:00+00:00",
          "last_seen_at":  "2026-08-16T12:00:00+00:00",
          "reason": "unsupported_symbol",
          "status_code": 400,
          "provider_message": "**symbol** not found",
          "attempt_count": 1
        }
      }
    }

Writes are atomic (temp file + ``os.replace``) so a killed run cannot leave a
truncated cache behind.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

CACHE_VERSION = 1
MAX_PROVIDER_MESSAGE_CHARS = 300

# Strip anything that looks like a credential before persisting a message.
_SECRET_PATTERNS = (
    re.compile(r"(apikey|api_key|token|key)=([^&\s]+)", re.IGNORECASE),
    re.compile(r"(Bearer)\s+([A-Za-z0-9._\-]+)", re.IGNORECASE),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_provider_message(message: str | None) -> str | None:
    """Redact credential-looking substrings and bound the stored length."""
    if message is None:
        return None
    text = str(message)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    text = text.strip()
    if len(text) > MAX_PROVIDER_MESSAGE_CHARS:
        text = text[:MAX_PROVIDER_MESSAGE_CHARS] + "..."
    return text or None


@dataclass(frozen=True)
class UnsupportedSymbolEntry:
    ticker: str
    provider: str
    first_seen_at: str
    last_seen_at: str
    reason: str
    status_code: int | None = None
    provider_message: str | None = None
    attempt_count: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


class UnsupportedSymbolCache:
    """Load/inspect/update the permanent unsupported-symbol list for a provider.

    ``retry_unsupported=True`` keeps recording rejections but reports every
    symbol as *not* skipped, which is how the CLI's ``--retry-unsupported``
    flag re-tests a previously rejected universe without discarding history.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        provider: str = "twelve_data",
        retry_unsupported: bool = False,
    ) -> None:
        self.path = Path(path)
        self.provider = provider
        self.retry_unsupported = bool(retry_unsupported)
        self._entries: dict[str, UnsupportedSymbolEntry] = {}
        self._dirty = False
        self.load()

    # ---- persistence -------------------------------------------------

    def load(self) -> None:
        """Read the cache from disk; a corrupt/absent file starts empty."""
        self._entries = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt cache must never abort a backfill run: rebuild it.
            return
        symbols = raw.get("symbols") if isinstance(raw, dict) else None
        if not isinstance(symbols, dict):
            return
        for ticker, payload in symbols.items():
            if not isinstance(payload, dict):
                continue
            key = str(ticker).upper()
            now = _utcnow().isoformat()
            self._entries[key] = UnsupportedSymbolEntry(
                ticker=key,
                provider=str(payload.get("provider") or self.provider),
                first_seen_at=str(payload.get("first_seen_at") or now),
                last_seen_at=str(payload.get("last_seen_at") or now),
                reason=str(payload.get("reason") or "unsupported_symbol"),
                status_code=(
                    int(payload["status_code"])
                    if isinstance(payload.get("status_code"), (int, float))
                    else None
                ),
                provider_message=(
                    str(payload["provider_message"])
                    if payload.get("provider_message") is not None
                    else None
                ),
                attempt_count=int(payload.get("attempt_count") or 1),
            )

    def save(self, *, force: bool = False) -> bool:
        """Atomically persist the cache. Returns True when a write happened."""
        if not (self._dirty or force):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "provider": self.provider,
            "version": CACHE_VERSION,
            "updated_at": _utcnow().isoformat(),
            "symbols": {
                ticker: entry.as_dict()
                for ticker, entry in sorted(self._entries.items())
            },
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False
        return True

    # ---- queries -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, ticker: object) -> bool:
        return str(ticker).upper() in self._entries

    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entry(self, ticker: str) -> UnsupportedSymbolEntry | None:
        return self._entries.get(str(ticker).upper())

    def should_skip(self, ticker: str) -> bool:
        """True when a request for ``ticker`` must not be attempted.

        Always False under ``retry_unsupported`` so an intentional re-test can
        reach the network again.
        """
        if self.retry_unsupported:
            return False
        return str(ticker).upper() in self._entries

    def skipped_tickers(self, tickers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        return tuple(t for t in tickers if self.should_skip(t))

    # ---- mutations ---------------------------------------------------

    def record(
        self,
        ticker: str,
        *,
        reason: str = "unsupported_symbol",
        status_code: int | None = None,
        provider_message: str | None = None,
    ) -> UnsupportedSymbolEntry:
        """Record/refresh a permanent rejection and bump its attempt count."""
        key = str(ticker).upper()
        now = _utcnow().isoformat()
        message = sanitize_provider_message(provider_message)
        existing = self._entries.get(key)
        if existing is None:
            entry = UnsupportedSymbolEntry(
                ticker=key,
                provider=self.provider,
                first_seen_at=now,
                last_seen_at=now,
                reason=reason,
                status_code=status_code,
                provider_message=message,
                attempt_count=1,
            )
        else:
            entry = UnsupportedSymbolEntry(
                ticker=key,
                provider=existing.provider,
                first_seen_at=existing.first_seen_at,
                last_seen_at=now,
                reason=reason,
                status_code=status_code if status_code is not None else existing.status_code,
                provider_message=message or existing.provider_message,
                attempt_count=existing.attempt_count + 1,
            )
        self._entries[key] = entry
        self._dirty = True
        return entry

    def clear(self, tickers: list[str] | tuple[str, ...] | None = None) -> int:
        """Forget all (or specific) unsupported symbols. Returns count removed."""
        if tickers is None:
            removed = len(self._entries)
            self._entries = {}
        else:
            removed = 0
            for ticker in tickers:
                if self._entries.pop(str(ticker).upper(), None) is not None:
                    removed += 1
        if removed:
            self._dirty = True
        return removed

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "path": str(self.path),
            "retry_unsupported": self.retry_unsupported,
            "count": len(self._entries),
            "symbols": {t: e.as_dict() for t, e in sorted(self._entries.items())},
        }


def default_unsupported_path(cache_root: str | Path, provider: str = "twelve_data") -> Path:
    """``<cache_root>/<provider>_unsupported.json``."""
    return Path(cache_root) / f"{provider}_unsupported.json"
