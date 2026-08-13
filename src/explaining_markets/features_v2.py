"""Versioned V2 feature specification: FLS + company history + availability.

``MODEL_FEATURE_NAMES_V2`` is frozen and ordered; training, validation,
holdout, artifact serialization, and live inference all use exactly this
tuple. V1's ``MODEL_FEATURE_NAMES`` is embedded UNCHANGED as the prefix — V1
is never silently mutated. Artifact loading fails loudly on any mismatch
(see ``model.CompanyHistoryRidgeModel._validate``).

Imputation policy (documented once, applied identically train/live):
a ``None`` company-history value becomes 0.0 in the numeric vector, and its
family's availability indicator / count feature records the missingness.
0.0 is neutral under standardization fitted on the same policy; missing is
never conflated with a real economic zero because the indicator
distinguishes them.
"""

from __future__ import annotations

from dataclasses import dataclass

from explaining_markets.company_history import (
    COMPANY_HISTORY_FEATURE_NAMES,
    CompanyHistoryFeatures,
)
from explaining_markets.competition_history import (
    COMPETITION_FEATURE_NAMES,
    competition_feature_values,
)
from explaining_markets.forward_looking_features import (
    MODEL_FEATURE_NAMES,
    ForwardLookingFeatures,
)

FEATURE_SPEC_VERSION = "v2"

MODEL_FEATURE_NAMES_V2: tuple[str, ...] = (
    # existing FLS features — V1 order, byte-for-byte
    *MODEL_FEATURE_NAMES,
    # company price history + earnings reaction history + surprise history
    # + current-vs-history + similar events + recency + availability
    *COMPANY_HISTORY_FEATURE_NAMES,
    # competition-archive aggregates
    *COMPETITION_FEATURE_NAMES,
)

# Fields that must never appear as feature names (same discipline as
# features.py / feature_store.py / experiment.py).
FORBIDDEN_FEATURE_NAMES = frozenset(
    {"car1", "earnings_surprise", "surprise", "event_returns", "baseline_predictions",
     "y", "predicted_percentile", "realized_percentile"}
)

_LEAKED = FORBIDDEN_FEATURE_NAMES.intersection(MODEL_FEATURE_NAMES_V2)
if _LEAKED:  # pragma: no cover - structural guard, would fail at import time
    raise RuntimeError(f"forbidden field(s) in MODEL_FEATURE_NAMES_V2: {sorted(_LEAKED)}")


@dataclass(frozen=True)
class FeatureVectorV2:
    """One assembled V2 observation: ordered values + raw blocks for logging."""

    values: dict[str, float]
    fls: ForwardLookingFeatures
    history: CompanyHistoryFeatures

    def vector(self, names: tuple[str, ...] = MODEL_FEATURE_NAMES_V2) -> list[float]:
        return [float(self.values[name]) for name in names]


def build_feature_vector_v2(
    *,
    fls: ForwardLookingFeatures,
    history: CompanyHistoryFeatures,
) -> FeatureVectorV2:
    """Combine FLS + company-history blocks into the frozen V2 order.

    ``history.source_events`` also feeds the competition-archive aggregates
    (in this repository the archive is the only populated history source, so
    the competition block is derived from the same eligible prior events).
    """
    values: dict[str, float] = {}
    for name in MODEL_FEATURE_NAMES:
        values[name] = float(fls.values[name])

    history_values = history.as_dict()
    for name in COMPANY_HISTORY_FEATURE_NAMES:
        raw = history_values[name]
        values[name] = 0.0 if raw is None else float(raw)

    competition = competition_feature_values(list(history.source_events))
    for name in COMPETITION_FEATURE_NAMES:
        raw = competition[name]
        values[name] = 0.0 if raw is None else float(raw)

    assert set(values) == set(MODEL_FEATURE_NAMES_V2)
    return FeatureVectorV2(values=values, fls=fls, history=history)
