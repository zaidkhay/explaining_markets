"""Replaceable percentile-prediction model interface.

:class:`PercentileModel` is the contract ``predict.py`` and ``backtest.py``
both code against. :class:`BaselineModel` is the deterministic,
always-available fallback (no data, no training, no external dependencies).
:class:`HeuristicFactModel` is the MVP's actual strategy — a small, fully
transparent, rule-based mapping from a
:class:`~explaining_markets.features.FeatureVector` to a percentile,
documented well enough that a human can hand-verify any prediction it makes.

Swapping in a trained model later means writing a new class that implements
``predict_percentile`` (and optionally ``fit``) — nothing else in the
pipeline (``predict.py``, ``backtest.py``) needs to change; only
:func:`get_default_model` needs to point at the new class.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from explaining_markets.features import FeatureVector

# Calibration discipline mirrored from the starter's original LLM-based
# strategy: reserve the extremes for strong, unambiguous signal, and default
# toward the middle otherwise. Neither bound is reachable by construction —
# see HeuristicFactModel's docstring for the exact formula.
_MIN_PERCENTILE = 0.10
_MAX_PERCENTILE = 0.90
_SENTIMENT_SCALE = 4.0  # net_sentiment hits needed to approach the bounds


@runtime_checkable
class PercentileModel(Protocol):
    """Contract every prediction strategy must satisfy."""

    def predict_percentile(self, features: FeatureVector) -> float:
        """Return a predicted CAR1 percentile in ``[0, 1]`` for one ``(event, ticker)``."""
        ...

    def fit(self, training_rows: list[tuple[FeatureVector, float]]) -> None:
        """Optionally fit on historical ``(features, realized_percentile)`` pairs.

        The MVP models below no-op here (they are not trained); a future
        model can override this without changing any caller.
        """
        ...


class BaselineModel:
    """Deterministic 0.5 baseline. Always available; no inputs required.

    This is the fallback ``predict.py`` and ``get_default_model`` use
    whenever nothing better is available — matching the starter's original
    "round-trip works without burning credits" behavior, but as a first-class,
    independently testable component.
    """

    def predict_percentile(self, features: FeatureVector) -> float:  # noqa: ARG002
        return 0.5

    def fit(self, training_rows: list[tuple[FeatureVector, float]]) -> None:  # noqa: ARG002
        return None


class HeuristicFactModel:
    """MVP strategy: a transparent, rule-based mapping from fact sentiment to percentile.

    Formula (fully auditable, no hidden state, no training required)::

        score      = net_sentiment / _SENTIMENT_SCALE
        percentile = 0.5 + 0.5 * tanh(score)      # squashes to (0, 1), centered at 0.5
        percentile = clip(percentile, _MIN_PERCENTILE, _MAX_PERCENTILE)

    With no keyword hits (``net_sentiment == 0``) this returns exactly
    ``0.5`` — the same neutral value as :class:`BaselineModel` — so a
    fact-free or neutral disclosure never produces a confident prediction.
    ``tanh`` guarantees the output is smoothly bounded before the explicit
    clip is even applied; the clip exists as a second, independent guarantee.
    """

    def predict_percentile(self, features: FeatureVector) -> float:
        score = features.net_sentiment / _SENTIMENT_SCALE
        raw = 0.5 + 0.5 * math.tanh(score)
        return max(_MIN_PERCENTILE, min(_MAX_PERCENTILE, raw))

    def fit(self, training_rows: list[tuple[FeatureVector, float]]) -> None:  # noqa: ARG002
        # Rule-based, nothing to fit. Kept for interface parity with a future
        # trained model — see the module docstring.
        return None


def get_default_model() -> PercentileModel:
    """The model ``predict.py`` uses today. Change this to change the live strategy.

    Always returns an instance that requires no historical data and cannot
    raise on construction, so callers never need a fallback path just to
    obtain a model — see ``predict.py`` for the additional, independent
    runtime safety net around calling it.
    """
    return HeuristicFactModel()
