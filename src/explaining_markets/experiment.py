"""Experimental quantitative research layer.

Determines whether the point-in-time-safe historical features in
``feature_store.py`` add genuine out-of-sample predictive signal beyond the
existing disclosure-only heuristic (``model.HeuristicFactModel``) and the
naive earnings-surprise benchmark, using interpretable, non-LLM quantitative
models (Ridge regression, a shallow random forest).

This module is standalone offline research infrastructure, exactly like
``backtest.py`` and ``feature_store.py`` before it: it is never imported by
``predict.py``, ``modal_app.py``, or ``model.py``, and nothing here changes
production behavior. It produces the evidence documented in
``HISTORICAL_MODEL_EXPERIMENT.md``.

Chronological walk-forward discipline (STRICT — see that document's "locked
holdout" section for the full rationale):

    TRAIN       = 2025Q4   (earliest sealed quarter)
    VALIDATION  = 2026Q1   (used ONLY for model/hyperparameter selection —
                             never for the final reported holdout numbers)
    HOLDOUT     = 2026Q2   (LOCKED. Evaluated exactly once, after every
                             modeling decision has already been made using
                             TRAIN/VALIDATION only. Never iterated against.)

Realized/target fields (``car1``, ``earnings_surprise``, and anything
derived from them) never enter the feature matrix — see
:func:`assert_matrix_is_leakage_free` and the leakage-focused tests in
``tests/test_experiment.py``, including a full sweep of the real archive.

Requires the ``research`` dependency group (``numpy``, ``scikit-learn`` —
see ``pyproject.toml``). Not required by, and not installed into, the
deployed Modal image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from explaining_markets.backtest import run_backtest
from explaining_markets.model import BaselineModel
from explaining_markets.features import FORBIDDEN_KEYS, extract_features
from explaining_markets.feature_store import HistoricalFeatures, build_feature_store
from explaining_markets.historical import HistoricalEvent, labeled_events, load_historical_events
from explaining_markets.model import HeuristicFactModel

QUARTER_TRAIN = "2025Q4"
QUARTER_VALIDATION = "2026Q1"
QUARTER_HOLDOUT = "2026Q2"  # LOCKED. See module docstring.

DISCLOSURE_NUMERIC_FIELDS = (
    "n_facts",
    "text_length",
    "positive_hits",
    "negative_hits",
    "net_sentiment",
    "has_guidance_mention",
)

# The 6 historical fields that can be `None` (insufficient history) and
# therefore need median imputation + a missing-indicator column. The other
# 2 historical fields (`number_of_previous_positive_surprises`,
# `number_of_prior_earnings_events`) are always defined ints - no imputation
# needed for them.
HISTORICAL_IMPUTED_FIELDS = (
    "previous_car1",
    "rolling_mean_car1",
    "rolling_car1_volatility",
    "previous_earnings_surprise",
    "rolling_mean_surprise",
    "historical_reaction_asymmetry",
)
HISTORICAL_DIRECT_FIELDS = (
    "number_of_previous_positive_surprises",
    "number_of_prior_earnings_events",
)
HISTORICAL_ALL_FIELDS = HISTORICAL_IMPUTED_FIELDS + HISTORICAL_DIRECT_FIELDS

# Feature "families" for the ablation analysis (§ HISTORICAL_MODEL_EXPERIMENT.md).
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "disclosure": DISCLOSURE_NUMERIC_FIELDS,
    "historical_car1": (
        "previous_car1",
        "rolling_mean_car1",
        "rolling_car1_volatility",
        "historical_reaction_asymmetry",
    ),
    "historical_surprise": (
        "previous_earnings_surprise",
        "rolling_mean_surprise",
        "number_of_previous_positive_surprises",
    ),
    "historical_count": ("number_of_prior_earnings_events",),
}


# ----------------------------------------------------------------------
# Row assembly: reuses backtest.py's target definition exactly
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentRow:
    """One (event, ticker) observation, fully assembled for modeling.

    ``y`` (realized percentile) and ``surprise_percentile`` come directly
    from :func:`explaining_markets.backtest.run_backtest` — this module does
    not recompute the target, it reuses the existing backtest framework's
    definition verbatim, per the task's requirement.
    """

    event_id: str
    ticker: str
    quarter: str
    disclosure: dict  # explaining_markets.features.FeatureVector.as_dict()
    historical: dict  # explaining_markets.feature_store.HistoricalFeatures.feature_values()
    y: float
    surprise_percentile: float | None


def build_experiment_rows(events: list[HistoricalEvent]) -> list[ExperimentRow]:
    """Assemble every labeled event into an :class:`ExperimentRow`.

    Three independent construction steps, each delegated to its own,
    already-tested module rather than reimplemented here:

    1. The target ``y`` and the ``surprise_percentile`` benchmark come from
       :func:`explaining_markets.backtest.run_backtest` (called with the
       inert :class:`~explaining_markets.model.BaselineModel` purely to
       obtain its row-level target construction — the model's own
       predictions are discarded).
    2. Disclosure features come from
       :func:`explaining_markets.features.extract_features`, called on each
       event's own ``disclosure`` — exactly what ``predict.py`` and
       ``backtest.py`` already use.
    3. Historical features come from
       :func:`explaining_markets.feature_store.build_feature_store`, called
       on the FULL multi-quarter ``events`` list (so cross-quarter history
       is available, per that module's own walk-forward design) — never
       restricted to a single quarter before this point.
    """
    probe = run_backtest(labeled_events(events), BaselineModel())
    historical_by_id = {row.target_event_id: row for row in build_feature_store(events)}
    events_by_id = {event.event_id: event for event in events}

    rows: list[ExperimentRow] = []
    for backtest_row in probe.rows:
        event = events_by_id[backtest_row.event_id]
        disclosure_features = extract_features(
            ticker=event.ticker, event_type=event.event_type, disclosure=event.disclosure
        )
        historical_features = historical_by_id[backtest_row.event_id]
        rows.append(
            ExperimentRow(
                event_id=backtest_row.event_id,
                ticker=backtest_row.ticker,
                quarter=backtest_row.quarter or "UNKNOWN",
                disclosure=disclosure_features.as_dict(),
                historical=historical_features.feature_values(),
                y=backtest_row.realized_percentile,
                surprise_percentile=backtest_row.surprise_percentile,
            )
        )
    return rows


# ----------------------------------------------------------------------
# Feature matrix construction (train-only imputation, no leakage)
# ----------------------------------------------------------------------


@dataclass
class MatrixBuilder:
    """Builds a numeric feature matrix, fitting imputation on TRAIN rows only.

    ``include_disclosure`` / ``include_historical`` select which family of
    columns to emit (used to build the "historical features only",
    "disclosure + historical features", and ablation matrices from the same
    class). Median imputation values for :data:`HISTORICAL_IMPUTED_FIELDS`
    are computed once, from whichever rows are passed to :meth:`fit`, and
    reused unchanged for every subsequent :meth:`transform` call — so
    fitting on TRAIN and transforming VALIDATION/HOLDOUT never lets those
    later quarters influence the imputation values.
    """

    include_disclosure: bool = True
    include_historical: bool = True
    _medians: dict[str, float] = field(default_factory=dict)
    _fitted: bool = False

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.include_disclosure:
            names.extend(DISCLOSURE_NUMERIC_FIELDS)
        if self.include_historical:
            names.extend(HISTORICAL_ALL_FIELDS)
            names.extend(f"{name}_was_missing" for name in HISTORICAL_IMPUTED_FIELDS)
        return names

    def fit(self, rows: list[ExperimentRow]) -> "MatrixBuilder":
        for name in HISTORICAL_IMPUTED_FIELDS:
            values = [
                row.historical[name] for row in rows if row.historical.get(name) is not None
            ]
            # Fallback to 0.0 only if TRAIN itself has zero coverage for this
            # field (e.g. a single-quarter train set with no rolling
            # volatility observations yet) - never falls back using
            # validation/holdout data.
            self._medians[name] = float(np.median(values)) if values else 0.0
        self._fitted = True
        return self

    def transform(self, rows: list[ExperimentRow]) -> np.ndarray:
        """Build the numeric matrix, using ONLY imputation values from :meth:`fit`.

        Column order always matches :meth:`feature_names`: disclosure fields
        (if included), then every historical field (imputed where missing),
        then one missing-indicator column per :data:`HISTORICAL_IMPUTED_FIELDS`
        (if historical features are included).
        """
        if not self._fitted:
            raise RuntimeError("MatrixBuilder.transform() called before fit()")
        matrix: list[list[float]] = []
        for row in rows:
            values: list[float] = []
            if self.include_disclosure:
                for name in DISCLOSURE_NUMERIC_FIELDS:
                    v = row.disclosure[name]
                    values.append(float(int(v)) if isinstance(v, bool) else float(v))
            if self.include_historical:
                for name in HISTORICAL_ALL_FIELDS:
                    v = row.historical[name]
                    if v is None:
                        values.append(self._medians.get(name, 0.0))
                    else:
                        values.append(float(v))
                for name in HISTORICAL_IMPUTED_FIELDS:
                    values.append(1.0 if row.historical[name] is None else 0.0)
            matrix.append(values)
        return np.asarray(matrix, dtype=float)


def assert_matrix_is_leakage_free(feature_names: list[str]) -> None:
    """Raise if any column name in a built matrix is a forbidden, realized field.

    Belt-and-suspenders, matching the same discipline already applied in
    ``features.py``/``backtest.py``/``feature_store.py``: this should never
    trigger, since :data:`DISCLOSURE_NUMERIC_FIELDS` and
    :data:`HISTORICAL_ALL_FIELDS` are hardcoded, known-safe lists, but it
    costs nothing to check.
    """
    leaked = [name for name in feature_names if name in FORBIDDEN_KEYS]
    if leaked:
        raise ValueError(f"leaked realized field(s) into the feature matrix: {leaked}")


# ----------------------------------------------------------------------
# Quantitative models (Ridge, shallow random forest) - no LLM
# ----------------------------------------------------------------------


class RidgeQuantModel:
    """Ridge (L2-regularized linear) regression, clipped to [0, 1]."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeQuantModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self._model.predict(X), 0.0, 1.0)

    @property
    def coefficients(self) -> np.ndarray:
        return self._model.coef_


class RandomForestQuantModel:
    """A shallow random forest, kept small/interpretable on purpose.

    ``max_depth`` is deliberately small (default 4) given the limited
    training set size (a single quarter, ~1,800-2,400 rows) — a deep,
    unconstrained forest would overfit trivially. ``random_state`` is fixed
    for reproducibility.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 4, random_state: int = 0) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestQuantModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self._model.predict(X), 0.0, 1.0)

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    n_obs: int
    correlation: float | None
    mean_abs_error: float | None


def compute_metrics(predicted: np.ndarray | list[float], realized: list[float]) -> Metrics:
    predicted = list(np.asarray(predicted, dtype=float))
    n = len(predicted)
    if n == 0:
        return Metrics(n_obs=0, correlation=None, mean_abs_error=None)
    correlation = _pearson(predicted, realized)
    mae = sum(abs(p - r) for p, r in zip(predicted, realized, strict=True)) / n
    return Metrics(n_obs=n, correlation=correlation, mean_abs_error=mae)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; ``None`` if undefined (n<2 or zero variance).

    Deliberately the same formula as ``backtest.py``'s private ``_pearson``
    (not imported from there, to keep this module's public surface
    self-contained) — verified identical by shared test fixtures in
    ``tests/test_experiment.py``.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    s_xx = sum((x - mean_x) ** 2 for x in xs)
    s_yy = sum((y - mean_y) ** 2 for y in ys)
    if s_xx == 0.0 or s_yy == 0.0:
        return None
    s_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return s_xy / math.sqrt(s_xx * s_yy)


# ----------------------------------------------------------------------
# Ablation / permutation importance
# ----------------------------------------------------------------------


def permutation_ablation(
    *,
    model,
    X: np.ndarray,
    y: list[float],
    feature_names: list[str],
    families: dict[str, tuple[str, ...]],
    seed: int = 0,
) -> dict[str, float]:
    """Correlation drop when each feature family's columns are shuffled.

    For each family, the columns belonging to it are permuted (row order
    shuffled) independently, holding every other column fixed, and the
    model's (already-fitted) correlation on the resulting matrix is compared
    to its correlation on the unpermuted matrix. A larger drop means the
    family contributed more to the model's predictive correlation. A fixed
    ``seed`` makes this reproducible.

    This is a diagnostic on an ALREADY-FITTED model — it fits nothing itself
    and must never be run against data used to select that model's
    hyperparameters if the intent is an unbiased read on generalization
    (see HISTORICAL_MODEL_EXPERIMENT.md for exactly which split this is run
    on and why).
    """
    rng = np.random.RandomState(seed)
    baseline_pred = model.predict(X)
    baseline_corr = _pearson(list(baseline_pred), y) or 0.0

    drops: dict[str, float] = {}
    for family_name, family_fields in families.items():
        indices = [i for i, name in enumerate(feature_names) if name in family_fields]
        if not indices:
            continue
        permuted = X.copy()
        perm_order = rng.permutation(len(X))
        for idx in indices:
            permuted[:, idx] = permuted[perm_order, idx]
        permuted_pred = model.predict(permuted)
        permuted_corr = _pearson(list(permuted_pred), y) or 0.0
        drops[family_name] = baseline_corr - permuted_corr
    return drops


# ----------------------------------------------------------------------
# Full experiment orchestration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentReport:
    per_quarter: dict[str, dict[str, Metrics]]  # quarter -> model_name -> Metrics
    holdout: dict[str, Metrics]  # model_name -> Metrics, QUARTER_HOLDOUT only
    selected_hyperparameters: dict[str, object]
    ablation_validation: dict[str, float]
    ablation_holdout: dict[str, float]


def run_experiment(events: list[HistoricalEvent] | None = None) -> ExperimentReport:
    """Run the full, locked-holdout experiment described in the module docstring.

    Safe to call with no arguments (loads ``data/historical/`` itself); an
    explicit ``events`` list is accepted mainly for tests, so they can pass a
    small synthetic dataset instead of the real archive.
    """
    if events is None:
        events = load_historical_events()

    rows = build_experiment_rows(events)
    by_quarter: dict[str, list[ExperimentRow]] = {}
    for row in rows:
        by_quarter.setdefault(row.quarter, []).append(row)

    train_rows = by_quarter.get(QUARTER_TRAIN, [])
    validation_rows = by_quarter.get(QUARTER_VALIDATION, [])
    holdout_rows = by_quarter.get(QUARTER_HOLDOUT, [])
    train_plus_validation = train_rows + validation_rows

    # ---- Model/hyperparameter selection: TRAIN -> VALIDATION only --------
    hist_builder = MatrixBuilder(include_disclosure=False, include_historical=True).fit(train_rows)
    combo_builder = MatrixBuilder(include_disclosure=True, include_historical=True).fit(train_rows)

    X_train_hist = hist_builder.transform(train_rows)
    X_val_hist = hist_builder.transform(validation_rows)
    X_train_combo = combo_builder.transform(train_rows)
    X_val_combo = combo_builder.transform(validation_rows)
    assert_matrix_is_leakage_free(hist_builder.feature_names())
    assert_matrix_is_leakage_free(combo_builder.feature_names())

    y_train = [r.y for r in train_rows]
    y_val = [r.y for r in validation_rows]

    selected: dict[str, object] = {}

    def _select_ridge(X_train, X_val, label: str) -> RidgeQuantModel:
        # `best_model is None` (not just `corr > best_corr`) guards the first
        # candidate: if EVERY candidate's validation correlation is undefined
        # (e.g. the training matrix has near-zero variance - see the
        # "cold start" note in HISTORICAL_MODEL_EXPERIMENT.md for when this
        # happens), a strict `-inf > -inf` comparison would never trigger and
        # `best_alpha` would stay `None`, crashing the final refit.
        best_alpha, best_corr, best_model = None, float("-inf"), None
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidate = RidgeQuantModel(alpha=alpha).fit(X_train, y_train)
            corr = compute_metrics(candidate.predict(X_val), y_val).correlation
            corr = corr if corr is not None else float("-inf")
            if best_model is None or corr > best_corr:
                best_alpha, best_corr, best_model = alpha, corr, candidate
        selected[f"ridge_alpha_{label}"] = best_alpha
        return best_model

    def _select_forest(X_train, X_val, label: str) -> RandomForestQuantModel:
        best_depth, best_corr, best_model = None, float("-inf"), None
        for depth in (2, 3, 4, 6):
            candidate = RandomForestQuantModel(max_depth=depth).fit(X_train, y_train)
            corr = compute_metrics(candidate.predict(X_val), y_val).correlation
            corr = corr if corr is not None else float("-inf")
            if best_model is None or corr > best_corr:
                best_depth, best_corr, best_model = depth, corr, candidate
        selected[f"forest_max_depth_{label}"] = best_depth
        return best_model

    _select_ridge(X_train_hist, X_val_hist, "historical")
    _select_forest(X_train_hist, X_val_hist, "historical")
    _select_ridge(X_train_combo, X_val_combo, "combined")
    _select_forest(X_train_combo, X_val_combo, "combined")

    # ---- Lock hyperparameters; refit on TRAIN+VALIDATION; evaluate once --
    hist_builder_final = MatrixBuilder(include_disclosure=False, include_historical=True).fit(
        train_plus_validation
    )
    combo_builder_final = MatrixBuilder(include_disclosure=True, include_historical=True).fit(
        train_plus_validation
    )
    y_trainval = [r.y for r in train_plus_validation]

    ridge_hist = RidgeQuantModel(alpha=selected["ridge_alpha_historical"]).fit(
        hist_builder_final.transform(train_plus_validation), y_trainval
    )
    forest_hist = RandomForestQuantModel(max_depth=selected["forest_max_depth_historical"]).fit(
        hist_builder_final.transform(train_plus_validation), y_trainval
    )
    ridge_combo = RidgeQuantModel(alpha=selected["ridge_alpha_combined"]).fit(
        combo_builder_final.transform(train_plus_validation), y_trainval
    )
    forest_combo = RandomForestQuantModel(max_depth=selected["forest_max_depth_combined"]).fit(
        combo_builder_final.transform(train_plus_validation), y_trainval
    )
    heuristic = HeuristicFactModel()

    def _heuristic_predict(quarter_rows: list[ExperimentRow]) -> list[float]:
        preds = []
        for row in quarter_rows:
            # Rebuild the FeatureVector the heuristic model expects, from the
            # already-extracted disclosure dict (avoids re-fetching anything).
            from explaining_markets.features import FeatureVector

            fv = FeatureVector(
                ticker=row.ticker,
                event_type="EARNINGS_RELEASE",
                n_facts=int(row.disclosure["n_facts"]),
                text_length=int(row.disclosure["text_length"]),
                positive_hits=int(row.disclosure["positive_hits"]),
                negative_hits=int(row.disclosure["negative_hits"]),
                net_sentiment=int(row.disclosure["net_sentiment"]),
                has_guidance_mention=bool(row.disclosure["has_guidance_mention"]),
            )
            preds.append(heuristic.predict_percentile(fv))
        return preds

    def _evaluate_all(quarter_rows: list[ExperimentRow]) -> dict[str, Metrics]:
        if not quarter_rows:
            return {}
        y_true = [r.y for r in quarter_rows]
        X_hist = hist_builder_final.transform(quarter_rows)
        X_combo = combo_builder_final.transform(quarter_rows)
        surprise_rows = [(r.surprise_percentile, r.y) for r in quarter_rows if r.surprise_percentile is not None]
        return {
            "constant_0.5": compute_metrics([0.5] * len(quarter_rows), y_true),
            "disclosure_heuristic": compute_metrics(_heuristic_predict(quarter_rows), y_true),
            "historical_ridge": compute_metrics(ridge_hist.predict(X_hist), y_true),
            "historical_forest": compute_metrics(forest_hist.predict(X_hist), y_true),
            "combined_ridge": compute_metrics(ridge_combo.predict(X_combo), y_true),
            "combined_forest": compute_metrics(forest_combo.predict(X_combo), y_true),
            "surprise_benchmark": compute_metrics(
                [s for s, _ in surprise_rows], [y for _, y in surprise_rows]
            ),
        }

    per_quarter = {
        QUARTER_TRAIN: _evaluate_all(train_rows),
        QUARTER_VALIDATION: _evaluate_all(validation_rows),
        QUARTER_HOLDOUT: _evaluate_all(holdout_rows),
    }
    holdout_metrics = per_quarter[QUARTER_HOLDOUT]

    # ---- Ablation: validation (decision-safe) and holdout (reported once) -
    ablation_validation = permutation_ablation(
        model=ridge_combo,
        X=X_val_combo,
        y=y_val,
        feature_names=combo_builder.feature_names(),
        families=FEATURE_FAMILIES,
    )
    X_holdout_combo = combo_builder_final.transform(holdout_rows)
    y_holdout = [r.y for r in holdout_rows]
    ablation_holdout = permutation_ablation(
        model=ridge_combo,
        X=X_holdout_combo,
        y=y_holdout,
        feature_names=combo_builder_final.feature_names(),
        families=FEATURE_FAMILIES,
    )

    return ExperimentReport(
        per_quarter=per_quarter,
        holdout=holdout_metrics,
        selected_hyperparameters=selected,
        ablation_validation=ablation_validation,
        ablation_holdout=ablation_holdout,
    )


def _format_metrics(m: Metrics) -> str:
    corr = f"{m.correlation:.4f}" if m.correlation is not None else "None"
    mae = f"{m.mean_abs_error:.4f}" if m.mean_abs_error is not None else "None"
    return f"n={m.n_obs:5d}  r={corr:>8}  MAE={mae:>8}"


def main() -> None:  # pragma: no cover - manual/reporting entry point
    report = run_experiment()
    print("Selected hyperparameters (chosen on TRAIN->VALIDATION only):")
    for k, v in report.selected_hyperparameters.items():
        print(f"  {k}: {v}")
    print()
    for quarter, models in report.per_quarter.items():
        label = " (LOCKED HOLDOUT)" if quarter == QUARTER_HOLDOUT else ""
        print(f"=== {quarter}{label} ===")
        for name, metrics in models.items():
            print(f"  {name:22s} {_format_metrics(metrics)}")
        print()
    print("Ablation (validation, 2026Q1 - decision-safe):")
    for family, drop in report.ablation_validation.items():
        print(f"  {family:22s} correlation drop when shuffled: {drop:.4f}")
    print()
    print("Ablation (holdout, 2026Q2 - reported once, post-hoc only):")
    for family, drop in report.ablation_holdout.items():
        print(f"  {family:22s} correlation drop when shuffled: {drop:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
