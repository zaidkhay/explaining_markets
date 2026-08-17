"""Correlation-safe production calibration and explanation helpers for V1.

The ranking model remains ``fls_ridge_v1``.  Production calibration is a
predeclared affine transform derived from the model's historical out-of-sample
validation dispersion.  An affine transform preserves Pearson and Spearman
exactly until clipping, unlike the more aggressive empirical-CDF research
calibrator.  The configured target standard deviation is intentionally below
the theoretical standard deviation of a uniform percentile target.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from explaining_markets.explanation_packet import ExplanationPacket, build_explanation_packet, log_explanation

DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "fls_ridge_v1_production_calibration.json"
CALIBRATION_METHOD = "affine_oos_scale_v1"
CALIBRATION_VERSION = "production_calibration_v1"


@dataclass(frozen=True)
class AffineProductionCalibrator:
    model_version: str
    center_raw: float
    center_target: float
    scale: float
    bounds: tuple[float, float]
    source: str
    validation_raw_std: float
    target_output_std: float
    method: str = CALIBRATION_METHOD
    version: str = CALIBRATION_VERSION

    def __post_init__(self) -> None:
        numbers = (
            self.center_raw,
            self.center_target,
            self.scale,
            self.validation_raw_std,
            self.target_output_std,
            *self.bounds,
        )
        if not all(math.isfinite(float(value)) for value in numbers):
            raise ValueError("production calibration contains non-finite values")
        low, high = self.bounds
        if not (0.0 <= low < high <= 1.0):
            raise ValueError("invalid production calibration bounds")
        if self.scale <= 0.0:
            raise ValueError("production calibration scale must be positive")
        if self.validation_raw_std <= 0.0 or self.target_output_std <= 0.0:
            raise ValueError("production calibration standard deviations must be positive")
        if not self.source.strip():
            raise ValueError("production calibration requires provenance")

    def calibrate(self, score: float) -> float:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("cannot calibrate non-finite score")
        transformed = self.center_target + self.scale * (value - self.center_raw)
        low, high = self.bounds
        return float(max(low, min(high, transformed)))

    def as_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "method": self.method,
            "version": self.version,
            "center_raw": self.center_raw,
            "center_target": self.center_target,
            "scale": self.scale,
            "bounds": list(self.bounds),
            "source": self.source,
            "validation_raw_std": self.validation_raw_std,
            "target_output_std": self.target_output_std,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AffineProductionCalibrator":
        if str(payload.get("method")) != CALIBRATION_METHOD:
            raise ValueError(f"unsupported production calibration method: {payload.get('method')}")
        bounds = payload.get("bounds") or [0.05, 0.95]
        return cls(
            model_version=str(payload["model_version"]),
            center_raw=float(payload["center_raw"]),
            center_target=float(payload["center_target"]),
            scale=float(payload["scale"]),
            bounds=(float(bounds[0]), float(bounds[1])),
            source=str(payload["source"]),
            validation_raw_std=float(payload["validation_raw_std"]),
            target_output_std=float(payload["target_output_std"]),
            version=str(payload.get("version") or CALIBRATION_VERSION),
        )


def load_production_calibrator(
    path: str | Path | None = None,
    *,
    expected_model_version: str = "fls_ridge_v1",
) -> AffineProductionCalibrator | None:
    """Load the production calibrator; fail closed to raw V1 if unavailable."""
    enabled = os.getenv("PRODUCTION_CALIBRATION_ENABLED", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    source = Path(path) if path is not None else DEFAULT_ARTIFACT_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not bool(payload.get("enabled", True)):
            return None
        calibrator = AffineProductionCalibrator.from_dict(payload)
        if calibrator.model_version != expected_model_version:
            raise ValueError(
                f"calibration model mismatch: {calibrator.model_version} != {expected_model_version}"
            )
        return calibrator
    except FileNotFoundError:
        print(f"[PROD_CALIBRATION] artifact missing path={source}; using raw V1")
    except Exception as exc:
        print(f"[PROD_CALIBRATION] invalid artifact error={type(exc).__name__}; using raw V1")
    return None


def build_v1_explanation(
    *,
    model,
    features,
    ticker: str,
    raw_prediction: float,
    calibrator: AffineProductionCalibrator | None,
    disclosure_available: bool,
) -> ExplanationPacket:
    values = features.vector(model.feature_names)
    packet = build_explanation_packet(
        ticker=ticker,
        model_version=model.model_version,
        raw_prediction=float(raw_prediction),
        feature_names=model.feature_names,
        feature_values=values,
        coefficients=model.coefficients,
        means=model.means,
        stds=model.standard_deviations,
        intercept=model.intercept,
        availability={
            "disclosure": 1.0 if disclosure_available else 0.0,
            "fls": 1.0,
        },
        fallback_status="none" if calibrator is not None else "calibration_unavailable",
        audit_status="DISCLOSURE_ONLY_POINT_IN_TIME",
        calibrator=calibrator,
        bounds=(calibrator.bounds if calibrator is not None else (model.clip_lower, model.clip_upper)),
    )
    log_explanation(packet)
    return packet


def persist_production_explanation(
    *,
    packet: ExplanationPacket,
    event_id: str,
    cutoff: datetime,
    information_url_fetch_success: bool,
    directory: str | Path | None = None,
) -> Path:
    """Write-once immutable production explanation record."""
    configured = os.getenv("V3_EVIDENCE_DIR")
    root = Path(directory) if directory is not None else Path(configured) if configured else Path("data") / "evidence"
    root = root / "production"
    root.mkdir(parents=True, exist_ok=True)
    safe_event = "".join(c for c in str(event_id) if c.isalnum() or c in "-_") or "unknown"
    safe_ticker = "".join(c for c in packet.ticker.upper() if c.isalnum() or c in ".-_") or "UNKNOWN"
    path = root / f"{safe_event}__{safe_ticker}.json"
    payload = {
        "event_id": str(event_id),
        "cutoff": cutoff.astimezone(timezone.utc).isoformat(),
        "information_url_fetch_success": bool(information_url_fetch_success),
        "submitted_percentile": (
            float(packet.calibrated_percentile)
            if packet.calibrated_percentile is not None
            else float(packet.raw_prediction)
        ),
        "explanation": packet.as_dict(),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    except FileExistsError:
        pass
    return path


def production_scenario_report() -> dict:
    """Non-submitting deterministic production smoke test."""
    from explaining_markets.model import ForwardLookingRidgeModel

    model = ForwardLookingRidgeModel()
    calibrator = load_production_calibrator(expected_model_version=model.model_version)
    scenarios = {
        "negative": [
            "We expect earnings to decline 25% next year.",
            "We lowered full-year guidance and expect revenue to decrease 20%.",
        ],
        "neutral": ["Quarterly results were in line with expectations."],
        "positive": [
            "We expect earnings to increase 25% next year.",
            "We raised full-year guidance and expect revenue growth of 20%.",
        ],
    }
    outputs = {}
    for label, disclosure in scenarios.items():
        raw, features = model.predict_with_features(disclosure)
        final = calibrator.calibrate(raw) if calibrator is not None else raw
        packet = build_v1_explanation(
            model=model,
            features=features,
            ticker=f"SCENARIO_{label.upper()}",
            raw_prediction=raw,
            calibrator=calibrator,
            disclosure_available=True,
        )
        outputs[label] = {
            "raw": float(raw),
            "final": float(final),
            "top_drivers": [d.name for d in packet.key_drivers[:3]],
        }
    final_values = [outputs[name]["final"] for name in ("negative", "neutral", "positive")]
    spread = max(final_values) - min(final_values)
    return {
        "model_version": model.model_version,
        "calibration_loaded": calibrator is not None,
        "calibration_method": calibrator.method if calibrator is not None else None,
        "calibration_version": calibrator.version if calibrator is not None else None,
        "scenarios": outputs,
        "ordered": outputs["negative"]["final"] < outputs["neutral"]["final"] < outputs["positive"]["final"],
        "spread": float(spread),
        "meaningfully_differentiated": spread >= 0.10,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
