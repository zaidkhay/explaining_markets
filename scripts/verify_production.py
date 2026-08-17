#!/usr/bin/env python3
"""Verify the exact production V1 + calibration + explanation path locally."""
from __future__ import annotations

import json

from explaining_markets.production_runtime import production_scenario_report


def main() -> int:
    report = production_scenario_report()
    print("=== PRODUCTION V1 VERIFICATION ===")
    print(f"model: {report['model_version']}")
    print(f"calibration_loaded: {report['calibration_loaded']}")
    print(f"calibration_method: {report['calibration_method']}")
    print(f"calibration_version: {report['calibration_version']}")
    print()
    for label in ("negative", "neutral", "positive"):
        row = report["scenarios"][label]
        print(
            f"{label:<8} raw={row['raw']:.4f} final={row['final']:.4f} "
            f"drivers={','.join(row['top_drivers'])}"
        )
    print()
    print(f"ordered: {report['ordered']}")
    print(f"spread: {report['spread']:.4f}")
    print(f"meaningfully_differentiated: {report['meaningfully_differentiated']}")
    ok = bool(
        report["calibration_loaded"]
        and report["ordered"]
        and report["meaningfully_differentiated"]
    )
    print("\n=== FINAL ===")
    print("PASS" if ok else "FAIL")
    if not ok:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
