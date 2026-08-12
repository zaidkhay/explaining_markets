"""Emit differentiated real historical predictions for deployment smoke checking."""

from __future__ import annotations

import argparse
import json

from explaining_markets.forward_looking_features import extract_forward_looking_features
from explaining_markets.historical import load_historical_events
from explaining_markets.model import ForwardLookingRidgeModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/historical")
    parser.add_argument("--artifact", default="src/explaining_markets/artifacts/fls_ridge_v1.json")
    parser.add_argument("--output", default="fls_smoke_report.json")
    args = parser.parse_args()

    model = ForwardLookingRidgeModel(args.artifact)
    scored = []
    for event in load_historical_events(args.source):
        if not event.disclosure:
            continue
        prediction, f = model.predict_with_features(event.disclosure)
        v = f.values
        scored.append({
            "event_id": event.event_id,
            "ticker": event.ticker,
            "quarter": event.quarter,
            "prediction": prediction,
            "fls_ratio": v["fls_ratio"],
            "quant_earnings_fls_ratio": v["quant_earnings_fls_ratio"],
            "other_fls_ratio": v["other_fls_ratio"],
            "signed_forward_tone": v["signed_forward_tone"],
            "guidance_direction": v["guidance_direction"],
        })
    if len(scored) < 3:
        raise RuntimeError("need at least three historical disclosures for smoke report")
    scored.sort(key=lambda row: row["prediction"])
    examples = [scored[0], scored[len(scored) // 2], scored[-1]]
    if len({round(x["prediction"], 12) for x in examples}) < 2:
        raise RuntimeError("production model produced no differentiated historical predictions")
    report = {"model_version": model.model_version, "examples": examples}
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
