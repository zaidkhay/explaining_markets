"""Train/evaluate fls_ridge_v1 on the three sealed competition quarters."""

from __future__ import annotations

import argparse
import json

from explaining_markets.fls_training import DEFAULT_ARTIFACT, train_and_serialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="Historical archive directory (default: data/historical)")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    args = parser.parse_args()
    artifact = train_and_serialize(args.source, args.artifact)
    print(json.dumps({
        "model_version": artifact["model_version"],
        "selected_alpha": artifact["selected_alpha"],
        "training_quarters": artifact["training_quarters"],
        "validation": artifact["training_metadata"]["validation_comparison"],
        "locked_holdout": artifact["training_metadata"]["locked_holdout_comparison"],
        "artifact": args.artifact,
    }, indent=2))


if __name__ == "__main__":
    main()
