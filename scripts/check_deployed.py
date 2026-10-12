"""Run check_production against the DEPLOYED app, not a local ephemeral copy.

    uv run python scripts/check_deployed.py

`modal run modal_app.py::check_production` builds a temporary app from the
local checkout (note its "-dev" URL), so it validates local files, which may
differ from what is deployed. This calls the deployed function by name, so
it validates exactly what production is running (CLAUDE.md §8.3).

Prints the full diagnostic. Exit code 0 on PASS, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys

import modal

APP_NAME = "explaining-markets-starter"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default=APP_NAME)
    args = parser.parse_args()

    result = modal.Function.from_name(args.app, "check_production").remote()
    print("[DEPLOYED_CHECK] " + " ".join(f"{key}={value}" for key, value in result.items()))
    passed = result.get("status") == "PASS"
    print(f"\ndeployed app {args.app!r}: {'PASS' if passed else 'FAIL'} "
          f"model={result.get('model')} submission={result.get('submission')} "
          f"promoted={result.get('promoted')}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
