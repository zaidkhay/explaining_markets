# Contributing

## Scope

Changes should preserve the repository's point-in-time and leakage-safety guarantees.

## Before submitting

- Keep live inference and retraining changes clearly separated.
- Do not add future-looking data to model inputs.
- Keep credentials and provider secrets out of source control.
- Run the existing test suite before merging behavioral changes.
- Update documentation when the production path, model artifact, or deployment process changes.

Use focused commits with clear messages so production changes remain auditable.