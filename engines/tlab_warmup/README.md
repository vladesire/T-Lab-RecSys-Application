# Recommendation engine

Temporal user/item/pair features, candidate sampling, gradient boosting and bounded group selection for Yambda.

Use the repository-level [reproduction guide](../../docs/reproduction.md) for the evaluated study.
The engine uses its own `uv.lock` and environment, installed by `make setup` from the repository root.

- `src/tlab_warmup/representations.py`: feature dictionary and group definitions.
- `src/tlab_warmup/sampling.py`: candidate protocols and target eligibility.
- `src/tlab_warmup/experiment_jobs.py`: model fitting, selection and scoring.
- `configs/experiments.json`: default batch settings.
- `provenance/iteration0/results.tar.gz`: required initial reference results.

The workflow adapter supports development batches. Frozen temporal-test evaluation is implemented separately in `tools/evaluate_frozen_final.py`.
