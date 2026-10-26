# Published runs

## Development

`development/analysis.tar.xz` contains the complete analysis `A-finish-submission-search-001-4f48b8bc01546588c19d1c54`, including per-query records for 24 method/protocol series. `study.json` records earlier/later validation metrics, paired intervals, selected groups, search trials and costs. The ledger, configuration, readiness, job results and original manifests are retained alongside it.

The parent run is `R-finish-search-isolated-001`. Large input caches, feature matrices, staged environments and logs are omitted. This directory is a selected export, not the original complete run bundle. The analysis archive itself is complete and verified against its original inventory.

## Mac reproduction

`mac/analysis.tar.xz` preserves the complete author-supplied analysis `A-finish-search-001-fb4c7e0bdcdce6e35b83b6b1`, including its original manifest and per-query records. Its 288 common metric cells exactly match the Linux development analysis. Both use the same cohort; this is a reproduction, not an independent sample.

## Temporal test

`final/selected-evidence.tar.xz` contains frozen fitted models, P20/U20 queries, predictions, per-query metrics, summaries, the pre-inference freeze and `EXPORT.json`. Its original runtime manifest is retained as `source-manifest.json`. Recomputable feature matrices are excluded explicitly. The uncompressed results and freeze are also available as `final/result.json` and `final/freeze.json`.

## Verify or extract

```bash
uv run --frozen python tools/verify_saved_runs.py
uv run --frozen python tools/verify_saved_runs.py --extract-final build/saved-final
```

`INDEX.json` pins the checked-in files. Verification also checks both compressed archives internally and matches the final models to their pre-inference hashes. Raw data are unnecessary for these checks. The [reproduction guide](../../docs/reproduction.md) covers fresh computation.
