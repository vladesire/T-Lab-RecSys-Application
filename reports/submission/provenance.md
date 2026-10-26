# Selected evidence and provenance

- Development run: `R-finish-search-isolated-001`; parent run manifest SHA256 `0f192967e1357d509ee3b3450021e8c74d9ab3ad6fd32c9165e4b5f97f8d7c60`.
- Native subset export SHA256: `eaf4f350c46da072074f3e6f73d091f65e381dd3b060efc9e01c7333b2887383`. [Exact JSON](../../evidence/submission/study.json).
- Frozen holdout result SHA256: `599893d2ae9e0437415cb9b095721a5c424a391111f81ebd989d6ae0b2455594`. [Exact JSON](../../evidence/submission/final.json).
- Final freeze SHA256: `96c3d24f322623ce63e68ba785abc99fbe09b7924098ad9801836279498b51a1`. Models/features were frozen before scoring.
- Selected complete generic analysis: `A-finish-submission-search-001-4f48b8bc01546588c19d1c54`; manifest SHA256 `ffe1f7a878c69fdf3cb40c72036a66f5d9834089faf255cea5dae4827d80b8f2`.
- Renderer SHA256: `813340f0a07357f94f847a675d54192dcacb4cacd8582d386c9ff2e43bdcee93`. [Generation code](../../tools/render_submission.py).
- Standalone final evaluator: [code](../../tools/evaluate_frozen_final.py); six boundary tests in
  [test_final_holdout.py](../../tests/reproduction/test_final_holdout.py).

The native export independently recomputes saved per-query metrics and exact subset means; paired check intervals
use the native user-cluster bootstrap. The final evaluator loads frozen models, generates identical candidates
within protocol, computes strictly earlier history features, checks original columns by direct masks and scores
the reserved temporal interval once. It is a separate scientific stage; M19 remains development-only.

The generic analysis supports whole-development fixed comparisons and includes selection rows.
Its selected-search whole-development scores are not the headline out-of-selection result.
Full analysis per-query records and final models/predictions are included under `runs/published/`; raw data and large
feature matrices stay out of Git. Compact JSON exports are not relabelled as complete native runs.

