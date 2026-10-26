# Reproduction

Run commands from the repository root. Requirements are Python 3.12, `uv`, and a local directory with enough space for data, feature matrices and run outputs. The root environment runs infrastructure; the nested engine environment uses its own pinned numerical dependencies. `make setup` installs both from their lockfiles.

## Saved results, figures and PDF

```bash
make setup
uv run --frozen python tools/verify_saved_runs.py
uv run --frozen python tools/rebuild_submission.py --build
```

The checked-in development archive is restored into `analyses/` and verified against its original manifest. The renderer reads `evidence/submission/selection.json`, `study.json`, `final.json` and `freeze.json`. It regenerates figures, tables and `paper/submission/main.tex`; the compiler writes `reports/submission/main.pdf`. No data download, training or inference occurs. For figures and TeX sources only, omit `--build`.

PDF compilation requires `latexmk`, `pdflatex`, `graphicx` and `booktabs`. The scientific article compiles independently of the generic workflow's draft/release status system; this does not change its release gates. Edit `paper/templates/article.tex` for narrative changes, then rebuild. Use `--author "Full Name"` to change author metadata.

The final archive contains fitted models, queries, predictions, per-query metrics and a file inventory. It is a selected export; large feature matrices are deliberately absent. `tools/verify_saved_runs.py --extract-final build/saved-final` checks and safely extracts it. Models use joblib and are intended for the pinned engine environment.

## Data acquisition

```bash
make data
```

This downloads the exact `flat/50m/multi_event.parquet` from `yandex/yambda` at revision `dd6f3a19eef5866e346c3270e098baa641a44948`. File size and SHA256 are verified before use. To reuse existing bytes:

```bash
make data DATA="/absolute/path/to/multi_event.parquet"
```

The verified path is stored in local `data/source.json`; raw data are not tracked.

## Full experiment

```bash
uv run --frozen python tools/reproduce_submission.py --run-id R-finish-reproduction-001
```

Use a fresh `R-finish-*` ID. Optional arguments are `--data PATH`, `--planning-hours 1.5`, `--no-pdf` and `--author "Full Name"`. The default planning budget is a resource ceiling, not an expected runtime. The engine may refuse a batch if calibration exceeds that budget. Keep failed evidence and use a fresh configuration rather than changing old manifests.

The command prepares a development batch, fits fixed baselines, evaluates all six one-group additions, exports validation slices and intervals, analyzes the saved predictions, freezes fitted finalists, evaluates the reserved temporal test and generates a new paper. Exact paths are recorded in `build/reproductions/RUN_ID/progress.json`. The original published results remain unchanged.

The cohort retains the initial 100 users plus 900 deterministically sampled users, with complete released histories. Query caps are 30,000 training and 20,000 development/test; they do not truncate histories. Training ends at 20,800,000, development at 23,400,000 and the reserved test at 26,000,000. The development midpoint is 22,100,000. P20 earlier validation selects features; U20 applies the same selected feature list. All models retain training-only fits. Histories include only events strictly before each prediction, including earlier test events when scoring later test queries. Equal-time events and the target are excluded.

Test evaluation is separate from the development adapter. `tools/evaluate_frozen_final.py` freezes model/source hashes, feature lists, sampling seed/cap, interval and comparisons before inference. It refuses a repeated start in the same output directory. Reproduction must not be used to choose models after inspecting the holdout.

## Checks

```bash
uv run --frozen --extra dev pytest tests -q
engines/tlab_warmup/.venv/bin/python -m unittest discover -s tests/reproduction -p test_final_holdout.py -v
uv run --frozen researchflow smoke --no-tex
```

Run native engine tests from `engines/tlab_warmup/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

For real TeX integration and engine-adapter tests, use:

```bash
RESEARCHFLOW_REAL_TEX=1 M19_WARMUP_PROJECT=engines/tlab_warmup uv run --frozen --extra dev pytest tests -q
```

`make cold-start` performs the smaller anchored baseline workflow from data through PDF. It is separate from the automatic-feature study. `make help` and [COMMANDS.md](COMMANDS.md) document the general infrastructure. Runner and analyzer extension contracts are in `interfaces/`.
