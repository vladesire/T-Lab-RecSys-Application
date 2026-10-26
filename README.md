# Automatic selection of temporal features for music recommendation

Vladislav Neklydov

[Paper](reports/submission/main.pdf) · [Results and figures](reports/submission/README.md) · [Reproduction](docs/reproduction.md)

We compare automatic selection of temporal feature groups with a manual gradient-boosting baseline on Yambda. The study uses complete histories for 1,000 users, chronological splits, strictly past features and identical candidates within each protocol.

Automatic selection adds eight pair-decay features to the 17-feature manual baseline. On 20,000 temporal-test queries from 772 users, P20 NDCG@10 is **0.778708**, versus **0.778141** for manual features. The paired difference is **+0.000567**, with a 95% user-cluster interval of **[-0.000550, +0.001672]**. The experiment does not establish an improvement. Full-dictionary and decay controls, search traces, repeat/novel ablations and computational costs are reported in the paper.

## Research and engineering approach

I built this project over two days to investigate whether automatic temporal feature selection could compete with manually designed features for recommendation. With a fixed deadline and a single laptop, I prioritized a complete, reproducible experiment: a substantive baseline, explicit information boundaries, a bounded search, and an honest analysis of the outcome.

I started with warmup experiments to establish the task and check the assumptions before expanding the system. These used a small user cohort with complete interaction histories, chronological splits, and features computed strictly from past events. Simple controls and a pair-feature ablation helped identify what the baseline was exploiting. They also made an important limitation visible: aggregate ranking quality can conceal very different behavior on repeat and novel interactions.

The next decision was to invest part of the remaining time in infrastructure. I wanted subsequent experiments to produce comparable evidence without repeatedly assembling scripts, checking paths, and copying numbers into a report. I decomposed the workflow into modules with explicit interfaces: configuration and provenance capture, experiment execution, ranking analysis, research records, artifact generation, and publication. Development proceeded module by module, using specification-driven, LLM-assisted implementation, isolated tests, and integration checks.

I then integrated the scientific engine into the repository and connected the full path from pinned data to experiments, saved predictions, statistical comparisons, figures, and PDF. Computation and reporting remain separate: rebuilding the paper does not retrain models, and published results refer to explicitly selected evidence rather than an implicit “latest” run. The repository includes the dependency locks, saved analyses, frozen models and predictions, and commands for both inspecting existing results and recomputing the study.

The resulting experiment selected eight additional temporal features, but the held-out comparison did not establish an improvement over the manual baseline. That limits the scientific claim. The engineering outcome is more concrete: a working foundation for testing further hypotheses while preserving their configurations, evidence, and interpretation. The initial infrastructure investment was substantial within a two-day project; whether it pays off in iteration speed requires longer use and measurement.

This project reflects the role I want to play in a research team: understanding the scientific question, implementing the experiment carefully, and building tools that make the next experiment easier to run, inspect, and reproduce.

## Rebuild the figures and paper

Requirements: Python 3.12, `uv`, and `latexmk`/`pdflatex` with `graphicx` and `booktabs` for PDF compilation.

```bash
make setup
uv run --frozen python tools/rebuild_submission.py --build
```

All saved evidence needed for this command is included. It restores and verifies the development analysis, regenerates figures/tables and compiles the scientific article. No raw data or model fitting is required. Omit `--build` to regenerate figures and sources without TeX.

## Recompute the experiments

```bash
make setup
make data
uv run --frozen python tools/reproduce_submission.py --run-id R-finish-reproduction-001
```

`make data` downloads and verifies the pinned 384.5 MB Yambda file. Use a new run ID for each execution. The study fits fixed baselines, selects features on earlier validation, evaluates later validation, freezes models and evaluates the reserved temporal test. Results are written to `build/reproductions/R-finish-reproduction-001/`. The [reproduction guide](docs/reproduction.md) describes data reuse, resource limits and tests.

## Repository layout

| Path | Contents |
| --- | --- |
| `engines/tlab_warmup/` | Scientific engine, numerical dependency lock and tests |
| `src/researchflow/` | Configuration, execution, analysis, provenance and publication infrastructure |
| `tools/` | Dataset acquisition, study orchestration, evidence checks and paper generation |
| `runs/published/` | Development records, complete compressed analysis, frozen test models and predictions |
| `evidence/` | Selected numerical results, model/source identities and baseline/Mac reproduction evidence |
| `research/` | Scientific hypotheses, design and decisions |
| `paper/` | Scientific article template, generated sources and figures; small workflow example |
| `reports/submission/` | Paper and readable results |
| `tests/`, `interfaces/` | Infrastructure tests, schemas and extension contracts |

Raw data, feature matrices, environments and temporary workspaces are excluded from Git. Published run exports state their contents and omissions explicitly; the original manifests are preserved. See [saved runs](runs/published/README.md) and [data and artifacts](docs/data-and-artifacts.md).

Selection uses P20 earlier validation only. Models retain their training-only fits. Earlier recorded test events may enter features for later test predictions under the shared prequential policy. This is sampled positive-target ranking with 20 alternatives, not click prediction or full-catalog evaluation. The one-round search and matched control enumerate the same six additions, so their agreement does not establish optimizer superiority.
