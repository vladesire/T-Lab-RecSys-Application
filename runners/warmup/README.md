# Warmup runner bridge (M19)

This adapter calls the **existing** `python -m tlab_warmup.batch iteration1|overnight` CLI from the supplied warmup project. It does not implement models, sampling, temporal features, search, or final-test scoring. Its local runtime needs NumPy; the explicitly selected upstream Python needs the project's installed dependencies. There is no automatic installation, download, data acquisition, or fallback environment.

## Mapping and supported choices

The request is the common `runner_request` envelope. All input paths must be absolute; the infrastructure resolves relative paths against its project root before invocation. The scientific protocol must be exactly:

```json
{"id":"warmup-development-v1","candidates":["U20","P20"],"history":"strict-past-complete","train_end_timestamp":20800000,"development_end_timestamp":23400000,"timestamp_unit":"seconds","final_test":"not-scored"}
```

Synthetic anchors use their actual resolved boundaries. Only `phase=fixture|development` is supported. Final phase is rejected even if an outer gate approved it: this upstream project has no final-scoring implementation.

`scientific.parameters` requires exactly these fields:

| Field | Mapping / restriction |
|---|---|
| `batch_phase` | Exact native subcommand `iteration1` or `overnight`. |
| `controls_only` | Native `--controls-only`, overnight only. True in iteration1 fails because it has no effect. |
| `batch_config` | Complete `configs/experiments.json` structure, with unknown keys rejected recursively. Written as `resolved-batch-config.json` and passed using `--config`. |
| `seed` | Must equal the anchor's base seed. Arbitrary seed overrides are unsupported and fail. Native random/control/tie seeds and eligible-target sampling offsets remain unchanged. |
| `anchor_config_sha256` | Canonical SHA256 of the entire archived resolved configuration. Freezes model parameters, eligibility, histories, source, windows and cutoffs. |
| `feature_manifest_sha256` | Canonical SHA256 of `tlab_warmup.representations.manifest()`. Freezes names, column order, families, groups and semantics. Arbitrary selected-feature overrides fail. |

Active batch configuration fields are mapped unchanged. Threads, budgets, guards, row caps and bootstrap replicates are supported in both modes. In iteration1 only its panel caps may change, with the entire anchor cohort/training sample and at least all anchor development queries retained. Overnight supports calibration/main caps. Search limits/thresholds and additional model seed offsets may change only in overnight without `controls_only`. Inactive fields and panels must equal the declared upstream default configuration; changing them fails instead of accepting an ineffective setting. `data` must be null and `iteration1_output` must be `runs/iteration1`: effective data/readiness paths come exclusively from input references. Optional deferred branches cannot be enabled by changing their status strings.

Iteration1 computes both U20/P20 and native random/popularity/recency/original/no-pair controls. Overnight uses its native fixed representations, search, freeze and refit queue, or its controls-only queue. Individual arbitrary methods/features are not selectable through this bridge. Realized model seeds, exact feature names/order, query IDs, candidate order, scores and positive labels are exported. Familiarity is read from the original strict-past `user_item_count` column, including alternative candidates. Native saved tie keys are retained when present; current batch exports omit them, so the bridge calls the verified upstream pure `core.tie_key` function using the base seed. No labels enter that function.

Each `data` reference has `{id, local_path, manifest}`. Exactly one of each required kind is accepted:

| Manifest kind | `local_path` | Manifest fields |
|---|---|---|
| `warmup_source_v1` | Original warmup project directory | `kind,files,tree_sha256,python,python_sha256,environment`, generated below. |
| `warmup_anchor_v1` | Original `provenance/iteration0/results.tar.gz` | `kind,sha256` |
| `warmup_raw_v1` | Existing pinned full-release Parquet | `kind,sha256`; also checked against archived bytes/hash. |
| `warmup_readiness_v1` | Successful prior iteration1 `readiness.json` | `kind,sha256`; required only for overnight. Native code-hash/readiness gate remains active. |

The source manifest inventories Python source, upstream tests, JSON configs, `pyproject.toml` and `uv.lock`. Its environment records the actual interpreter/version, executable hash, OS/architecture, package versions and feature dictionary. The source files, interpreter and environment must still match before launch; source/environment/data/archive identities are checked again after completion. This captures installed versions rather than claiming that a lockfile proves the installed environment matches it. Use the project's frozen installation for actual reproduction; a changed environment can fail the archived prediction criterion.

## Prepare an actual iteration1 configuration

Set these **absolute paths** for your installation. Keep `M19_SETUP_DIR` outside the warmup project and use a new path. This preparation reads existing inputs and creates explicit manifests/configuration; it does not run experiments.

```bash
export RF_ROOT=/absolute/path/to/researchflow
export WARMUP=/absolute/path/to/tlab_warmup_clean
export WARMUP_PYTHON=/absolute/path/to/tlab_warmup_clean/.venv/bin/python
export WARMUP_DATA=/absolute/path/to/multi_event.parquet
export M19_SETUP_DIR=/absolute/path/to/new-m19-anchor-inputs

"$WARMUP_PYTHON" - <<'PY'
import importlib.util, json, os
from pathlib import Path
rf=Path(os.environ['RF_ROOT'])
spec=importlib.util.spec_from_file_location('warmup_bridge',rf/'runners/warmup/bridge.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
project=Path(os.environ['WARMUP']).resolve()
raw=Path(os.environ['WARMUP_DATA']).resolve()
python=Path(os.environ['WARMUP_PYTHON']).absolute()  # preserve venv symlink spelling
out=Path(os.environ['M19_SETUP_DIR']).resolve();out.mkdir(parents=True,exist_ok=False)
archive=project/'provenance/iteration0/results.tar.gz'
anchor=b.anchor_config(archive)
source=b.capture_source(project,python)
refs=[]
for name,path,manifest in [
 ('source',project,source),
 ('anchor',archive,{'kind':'warmup_anchor_v1','sha256':b.sha(archive)}),
 ('raw',raw,{'kind':'warmup_raw_v1','sha256':b.sha(raw)})]:
 manifest_path=out/(name+'-manifest.json');b.write(manifest_path,manifest)
 refs.append({'id':name,'manifest':str(manifest_path),'local_path':str(path)})
cfg=json.loads((project/'configs/experiments.json').read_text())
# Preserve the source's cohort/caps. Use the existing iteration1 expanded cap.
bounds=anchor['resolved_boundaries']
scientific={'experiment_id':'E-warmup-anchor','branch_id':'B-warmup-anchor',
 'hypothesis_ids':[],'phase':'fixture' if anchor['source']['synthetic'] else 'development',
 'runner_id':'warmup','protocol':{'id':'warmup-development-v1','candidates':['U20','P20'],
 'history':'strict-past-complete','train_end_timestamp':bounds['train_end_exclusive'],
 'development_end_timestamp':bounds['development_end_exclusive'],'timestamp_unit':'seconds','final_test':'not-scored'},
 'parameters':{'batch_phase':'iteration1','controls_only':False,'seed':anchor['seed'],
 'anchor_config_sha256':b.digest(anchor),'feature_manifest_sha256':b.digest(source['environment']['features']),
 'batch_config':cfg}}
experiment={'schema_version':1,'scientific':scientific,'runtime':{
 'command':[str(python),str(rf/'runners/warmup/runner.py')],
 'code_paths':[str(rf/'runners/warmup/runner.py'),str(rf/'runners/warmup/bridge.py')],
 'data_refs':refs,'research_paths':[],
 'resources':{'timeout_seconds':23000,'memory_limit_bytes':None}}}
b.write(out/'warmup-experiment.json',experiment)
print(out/'warmup-experiment.json')
PY
```

After the infrastructure dependencies have merged, execute the generated configuration with its documented public CLI:

```bash
uv run researchflow experiment --root "$RF_ROOT" --config "$M19_SETUP_DIR/warmup-experiment.json" --run-id R-warmup-anchor-001
```

That is an actual-data run when using the supplied real anchor/raw file. Review the returned payload's `result.json`, `provenance.json`, `upstream.log` and native `workspace/runs/R-warmup-anchor-001/jobs/U20_original/result.json`. Acceptance requires `anchor_prediction_max_absolute_difference <= 1e-8`. Upstream also verifies original U20 candidates and all archived original features at tight tolerance, and verifies complete-history extraction. Do not treat synthetic/fake tests or saved-score normalization as this acceptance. Do not launch a final phase for verification.

The standalone invocation requires an already resolved common request, with input manifest **objects**, and needs no installed researchflow package:

```bash
"$WARMUP_PYTHON" "$RF_ROOT/runners/warmup/runner.py" --request /absolute/runner_request.json --output /absolute/new-payload
```

A source manifest alone can also be captured with `runner.py --capture-source --project "$WARMUP" --python "$WARMUP_PYTHON" --output /absolute/new-source-manifest.json`.

For overnight, prepare a distinct experiment/configuration with `batch_phase=overnight`, provide the checksum-declared successful iteration1 readiness file as a fourth data reference, and keep identical source/lock hashes. The adapter copies only that readiness record to its isolated `workspace/runs/iteration1/readiness.json`; native validation compares its computational code hash. Use a distinct run ID and output.

## Output safety, resume and provenance

The original project is never the execution root. A verified copy of the declared small source/config/test/lock files is created in the selected payload's `workspace/`. Native data caches, tests, job directories and archive remain there. The explicit raw and anchor files are read in place. Input/output overlap, symlink aliases, preexisting foreign outputs and completed payload reuse fail. No cleanup deletes old output. This is isolation of the inspected trusted code's write locations, not an OS sandbox for arbitrary upstream code.

Rerunning the exact same request into an interrupted bridge payload delegates to the **same native run directory**, plan identity, completed-job checksums, STOP handling and job-level resume rules. Nothing converts another prior output into a compatible run. Native code/environment/configuration changes or deterministic failures can require a new output identity. Remove a native `workspace/runs/<run_id>/STOP` explicitly to resume after STOP. An interruption during the initial source-copy step before `bridge-input.json` is committed requires a new payload. Source identity is content-based; Git discovery stops at the staging parent so the containing infrastructure checkout's changing Git status does not change a native resume plan.

`provenance.json` retains original source path/file/lock hashes, selected Python/environment, full anchor config/hash, raw hash, exact effective command/config, adapter hashes/environment, native plan and output archive hash. Original upstream archives remain unmodified. Standard `ranking_npz_v1` predictions preserve row order and uint64 tie keys and compute canonical score-independent query/candidate identities. Upstream metrics are retained in the native archive rather than presented as freshly recomputed port metrics. If a 64-bit tie-key collision occurs, upstream's last tiebreak uses integer item IDs while the port specifies lexical strings; keys are not modified to hide that distinction.

## Verification

```bash
python -m pytest tests/modules/test_m19_warmup_bridge.py tests/contracts -q
# Optional inspection of the supplied SAVED development export; no fitting:
M19_WARMUP_PROJECT=/absolute/path/to/tlab_warmup_clean python -m pytest tests/modules/test_m19_warmup_bridge.py tests/contracts -q
```

Synthetic fake-CLI tests launch with the exact Python interpreter running pytest, using a quoted POSIX shell launcher. They do not select `python3` from PATH; paths containing spaces or quotes are supported. Regression tests exercise an unrelated PATH Python without NumPy and verify the former failure before checking the corrected launcher.

The optional export test checks artifact hashes and five saved development query groups, labels/scores, IDs, candidate order, familiarity and tie keys against the actual pure upstream core. It does not score raw/final data or demonstrate fresh anchor reproduction. See `docs/implementation/M19.md` for the executed checks and remaining acceptance.
