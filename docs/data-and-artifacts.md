# Data and saved artifacts

The dataset identity is recorded in [dataset.json](../reproducibility/dataset.json): `yandex/yambda`, revision `dd6f3a19eef5866e346c3270e098baa641a44948`, file `flat/50m/multi_event.parquet`, 384,465,869 bytes, SHA256 `b92f6c97277a554c8a7d6cfea900eca51639fd53a3e6c4fa58b82e35fd35804d`.

`make data` acquires and verifies this file. The full raw dataset is excluded from Git. The repository includes derived queries, identifiers, scores, metrics and models, not an identifier-free dataset.

The [published runs](../runs/published/README.md) contain the complete development analysis and selected final evidence. File inventories and hashes are checked by `tools/verify_saved_runs.py`. Original source manifests describe original runtimes; they are retained as references and are not rewritten to hide excluded files. Feature matrices, extracted interaction histories, environments and transient workspaces are recomputed locally.

The initial reference archive at `engines/tlab_warmup/provenance/iteration0/results.tar.gz` is required by the engine and is included. It contains derived reference records, not the full interaction release. Baseline and Mac-run evidence are retained under `evidence/`; these are selected exports rather than complete native workspaces. Repeating the same cohort is reproducibility evidence, not an independent statistical replication.

Both dependency locks and the engine source inventory are retained. Historical provenance may contain original absolute paths; fresh runs capture new local paths and identities. No GitHub Release download is required to rebuild the paper from this checkout.
