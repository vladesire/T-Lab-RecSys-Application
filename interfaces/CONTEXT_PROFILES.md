# Context source profiles v1

Profiles restrict what is supplied to future workers. These are context-selection rules, not an OS sandbox.

runner_author roles: port, schema, example, hypothesis, protocol, config, decision. Mandatory: runner port + at least one config/protocol and the selected scientific intent (hypothesis or task description). Never include workflow source. Port/schema/example files come from interfaces/runner; other inputs are explicitly named research/config records.

analyzer_author roles: port, schema, example, run_manifest, sample_payload, analysis_config, hypothesis. Mandatory: analyzer port, run contract/representative bundle, requested analysis intent. No workflow implementations. Sample payload is explicit and small; no silent large-payload truncation.

results_reviewer roles: analysis, hypothesis, config, decision, history, checks. Mandatory: full selected analysis.json (including phases/origins/checks/limitations), selected intent and exact source references. Huge source arrays can be referenced by checksummed attachments, not silently omitted facts. Do not include infrastructure source.

state_reviewer roles: state, hypothesis, decision, journal, evidence. Mandatory: full selected view.json including exclusions/unresolved entries and its selected intent/decisions. No whole-repository upload.

max_bytes counts UTF-8 prompt/evidence context, including mandatory profile instructions. If required facts exceed the budget, fail with a useful source-size breakdown. Manifest lists all included and excluded explicit sources. Absolute local paths are operational pointers, not scientific dataset identities.
