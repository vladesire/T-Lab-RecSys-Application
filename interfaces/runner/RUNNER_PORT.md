# External experiment runner port v1

This is the complete infrastructure boundary for an independent runner author. You do not need researchflow implementation source.

Invocation: COMMAND --request /absolute/runner_request.json --output /absolute/payload-directory

The payload directory exists and is empty. Read only explicitly provided scientific config/data references. Validate all branch-specific parameters, including unknown keys; never silently ignore a consequential choice. Write payload/result.json conforming to runner_result.schema.json and its declared files. Exit zero only after complete output. On error exit nonzero and explain on stderr; infrastructure preserves the failed attempt. Do not write the run manifest: the infrastructure owns it.

Request scientific.phase is fixture/development/final. The outer executor authorizes final work through a frozen configuration. A runner must preserve the declared protocol/history rules and must not score final targets during development. An envelope check cannot verify temporal feature correctness; the branch's independent reference tests remain necessary.

Result kind=ranking has one or more predictions descriptors referring to NPZ files below payload. kind=metrics can expose non-ranking metrics for a suitable external analyzer; the built-in ranking analyzer requires ranking payloads. All artifact paths are safe relative paths. Include coverage, costs, checks and limitations even when empty. Unverified assumptions belong in checks/limitations. No model/LLM narrative counts as observed evidence.

See RANKING_NPZ.md for the v1 ranking payload. Inspect schema_examples for structurally valid SYNTHETIC examples. Zero hashes there are placeholders; calculate real fingerprints for actual output. Runners can use any local environment as long as the command/dependency/code identity is captured.

Minimal runner deliverable: runner.py, its explicit config example, dependency declaration, tests with an independent small reference, and README. It must not import the infrastructure package. Scientific helper modules supplied beside the runner are allowed and must be included in declared code_paths. No metadata claim of a successful full pipeline follows from a runner-only test.
