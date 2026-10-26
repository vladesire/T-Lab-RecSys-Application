# External scientific analyzer port v1

This boundary lets an author implement a new deterministic scientific analyzer without reading researchflow source.

Invocation: COMMAND --request /absolute/analysis_request.json --output /absolute/output-dir

Read only the explicit immutable input bundles and resolved analysis config. Verify/understand the supplied run contract; do not execute experiments, fit models or acquire raw data implicitly. Write output-dir/analysis.json conforming to analysis_result.schema.json and any contained tables needed for audit. The service imports the complete contained output and produces the outer analysis bundle/report. No unlisted external dependency may be required to read the exported numerical evidence.

Use stable series IDs, explicit protocol/split/subgroup/weighting, input fingerprints, coverage, costs, limitations and checks. Estimates and uncertainty require a valid declared method. Missing comparisons have status=unavailable, null estimate/interval and a reason. Do not substitute zero or manufacture confidence intervals from aggregate values.

Metric cells contain series_id plus common metric_row fields. Metric names include K when applicable (ndcg@10, recall@10). A comparison's left-minus-right estimate follows the same name/weighting/subgroup. Scientific methods outside the single-positive ranking convention require their own documented payload interpretation; the generic envelope is not verification of their science.

Keep reproduction/estimator-seed/search-seed/population evidence distinct. Output facts and explicit scope; an LLM's interpretation is a separate review, not a numerical result. Include code/config identities and independent small reference tests.
