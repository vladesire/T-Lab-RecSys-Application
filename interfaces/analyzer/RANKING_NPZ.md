# ranking_npz_v1

NPZ loaded with numpy.load(..., allow_pickle=False). Required equal-length 1-D arrays:
query_id, user_id, item_id, timestamp, label, score, tie_key, is_familiar, is_repeat_target.

Identifiers use Unicode or integer arrays (no object dtype); normalize each ID to its exact string representation, never truncate. Timestamp is a nonnegative integer in the branch's declared time units. label, is_familiar and is_repeat_target are boolean/0-or-1 integer arrays. score is finite real numeric. tie_key is uint64 (or a validated nonnegative integer value within uint64). Every query has exactly one positive and distinct item IDs; user, timestamp and is_repeat_target are constant within query. Positive is_familiar must equal is_repeat_target. Query/item identifiers are stable across methods.

Rank by (-score, tie_key, normalized_item_id lexical ascending). Labels never break ties. Exported tie keys must be generated without labels/future outcomes; the run records the algorithm/source. Existing historical tie keys are preserved by adapters. A schema cannot independently establish their historical independence.

For positive rank r: recall=1[r<=K]; ndcg=1[r<=K]/log2(r+1). Single-positive IDCG=1. Repeat/novel refer to prior observed user-item history under the declared event semantics.

Identity byte convention: UTF-8 JSON, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False; SHA256 hex. Normalize NumPy scalars to Python scalars first.
- Per-query query record: {query_id,user_id,target_item_id,timestamp,is_repeat_target}; query_identity_sha256 hashes it.
- Per-query candidate record: sorted list of {item_id,is_familiar,tie_key}, ordered by normalized item_id. candidate_set_sha256 hashes this list. Scores are excluded; target identity is already in the query record.
- Whole query_set_sha256: hash the query records sorted by query_id.
- Whole candidate_set_sha256: hash sorted [{query_id,candidate_set_sha256}] by query_id.
- protocol_sha256: hash the resolved scientific.protocol object.

M07 returned per-query row keys: query_id,user_id,target_item_id,timestamp,is_repeat_target,query_identity_sha256,candidate_set_sha256,ndcg,recall,n_candidates,n_familiar_alternatives.

M07 aggregate output: {value: float|null,n_queries:int,n_users:int}. metric is ndcg or recall, weighting event or user, subgroup all/repeat/novel. Empty subgroup is null/0/0. M07 paired_interval output: {estimate:float,ci_low:float,ci_high:float,n_queries:int,n_users:int}. Empty paired subgroup raises CONTRACT; M08 emits an unavailable comparison with a reason.

Paired intervals resample contributing users jointly with replacement, using numpy.default_rng(seed), confidence quantiles [(1-c)/2,(1+c)/2], method='linear'. Event weighting uses the ratio of sampled user metric totals to sampled query counts, not a mean of user means. User weighting averages sampled user means. Both methods must align identical queries, users, target and candidate identities.

tests/fixtures/ranking_rows.json and ranking_expected.json provide independent tiny arithmetic checks. These files are synthetic and do not substantiate recommendation quality.
