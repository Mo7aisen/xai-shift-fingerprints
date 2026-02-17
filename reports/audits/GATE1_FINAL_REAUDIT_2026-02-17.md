# Gate 1 Final Re-Audit (Post-Backfill Integrity)

Date: 2026-02-17
Decision: **GO (constrained pilot only)**

## Summary Table
| Check | Status | Evidence |
|---|---|---|
| GT leakage risk in deployment endpoint | **PASS** | `endpoint_mode` gating in `src/xfp/fingerprint/runner.py`; deployment paths (`predicted_mask`, `mask_free`) do not consume GT mask for feature reduction. |
| Metadata backfill (`patient_id`) completeness | **PASS** | `python scripts/backfill_patient_id.py` completed; summary reports show `total_missing_after: 0`. |
| Backfill safety (non-destructive + reversible) | **PASS** | Timestamped backups created: `data/interim/*/*/metadata.parquet.bak.20260217T094235Z`. |
| Dataset-wide validation test | **PASS** | `pytest -q tests/test_all_metadata_have_patient_id.py ...` → all passing. |
| Run registry traceability entry with `input_data_hash` | **PASS** | Entry added to `reports_v2/run_registry.csv` for pre-pilot JSRT seed 42. |
| Protocol lock integrity | **PASS** | `configs/protocol_lock_v1.yaml` remains frozen (`1.0.0`, append-only). |

## Validation Outputs
- Backfill dry-run report: `reports_v2/audits/backfill_patient_id_20260217T094231Z.json`
- Backfill execution report: `reports_v2/audits/backfill_patient_id_20260217T094235Z.json`
- Backfill execution CSV: `reports_v2/audits/backfill_patient_id_20260217T094235Z.csv`
- Registry entry: `reports_v2/run_registry.csv`

## Test Command
- `pytest -q tests/test_all_metadata_have_patient_id.py tests/test_gt_leakage.py tests/test_smoke.py tests/test_metadata.py tests/test_shift_metrics.py`
- Result: **11 passed**

## Gate Scope and Constraint
This Gate 1 final status authorizes only:
1. Pilot run(s) with seed 42.
2. Endpoints: `predicted_mask` and `mask_free` only.
3. Small dataset subset (JSRT pilot track).

This Gate does **not** authorize full/heavy GPU reruns.
