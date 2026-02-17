# Gate 1 Re-Audit Report (Post-P0 Code Remediation)

Date: 2026-02-17
Decision: **NO-GO (conditional, improved)**

## Summary Table
| Check | Status | Evidence |
|---|---|---|
| GT leakage in `data/interim/*.npz` | **FAIL** | Cache artifacts still contain GT masks by design (`image`, `mask`) and existing legacy cache has not been fully rematerialized. |
| GT mask used in deployment extraction path | **PASS** | `endpoint_mode` introduced; GT is loaded only in `upper_bound_gt` mode (`src/xfp/fingerprint/runner.py`). `predicted_mask` and `mask_free` avoid GT usage in feature reduction. |
| Patient-level split enforcement in training code | **PASS (code-level)** | Grouped patient split logic added in `scripts/train_unet.py` and `scripts/train_unet_from_cache.py`; fallback now group-aware rather than pure image-level random split. |
| `patient_id` propagation in cache metadata | **PASS (pipeline-level)** | `patient_id` now written during cache build in `src/xfp/data/pipeline.py`; validated on pilot cache (`data/interim/jsrt/pilot1/metadata.parquet`). |
| Protocol lock integrity | **PASS** | `configs/protocol_lock_v1.yaml` frozen as semantic version `1.0.0`, append-only. |
| Anti-leakage guards in tests | **PASS** | `tests/test_gt_leakage.py` added and passing. |

## Verification Commands (CPU-only)
- `pytest -q tests/test_gt_leakage.py tests/test_smoke.py tests/test_metadata.py tests/test_shift_metrics.py` → **10 passed**
- `python scripts/prepare_data.py --dataset jsrt --subset pilot1` → metadata includes `patient_id`

## What Changed in P0
1. Endpoint separation implemented:
   - `upper_bound_gt` (analysis-only)
   - `predicted_mask` (realistic deployment proxy)
   - `mask_free` (deployment primary)
2. Deployment-safe feature extraction paths introduced in runner.
3. Patient-level grouping added to training split logic.
4. `patient_id` inference and persistence added to data pipeline.
5. Static anti-leakage tests added.

## Remaining Blockers Before Gate 1 Can Become Full PASS
1. Existing full caches (especially NIH full) still need controlled rematerialization to ensure `patient_id` column is present everywhere.
2. Run a formal pilot with `--endpoint-mode predicted_mask` and `--endpoint-mode mask_free` to validate end-to-end outputs and registry entries.
3. Add explicit `input_data_hash` entries in `reports_v2/run_registry.csv` for each pilot run.

## Gate Decision Rationale
- Scientific risk is significantly reduced at code level.
- Operationally, legacy cache state still carries GT-mask artifacts and incomplete `patient_id` backfill.
- Therefore Gate remains **NO-GO** for heavy/full runs until cache/registry operational steps are completed.
