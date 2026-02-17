# Day 1 CPU Dry-Run Report

Date: 2026-02-17
Decision: PASS

## Scope
- Dataset: `jsrt` subset `pilot5`
- Seed: `42`
- Endpoints: `predicted_mask`, `mask_free`
- Device: CPU only (`CUDA_VISIBLE_DEVICES=""`)

## Run Results
- `predicted_mask`: completed
  - start: `2026-02-17T10:03:59Z`
  - end: `2026-02-17T10:06:10Z`
  - output hash: `5c1ff874327190f7e5a0ee7bc482d797a79b405ce799c9e604fb437cb88492c2`
  - log: `logs_v2/cpu_dryrun_jsrt_pilot5_predicted_mask_seed42_20260217T100359Z.log`
- `mask_free`: completed
  - start: `2026-02-17T10:06:10Z`
  - end: `2026-02-17T10:08:22Z`
  - output hash: `ad9275fb4547fd7673f69365417c362ddef2615ca04eae895f4a75e5f3774ae0`
  - log: `logs_v2/cpu_dryrun_jsrt_pilot5_mask_free_seed42_20260217T100610Z.log`

## Validation
- Command: `pytest -q tests/test_all_metadata_have_patient_id.py tests/test_gt_leakage.py tests/test_smoke.py tests/test_metadata.py tests/test_shift_metrics.py`
- Result: `11 passed`

## Registry
- Lifecycle entries (`running` then `completed`) appended to `reports_v2/run_registry.csv` for both endpoints.

## Notes
- Constraint preserved: no GPU run was started.
- Day 2 can proceed with launch playbook and preflight automation.
