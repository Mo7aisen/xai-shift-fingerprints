# Gate Readiness Checklist (Day 2)

Date: 2026-02-17
Status: IN PROGRESS

## Preflight Controls
- [x] `preflight_gpu.sh` exists and is executable.
- [x] Fails when VRAM usage exceeds 8GB.
- [x] Fails when any compute process is active.

## Pilot Scope Lock
- [x] Seed fixed to `42`.
- [x] Dataset/subset fixed to `jsrt/pilot5`.
- [x] Endpoints fixed to `predicted_mask`, `mask_free`.
- [x] `upper_bound_gt` excluded from pilot launch commands.
- [x] One-command GPU launcher exists: `scripts/run_pilot_gpu_constrained.sh`.

## Traceability
- [x] Run registry contains lifecycle entries (`running/completed/failed`).
- [x] `input_data_hash` present in run registry.
- [x] Validation test suite command documented.

## Abort Policy
- [x] Abort conditions documented.
- [x] Preflight failure implies immediate no-launch.

## Gate Decision
- Current decision: `PENDING` (awaiting GPU window and preflight PASS at launch time).
