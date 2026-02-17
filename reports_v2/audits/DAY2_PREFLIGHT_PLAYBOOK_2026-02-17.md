# Day 2 Preflight & Playbook Report

Date: 2026-02-17
Decision: PASS (readiness), NO-LAUNCH (current GPU window)

## Delivered Artifacts
- `scripts/preflight_gpu.sh`
- `scripts/run_pilot_gpu_constrained.sh`
- `reports_v2/LAUNCH_PILOT.md`
- `reports_v2/audits/GATE_READINESS_DAY2_2026-02-17.md`
- `reports_v2/audits/PILOT_ABORT_CONDITIONS_2026-02-17.md`

## Preflight Verification
- Command: `./scripts/preflight_gpu.sh`
- Observed: **FAIL as designed**
- Reason: GPU memory used `36707 MiB` > threshold `8000 MiB`

## Enforcement Confirmed
- Launch script blocks scope drift (`jsrt/pilot5`, seed 42, endpoints only `predicted_mask` and `mask_free`).
- Launch script enforces preflight before any run.
- Launch script enforces 30-minute runtime budget (`MAX_RUNTIME_SEC=1800`).
- Launch script appends lifecycle entries to `reports_v2/run_registry.csv`.

## Operational Status
- System is ready for immediate pilot launch when preflight passes.
- Current status remains no-launch until GPU VRAM falls within threshold.
