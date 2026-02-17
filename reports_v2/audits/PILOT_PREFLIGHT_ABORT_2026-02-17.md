# Pilot Preflight Abort (GPU Constraint)

Date: 2026-02-17
Decision: ABORTED BEFORE START

## Constraint Check
- Max allowed VRAM usage before pilot start: 8000 MiB
- Observed GPU usage: 37433 MiB / 40960 MiB
- Observed compute process: `pid=2825817`, `python`, `37427 MiB`

## Result
Pilot GPU run was not started to comply with hard constraint.

## Next Condition to Proceed
- Re-check `nvidia-smi` and confirm used VRAM <= 8000 MiB before launch.
