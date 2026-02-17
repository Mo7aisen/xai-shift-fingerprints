# Pilot Abort Conditions (Constrained GPU)

Date: 2026-02-17

Abort immediately if any condition occurs:

1. Preflight fails (`scripts/preflight_gpu.sh` non-zero exit).
2. GPU VRAM used > 8000 MiB before launch.
3. Any active compute process appears on target GPU before launch.
4. Runtime exceeds 30 minutes for constrained pilot scope.
5. Unexpected traceback or non-zero exit from endpoint run.
6. Endpoint or dataset scope drift (`upper_bound_gt` used, non-`pilot5`, seed not `42`).
7. Validation tests fail after run.

Required action on abort:
- Mark run as failed/aborted in `reports_v2/run_registry.csv`.
- Save and reference log path.
- Write short incident note under `reports_v2/audits/`.
