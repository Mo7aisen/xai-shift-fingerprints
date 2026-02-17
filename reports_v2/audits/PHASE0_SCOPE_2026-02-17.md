# Phase 0 Scope (Organization Only)

Date: 2026-02-17
Mode: CPU-only

Implemented:
- Protocol lock versioning upgraded to semantic version + append-only semantics.
- v2 namespaces created (`data/fingerprints_v2`, `results_v2`, `reports_v2`, `logs_v2`).
- Run registry template initialized with traceability fields.
- Freeze manifests generated for legacy artifacts and code/config trees.

Constraints:
- No GPU execution.
- No overwrite of legacy `results/` or legacy paper outputs.
