# Launch Pilot (Constrained GPU)

## Scope (Locked)
- Dataset: `jsrt` subset `pilot5`
- Seed: `42`
- Endpoints: `predicted_mask`, `mask_free`
- `upper_bound_gt` is forbidden for pilot launch.

## One-Command Preflight
```bash
cd /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full
./scripts/preflight_gpu.sh
```

If preflight fails, do not launch pilot.

## One-Command Pilot Launch (after preflight PASS)
```bash
cd /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full
./scripts/run_pilot_gpu_constrained.sh pilot5 jsrt_baseline
```

Hard constraints are enforced by the launcher:
- preflight must pass
- scope lock (`seed=42`, `jsrt/pilot5`, endpoints only `predicted_mask` and `mask_free`)
- runtime budget 30 minutes total (`MAX_RUNTIME_SEC=1800`)
- registry lifecycle updates and output hashes

## Manual Constrained GPU Commands (approved scope only)
```bash
cd /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full
python scripts/run_fingerprint.py --experiment jsrt_baseline --subset pilot5 --device cuda --endpoint-mode predicted_mask
python scripts/run_fingerprint.py --experiment jsrt_baseline --subset pilot5 --device cuda --endpoint-mode mask_free
pytest -q tests/test_all_metadata_have_patient_id.py tests/test_gt_leakage.py tests/test_smoke.py tests/test_metadata.py tests/test_shift_metrics.py
```

## Post-Run Requirements
- Update `reports_v2/run_registry.csv` with final `status` and `output_hash`.
- Attach logs under `logs_v2/`.
- Record pass/fail in Gate readiness checklist.
