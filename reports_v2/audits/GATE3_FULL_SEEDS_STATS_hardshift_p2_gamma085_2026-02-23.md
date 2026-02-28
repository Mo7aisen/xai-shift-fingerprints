# Gate-3 Statistical Analysis (Full Seeds)

- Generated UTC: `2026-02-23T23:23:10.375012+00:00`
- Experiment: `shenzhen_to_shenzhen_pp_gamma085`
- ID dataset: `shenzhen`
- OOD dataset: `shenzhen_pp_gamma085`
- Seeds: `[42, 43, 44, 45, 46]`
- Endpoints: `['predicted_mask', 'mask_free']`
- Artifacts root: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/gate3_seed_artifacts`

## Criteria

- AUROC seed-CI width `< 0.06`
- Feature-stability Jaccard mean `> 0.75`
- Post-hoc power mean `> 0.80`

## Endpoint Summary

| Endpoint | AUROC mean | AUPR mean | Seed-CI width | Mean AUROC boot CI width | Mean AUPR boot CI width | FPR95 mean | ECE mean | Brier mean | Jaccard mean | Power mean | AUROC CI | Jaccard | Power | PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| predicted_mask | 0.5221 | 0.5296 | 0.0000 | 0.0933 | 0.0862 | 0.9329 | 0.3208 | 0.3591 | 1.0000 | 0.2274 | PASS | PASS | FAIL | FAIL |
| mask_free | 0.5233 | 0.5324 | 0.0000 | 0.0936 | 0.0871 | 0.9399 | 0.3205 | 0.3586 | 1.0000 | 0.2663 | PASS | PASS | FAIL | FAIL |

## Final Decision

- Gate-3 statistical status: `NO-GO / CONDITIONAL`

## Per-Seed Metrics

- CSV: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_gamma085_2026-02-23.csv`
- JSON: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY_hardshift_p2_gamma085_2026-02-23.json`
