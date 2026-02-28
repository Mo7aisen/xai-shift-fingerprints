# Gate-3 Statistical Analysis (Full Seeds)

- Generated UTC: `2026-02-28T20:37:48.691725+00:00`
- Experiment: `shenzhen_to_shenzhen_pp_jpeg90`
- ID dataset: `shenzhen`
- OOD dataset: `shenzhen_pp_jpeg90`
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
| predicted_mask | 0.5044 | 0.5089 | 0.0000 | 0.0940 | 0.0837 | 0.9717 | 0.2874 | 0.3442 | 1.0000 | 0.0400 | PASS | PASS | FAIL | FAIL |
| mask_free | 0.5123 | 0.5178 | 0.0000 | 0.0938 | 0.0858 | 0.9611 | 0.3105 | 0.3567 | 1.0000 | 0.0796 | PASS | PASS | FAIL | FAIL |

## Final Decision

- Gate-3 statistical status: `NO-GO / CONDITIONAL`

## Per-Seed Metrics

- CSV: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_jpeg90_2026-02-23.csv`
- JSON: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY_hardshift_p2_jpeg90_2026-02-23.json`
