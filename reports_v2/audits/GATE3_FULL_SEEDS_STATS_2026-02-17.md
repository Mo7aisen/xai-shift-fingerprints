# Gate-3 Statistical Analysis (Full Seeds)

- Generated UTC: `2026-02-17T13:31:53.923073+00:00`
- Experiment: `jsrt_to_montgomery`
- Seeds: `[42, 43, 44, 45, 46]`
- Endpoints: `['predicted_mask', 'mask_free']`
- Artifacts root: `reports_v2/gate3_seed_artifacts`

## Criteria

- AUROC seed-CI width `< 0.06`
- Feature-stability Jaccard mean `> 0.75`
- Post-hoc power mean `> 0.80`

## Endpoint Summary

| Endpoint | AUROC mean | Seed-CI width | Mean boot CI width | Jaccard mean | Power mean | AUROC CI | Jaccard | Power | PASS |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| predicted_mask | 0.8964 | 0.0000 | 0.0611 | 1.0000 | 1.0000 | PASS | PASS | PASS | PASS |
| mask_free | 0.8560 | 0.0000 | 0.0740 | 1.0000 | 1.0000 | PASS | PASS | PASS | PASS |

## Final Decision

- Gate-3 statistical status: `FULL PASS`

## Per-Seed Metrics

- CSV: `reports_v2/audits/GATE3_FULL_SEEDS_STATS.csv`
- JSON: `reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY.json`
