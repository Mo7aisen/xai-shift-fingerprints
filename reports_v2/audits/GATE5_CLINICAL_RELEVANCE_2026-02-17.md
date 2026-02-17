# Gate-5 Clinical Relevance Report

- Generated UTC: `2026-02-17T14:33:23.796114+00:00`
- Experiment: `jsrt_to_montgomery`
- Seed set: `[42, 43, 44, 45, 46]`
- Correlation threshold: `0.60`
- Dice reference mean (jsrt): `0.980383`

## Endpoint Summary

| Endpoint | Pearson(all) mean [95% CI] | Spearman(all) mean [95% CI] | Pearson(ood) mean | Spearman(ood) mean | PASS |
|---|---|---|---:|---:|---|
| predicted_mask | 0.6323 [0.6323, 0.6323] | 0.5747 [0.5747, 0.5747] | 0.5642 | 0.5489 | FAIL |
| mask_free | 0.5756 [0.5756, 0.5756] | 0.5132 [0.5132, 0.5132] | 0.5333 | 0.4999 | FAIL |

## Artifacts

- Per-seed metrics: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_CLINICAL_PER_SEED.csv`
- Qualitative cases: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_CLINICAL_CASES.csv`
- JSON summary: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_CLINICAL_SUMMARY.json`

## Final Decision

- Gate-5 status: `NO-GO`
