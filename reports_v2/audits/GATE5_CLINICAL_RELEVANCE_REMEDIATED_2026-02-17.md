# Gate-5 Clinical Relevance Report

- Generated UTC: `2026-02-17T14:55:57.346640+00:00`
- Experiment: `jsrt_to_montgomery`
- Seed set: `[42, 43, 44, 45, 46]`
- Correlation threshold: `0.60`
- Shift score method: `topk_weighted_abs_z`
- Shift score top-k: `5`
- Dice reference mean (jsrt): `0.980383`

## Endpoint Summary

| Endpoint | Pearson(all) mean [95% CI] | Spearman(all) mean [95% CI] | Pearson(ood) mean | Spearman(ood) mean | PASS |
|---|---|---|---:|---:|---|
| predicted_mask | 0.7200 [0.7200, 0.7200] | 0.7152 [0.7152, 0.7152] | 0.5277 | 0.5875 | PASS |
| mask_free | 0.6042 [0.6042, 0.6042] | 0.6236 [0.6236, 0.6236] | 0.3819 | 0.3206 | PASS |

## Artifacts

- Per-seed metrics: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_CLINICAL_PER_SEED_REMEDIATED.csv`
- Qualitative cases: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_CLINICAL_CASES_REMEDIATED.csv`
- JSON summary: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_CLINICAL_SUMMARY_REMEDIATED.json`

## Final Decision

- Gate-5 status: `PASS`
