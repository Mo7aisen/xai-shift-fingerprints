# Gate-4 IG Quality & Robustness Report

- Generated UTC: `2026-02-17T14:08:05.462320+00:00`
- Experiment: `jsrt_to_montgomery`
- Seed: `42`
- IG compare: `16 vs 32`

## Endpoint Results

| Endpoint | AUC(IG16) | AUC(IG32) | |Δ| | Top20 Retention | Max Robust Deg (%) | IG | Ablation | Robust | PASS |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| predicted_mask | 0.8964 | 0.9071 | 0.0107 | 1.0589 | 10.00 | PASS | PASS | PASS | PASS |
| mask_free | 0.8560 | 0.8726 | 0.0165 | 1.0149 | 12.03 | PASS | PASS | PASS | PASS |

## Final Decision

- Gate-4 status: `PASS`
- JSON summary: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE4_IG_ROBUSTNESS_SUMMARY.json`
