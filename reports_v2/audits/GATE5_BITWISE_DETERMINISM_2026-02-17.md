# Gate-5 Bitwise Determinism Re-Audit

- Generated UTC: `2026-02-17T14:49:07.748742+00:00`
- Root: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/gate5/determinism`
- Replay A/B: `run1` vs `run2`
- Experiment: `jsrt_to_montgomery`

| Endpoint | Match | Files A | Files B | Mismatched files | Hash A | Hash B |
|---|---|---:|---:|---:|---|---|
| predicted_mask | PASS | 3 | 3 | 0 | `4206a4989a7d` | `4206a4989a7d` |
| mask_free | PASS | 3 | 3 | 0 | `db05b5d4450b` | `db05b5d4450b` |

## Final Decision

- Match rate: `1.00`
- Bitwise deterministic: `PASS`
- CSV: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_BITWISE_DETERMINISM.csv`
- JSON: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/GATE5_BITWISE_DETERMINISM_SUMMARY.json`
