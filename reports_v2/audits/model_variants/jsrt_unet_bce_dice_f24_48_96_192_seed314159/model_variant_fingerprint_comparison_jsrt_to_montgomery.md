# Model Variant Fingerprint Comparison

- Experiment: `jsrt_to_montgomery`
- ID dataset: `jsrt`
- OOD dataset: `montgomery`
- Score method: `all_mean_abs_z`
- Top-k features: `20`
- Canonical root: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/data/fingerprints`
- Variant root: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/data/fingerprints_model_variants/jsrt_unet_bce_dice_f24_48_96_192_seed314159`

| Endpoint | Canon AUROC | Var AUROC | ΔAUROC | Canon AUPR | Var AUPR | ΔAUPR | Canon FPR95 | Var FPR95 | Top-k Jaccard | Effect Corr | Score Corr ID | Score Corr OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mask_free | 0.8560 | 0.9213 | 0.0653 | 0.7558 | 0.8763 | 0.1204 | 0.5223 | 0.3401 | 0.4286 | 0.6754 | 0.5337 | 0.5287 |
| predicted_mask | 0.8964 | 0.8448 | -0.0516 | 0.8367 | 0.7704 | -0.0663 | 0.3887 | 0.6356 | 0.4815 | 0.5896 | 0.6064 | 0.5116 |

## Notes

- `Top-k Jaccard` compares the top-k discriminative fingerprint features between canonical and variant runs.
- `Effect Corr` is Pearson correlation of standardized feature-effect magnitudes across common features.
- `Score Corr ID/OOD` are per-sample Pearson correlations after matching `sample_id`.

- CSV: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/model_variants/jsrt_unet_bce_dice_f24_48_96_192_seed314159/model_variant_fingerprint_comparison_jsrt_to_montgomery.csv`
- JSON: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/model_variants/jsrt_unet_bce_dice_f24_48_96_192_seed314159/model_variant_fingerprint_comparison_jsrt_to_montgomery.json`
