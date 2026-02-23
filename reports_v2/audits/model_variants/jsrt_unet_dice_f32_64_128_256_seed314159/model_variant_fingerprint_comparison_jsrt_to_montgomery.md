# Model Variant Fingerprint Comparison

- Experiment: `jsrt_to_montgomery`
- ID dataset: `jsrt`
- OOD dataset: `montgomery`
- Score method: `all_mean_abs_z`
- Top-k features: `20`
- Canonical root: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/data/fingerprints`
- Variant root: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/data/fingerprints_model_variants/jsrt_unet_dice_f32_64_128_256_seed314159`

| Endpoint | Canon AUROC | Var AUROC | ΔAUROC | Canon AUPR | Var AUPR | ΔAUPR | Canon FPR95 | Var FPR95 | Top-k Jaccard | Effect Corr | Score Corr ID | Score Corr OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mask_free | 0.8560 | 0.9192 | 0.0632 | 0.7558 | 0.8694 | 0.1136 | 0.5223 | 0.4170 | 0.4815 | 0.7232 | 0.4568 | 0.5872 |
| predicted_mask | 0.8964 | 0.8491 | -0.0474 | 0.8367 | 0.7713 | -0.0654 | 0.3887 | 0.6599 | 0.5385 | 0.7759 | 0.5625 | 0.5773 |

## Notes

- `Top-k Jaccard` compares the top-k discriminative fingerprint features between canonical and variant runs.
- `Effect Corr` is Pearson correlation of standardized feature-effect magnitudes across common features.
- `Score Corr ID/OOD` are per-sample Pearson correlations after matching `sample_id`.

- CSV: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/model_variants/jsrt_unet_dice_f32_64_128_256_seed314159/model_variant_fingerprint_comparison_jsrt_to_montgomery.csv`
- JSON: `/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/model_variants/jsrt_unet_dice_f32_64_128_256_seed314159/model_variant_fingerprint_comparison_jsrt_to_montgomery.json`
