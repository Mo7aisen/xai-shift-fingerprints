# Model Variant Fingerprint Comparison

- Experiment: `jsrt_to_montgomery`
- ID dataset: `jsrt`
- OOD dataset: `montgomery`
- Score method: `all_mean_abs_z`
- Top-k features: `20`
- Canonical root: `data/fingerprints`
- Variant root: `data/fingerprints_model_variants/jsrt_unet_dice_f24_48_96_192_seed314159`

| Endpoint | Canon AUROC | Var AUROC | ΔAUROC | Canon AUPR | Var AUPR | ΔAUPR | Canon FPR95 | Var FPR95 | Top-k Jaccard | Effect Corr | Score Corr ID | Score Corr OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mask_free | 0.8560 | 0.8633 | 0.0073 | 0.7558 | 0.7650 | 0.0092 | 0.5223 | 0.5263 | 0.5385 | 0.7857 | 0.4726 | 0.5534 |
| predicted_mask | 0.8964 | 0.8418 | -0.0547 | 0.8367 | 0.7446 | -0.0921 | 0.3887 | 0.5466 | 0.6000 | 0.7569 | 0.4994 | 0.5569 |

## Notes

- `Top-k Jaccard` compares the top-k discriminative fingerprint features between canonical and variant runs.
- `Effect Corr` is Pearson correlation of standardized feature-effect magnitudes across common features.
- `Score Corr ID/OOD` are per-sample Pearson correlations after matching `sample_id`.

- CSV: `reports_v2/audits/model_variants/jsrt_unet_dice_f24_48_96_192_seed314159/model_variant_fingerprint_comparison_jsrt_to_montgomery.csv`
- JSON: `reports_v2/audits/model_variants/jsrt_unet_dice_f24_48_96_192_seed314159/model_variant_fingerprint_comparison_jsrt_to_montgomery.json`
