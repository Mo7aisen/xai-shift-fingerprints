# Journal Main Results

- Experiment: `jsrt_to_montgomery`

## Fingerprint Endpoints

| Endpoint | AUROC | AUPR | FPR95 | ECE | Brier | Pearson(all) | Spearman(all) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mask_free | 0.8560 | nan | 0.5223 | 0.1537 | nan | 0.6042 | 0.6236 |
| predicted_mask | 0.8964 | nan | 0.3887 | 0.1535 | nan | 0.7200 | 0.7152 |

## OOD Baselines

| Method | AUROC | AUPR | FPR95 | TPR@5%FPR | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| mahalanobis_resnet50 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.2450 | 0.1763 |
| unet_entropy | 0.3824 | 0.2856 | 0.9514 | 0.0000 | 0.1982 | 0.2932 |
| unet_msp | 0.3850 | 0.2849 | 0.9433 | 0.0000 | 0.1806 | 0.2907 |
| unet_maxlogit | 0.2261 | 0.2387 | 1.0000 | 0.0000 | 0.3495 | 0.3574 |

- CSV: `reports_v2/audits/JOURNAL_MAIN_RESULTS_jsrt_to_montgomery_2026-02-23.csv`
- Baselines source: `results/baselines/jsrt_to_montgomery_full/journal_ood_baselines_jsrt_to_montgomery_full.csv`
