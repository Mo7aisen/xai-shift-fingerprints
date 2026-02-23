# Cross-Dataset Baseline Summary (2026-02-23)

Completed full baseline suites for all directed pairwise shifts among JSRT, Montgomery, and Shenzhen using upgraded OOD metrics.

## ResNet Mahalanobis (generic image-feature baseline)

| Experiment | AUROC | AUPR | FPR95 | TPR@5%FPR | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| jsrt_to_montgomery_full | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.2450 | 0.1763 |
| jsrt_to_shenzhen_full | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.5615 | 0.4591 |
| montgomery_to_jsrt_full | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.4262 | 0.2947 |
| montgomery_to_shenzhen_full | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.5913 | 0.4481 |
| shenzhen_to_jsrt_full | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.1677 | 0.1036 |
| shenzhen_to_montgomery_full | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.1127 | 0.0698 |

## Best Non-ResNet Baseline per Direction

| Experiment | Method | AUROC | AUPR | FPR95 | TPR@5%FPR |
|---|---|---:|---:|---:|---:|
| jsrt_to_montgomery_full | unet_msp | 0.3850 | 0.2849 | 0.9433 | 0.0000 |
| jsrt_to_shenzhen_full | unet_maxlogit | 0.5807 | 0.7623 | 0.9109 | 0.1007 |
| montgomery_to_jsrt_full | unet_maxlogit | 0.7740 | 0.8485 | 0.6667 | 0.2794 |
| montgomery_to_shenzhen_full | unet_maxlogit | 0.8241 | 0.9468 | 0.6014 | 0.3852 |
| shenzhen_to_jsrt_full | unet_msp | 0.4420 | 0.2682 | 0.9912 | 0.0405 |
| shenzhen_to_montgomery_full | unet_msp | 0.3279 | 0.1383 | 0.9894 | 0.0000 |

## Key conclusion

- ResNet Mahalanobis is saturated (AUROC=1.0, FPR95=0) on all six coarse cross-dataset shifts.
- Attribution fingerprints should be positioned as complementary/interpretability-oriented on these coarse shifts, not as universally superior OOD detectors.
- Highest-yield next experiments are harder shifts (operational/protocol/quality) and model-family generalization.

- CSV: `reports_v2/audits/BASELINE_CROSSDATASET_SUMMARY_2026-02-23.csv`
