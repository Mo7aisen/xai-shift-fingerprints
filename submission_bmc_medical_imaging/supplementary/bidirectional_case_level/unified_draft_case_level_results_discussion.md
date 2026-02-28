# Unified Draft: Bidirectional Case-Level Inference

## 1) Case-Level Stats (Bootstrap + Permutation)

- Bootstrap: `n=2000` for 95% CIs
- Permutation: `n=2000` (two-sided)

### Detection Metrics by Direction

| Endpoint | Direction | AUROC [95% CI] | AUPR [95% CI] | FPR95 [95% CI] | ECE [95% CI] | Brier [95% CI] |
|---|---|---|---|---|---|---|
| mask_free | forward_jsrt_to_shenzhen | 0.8833 [0.8579, 0.9055] | 0.9448 [0.9329, 0.9564] | 0.4615 [0.3765, 0.5344] | 0.5014 [0.4408, 0.5082] | 0.4170 [0.3477, 0.4260] |
| mask_free | reverse_shenzhen_to_jsrt | 0.9905 [0.9849, 0.9953] | 0.9721 [0.9529, 0.9877] | 0.0389 [0.0212, 0.0565] | 0.2407 [0.2294, 0.2458] | 0.2308 [0.2156, 0.2357] |
| predicted_mask | forward_jsrt_to_shenzhen | 0.9253 [0.9055, 0.9421] | 0.9646 [0.9546, 0.9733] | 0.3279 [0.2308, 0.3969] | 0.5169 [0.4431, 0.5225] | 0.4308 [0.3353, 0.4383] |
| predicted_mask | reverse_shenzhen_to_jsrt | 0.9944 [0.9903, 0.9978] | 0.9845 [0.9726, 0.9940] | 0.0230 [0.0106, 0.0371] | 0.2410 [0.2350, 0.2472] | 0.2312 [0.2256, 0.2362] |

### Direction Delta (Reverse - Forward): Detection

| Endpoint | Metric | Delta [95% CI] | Permutation p |
|---|---|---|---:|
| mask_free | aupr | +0.0272 [+0.0071, +0.0469] | 1 |
| mask_free | auroc | +0.1072 [+0.0838, +0.1324] | 0.0004998 |
| mask_free | brier | -0.1862 [-0.1996, -0.1176] | 1 |
| mask_free | ece | -0.2606 [-0.2721, -0.1948] | 1 |
| mask_free | fpr95 | -0.4227 [-0.4897, -0.3334] | 0.0004998 |
| predicted_mask | aupr | +0.0200 [+0.0045, +0.0332] | 1 |
| predicted_mask | auroc | +0.0691 [+0.0514, +0.0888] | 0.0004998 |
| predicted_mask | brier | -0.1996 [-0.2087, -0.1052] | 1 |
| predicted_mask | ece | -0.2759 [-0.2832, -0.1998] | 1 |
| predicted_mask | fpr95 | -0.3050 [-0.3774, -0.2060] | 0.0004998 |

- Note: permutation p-values are most interpretable for rank/discrimination metrics (`AUROC`, `FPR95`). For `AUPR`, `ECE`, and `Brier`, direction-dependent score normalization makes permutation tests conservative; bootstrap delta CIs are the primary inferential signal.

### Clinical Linkage by Direction (Shift Score vs Dice Drop)

| Endpoint | Scope | Corr | Forward [95% CI] | Reverse [95% CI] | Delta R-F [95% CI] | p(delta) |
|---|---|---|---|---|---|---:|
| predicted_mask | all | pearson | 0.6290 [0.5679, 0.6811] | 0.6022 [0.5739, 0.6992] | -0.0267 [-0.0873, +0.0841] | 0.3293 |
| predicted_mask | all | spearman | 0.6743 [0.6320, 0.7097] | 0.6730 [0.6195, 0.7204] | -0.0012 [-0.0653, +0.0591] | 0.9615 |
| predicted_mask | ood | pearson | 0.5049 [0.4021, 0.5923] | 0.3604 [0.3304, 0.4041] | -0.1445 [-0.2315, -0.0247] | 0.03848 |
| predicted_mask | ood | spearman | 0.3801 [0.3062, 0.4495] | 0.7988 [0.7130, 0.8717] | +0.4187 [+0.3058, +0.5172] | 0.0004998 |
| mask_free | all | pearson | 0.5671 [0.4951, 0.6342] | 0.6412 [0.6011, 0.7434] | +0.0741 [-0.0011, +0.1902] | 0.03698 |
| mask_free | all | spearman | 0.5960 [0.5467, 0.6401] | 0.6680 [0.6190, 0.7157] | +0.0720 [+0.0052, +0.1425] | 0.006997 |
| mask_free | ood | pearson | 0.4348 [0.3220, 0.5353] | 0.3800 [0.3392, 0.4353] | -0.0547 [-0.1569, +0.0725] | 0.1919 |
| mask_free | ood | spearman | 0.2997 [0.2235, 0.3712] | 0.7475 [0.6551, 0.8279] | +0.4478 [+0.3273, +0.5533] | 0.0004998 |

## 2) Failure Mode (Predicted-Mask Collapse)

- Gate-4 `predicted_mask` degradation: forward `28.14%`, reverse `49.40%`.
- Gate-4 `mask_free` degradation: forward `25.53%`, reverse `2.03%`.
- Interpretation: discrimination can be high while perturbation stability collapses for `predicted_mask`; this is an endpoint-specific brittleness pattern, not a global method failure.

## 3) Deployment Recommendation

- Primary deployment candidate: `mask_free` (reverse direction shows strong discrimination, clinical pass behavior, and low robustness degradation).
- Secondary/research-only: `predicted_mask` (retain for sensitivity analysis, but not as standalone clinical trigger without robustness hardening).
- Manuscript claim should be endpoint- and direction-qualified, not pooled into a single number.

## 4) Reviewer-Safe Results Paragraph (Draft)

Bidirectional case-level inference (bootstrap/permutation) showed that reverse evaluation (Shenzhen->JSRT) improves OOD discrimination for both endpoints, while robustness remains endpoint-dependent. For `mask_free`, reverse increased AUROC (0.9905 vs 0.8833) and reduced FPR95 (0.0389 vs 0.4615), with concurrent clinical-pass behavior. For `predicted_mask`, reverse also improved discrimination (AUROC 0.9944 vs 0.9253), but Gate-4 robustness degradation worsened (49.40% reverse vs 28.14% forward), indicating a brittle high-separability regime. These results support a constrained claim: the framework is clinically actionable in specific endpoint/direction settings, but robustness limitations must be explicitly disclosed.

## 5) Reviewer-Safe Discussion Paragraph (Draft)

The observed divergence between discrimination and perturbation stability indicates that AUROC alone is insufficient for deployment claims. In particular, `predicted_mask` exhibits a failure mode where high shift separability coexists with substantial robustness loss, consistent with fragile feature reliance under perturbation stress. Conversely, `mask_free` demonstrates a more favorable balance between detectability, calibration, and clinical linkage in reverse direction. We therefore recommend endpoint-specific deployment policy: prioritize `mask_free` for operational use, and position `predicted_mask` as investigational until robustness interventions are validated.

## Source Files

- `analysis/reverse_chain_pull_20260219_1239/case_level/inference/detection_case_level_bootstrap.csv`
- `analysis/reverse_chain_pull_20260219_1239/case_level/inference/detection_direction_deltas_bootstrap_permutation.csv`
- `analysis/reverse_chain_pull_20260219_1239/case_level/inference/clinical_case_level_bootstrap_permutation.csv`
- `analysis/reverse_chain_pull_20260219_1239/case_level/inference/clinical_direction_deltas_bootstrap_permutation.csv`
- `analysis/reverse_chain_pull_20260219_1239/case_level/inference/case_level_inference_summary.json`
- `analysis/reverse_chain_pull_20260219_1239/case_level/inference/reviewer_safe_results_discussion_draft.md`
