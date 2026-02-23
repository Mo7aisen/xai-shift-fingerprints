# Acceptance Strengthening Note (2026-02-23)

## Objective

Increase acceptance probability by strengthening the **baseline comparison** evidence with reviewer-facing OOD metrics beyond AUROC.

## What was upgraded (code)

Implemented a shared OOD evaluation utility and wired it into the journal baseline pipeline:

- `src/xfp/utils/ood_eval.py`
- `scripts/run_resnet_feature_baseline.py`
- `scripts/run_energy_ood_baseline.py`
- `scripts/run_journal_ood_baselines.py`
- `scripts/build_journal_results_table.py`

New baseline metrics now exported:

- AUROC (+ bootstrap CI)
- AUPR (+ bootstrap CI)
- FPR@95TPR (+ bootstrap CI)
- TPR@5%FPR (+ bootstrap CI)
- ECE (min-max score normalization)
- Brier score (min-max score normalization)

## Validation status

- Local tests: `13 passed, 2 skipped`
- End-to-end smoke run: `jsrt_to_montgomery` on `pilot5` succeeded
- Full baseline runs completed (initial local + additional Slurm executions):
  - `jsrt_to_montgomery` (`subset=full`)
  - `jsrt_to_shenzhen` (`subset=full`)
  - `montgomery_to_jsrt` (`subset=full`, Slurm)
  - `montgomery_to_shenzhen` (`subset=full`, Slurm)
  - `shenzhen_to_jsrt` (`subset=full`, Slurm)
  - `shenzhen_to_montgomery` (`subset=full`, Slurm)

## Key findings (important for manuscript positioning)

### 1) Simple ResNet baseline is extremely strong on gross cross-dataset shifts

Across all six directed pairwise shifts among `JSRT`, `Montgomery`, and `Shenzhen`, the `mahalanobis_resnet50` baseline achieved:

- AUROC = `1.0000`
- AUPR = `1.0000`
- FPR@95TPR = `0.0000`
- TPR@5%FPR = `1.0000`

### 2) UNet confidence baselines are weak for these shifts

UNet entropy / MSP / maxlogit remain substantially weaker overall (direction-dependent, roughly AUROC `0.18-0.82` across the six runs), which supports a stronger comparison against segmentation-confidence-only detectors.

### 3) Consequence for scientific framing

Do **not** position attribution fingerprints as universally superior to generic image-feature OOD detectors on coarse dataset shifts.

Stronger and more defensible framing:

- attribution fingerprints are **complementary** to strong generic OOD detectors,
- they provide **spatially grounded interpretability** and **clinical relevance linkage** (Gate-5 evidence),
- they may be most useful in **harder / subtler / operational** shift settings where coarse appearance drift is not trivially separable.

### 4) Backbone/configuration generalization pilot completed (Slurm, factorized)

A focused JSRT segmentation-model ablation on `jsrt_to_montgomery` is now complete (3 variants; jobs `2730`, `2732`, `2733`):

- `loss_only` (`dice`, canonical width)
- `capacity_only` (narrower width, canonical `bce_dice`)
- `combined_change` (`dice` + narrower width)

Main result:

- `mask_free` endpoint improved vs canonical in **all three** variants (`ΔAUROC > 0`)
- `predicted_mask` endpoint degraded vs canonical in **all three** variants (`ΔAUROC < 0`) and consistently worsened `FPR95`

This materially strengthens the paper's honesty and rigor:

- the method is **not configuration-invariant**
- robustness depends on **endpoint design**
- `mask_free` is the stronger default endpoint for robustness-oriented claims

## Artifacts generated (2026-02-23)

- `results/baselines/jsrt_to_montgomery_full/journal_ood_baselines_jsrt_to_montgomery_full.csv`
- `results/baselines/jsrt_to_montgomery_full/journal_ood_baselines_jsrt_to_montgomery_full.md`
- `results/baselines/jsrt_to_shenzhen_full/journal_ood_baselines_jsrt_to_shenzhen_full.csv`
- `results/baselines/jsrt_to_shenzhen_full/journal_ood_baselines_jsrt_to_shenzhen_full.md`
- `results/baselines/montgomery_to_jsrt_full/journal_ood_baselines_montgomery_to_jsrt_full.csv`
- `results/baselines/montgomery_to_jsrt_full/journal_ood_baselines_montgomery_to_jsrt_full.md`
- `results/baselines/montgomery_to_shenzhen_full/journal_ood_baselines_montgomery_to_shenzhen_full.csv`
- `results/baselines/montgomery_to_shenzhen_full/journal_ood_baselines_montgomery_to_shenzhen_full.md`
- `results/baselines/shenzhen_to_jsrt_full/journal_ood_baselines_shenzhen_to_jsrt_full.csv`
- `results/baselines/shenzhen_to_jsrt_full/journal_ood_baselines_shenzhen_to_jsrt_full.md`
- `results/baselines/shenzhen_to_montgomery_full/journal_ood_baselines_shenzhen_to_montgomery_full.csv`
- `results/baselines/shenzhen_to_montgomery_full/journal_ood_baselines_shenzhen_to_montgomery_full.md`
- `reports_v2/audits/JOURNAL_MAIN_RESULTS_jsrt_to_montgomery_2026-02-23.csv`
- `reports_v2/audits/JOURNAL_MAIN_RESULTS_jsrt_to_montgomery_2026-02-23.md`
- `reports_v2/audits/BASELINE_CROSSDATASET_SUMMARY_2026-02-23.csv`
- `reports_v2/audits/BASELINE_CROSSDATASET_SUMMARY_2026-02-23.md`
- `reports_v2/audits/MODEL_VARIANT_GENERALIZATION_NOTE_2026-02-23.md`
- `reports_v2/audits/model_variants/ABLATION_SUMMARY_jsrt_to_montgomery_2026-02-23.csv`
- `reports_v2/audits/model_variants/ABLATION_SUMMARY_jsrt_to_montgomery_2026-02-23.md`
- `reports_v2/audits/model_variants/jsrt_unet_dice_f24_48_96_192_seed314159/`
- `reports_v2/audits/model_variants/jsrt_unet_dice_f32_64_128_256_seed314159/`
- `reports_v2/audits/model_variants/jsrt_unet_bce_dice_f24_48_96_192_seed314159/`

## Highest-yield next experiments (acceptance impact)

1. **Harder-shift settings where ResNet baseline is not trivial**
   - within-dataset/site/protocol stratification
   - acquisition-quality or preprocessing shifts
   - label-quality/annotation-heterogeneity sensitivity analysis

2. **Paired added-value analysis**
   - compare fingerprint score vs ResNet OOD score on matched samples
   - evaluate complementarity (e.g., combined model, conditional failure cases)

3. **Extended model-family generalization (beyond UNet config)**
   - second segmentation family / architecture class if feasible
   - validate that the endpoint-level pattern (`mask_free` more robust than `predicted_mask`) persists

## Recommended immediate run order

1. Treat the 6-direction coarse-shift baseline matrix as a **completed transparency result** (main text or supplement)
2. Prioritize a **harder-shift** experiment where ResNet Mahalanobis is not trivially saturated
3. Add a **paired complementarity analysis** (fingerprint score + ResNet OOD score) on matched cases
4. Expand to **model-family generalization** only after the harder-shift setting is selected
