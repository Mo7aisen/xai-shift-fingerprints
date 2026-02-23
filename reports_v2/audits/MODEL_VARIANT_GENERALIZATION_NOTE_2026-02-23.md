# Model Variant Generalization Note (2026-02-23)

## Objective

Test whether the fingerprint-based shift signal for `jsrt_to_montgomery` is stable under JSRT segmentation model configuration changes, and separate the effects of:

- loss function (`bce_dice` vs `dice`)
- UNet width profile (`32,64,128,256` vs `24,48,96,192`)

## Factorized ablation executed (Slurm)

- Job `2730`: `jsrt_unet_dice_f24_48_96_192_seed314159` (`combined_change`)
- Job `2732`: `jsrt_unet_dice_f32_64_128_256_seed314159` (`loss_only`)
- Job `2733`: `jsrt_unet_bce_dice_f24_48_96_192_seed314159` (`capacity_only`)

Common setup:

- Training cache: `data/interim/jsrt/full`
- Epochs: `50`
- Experiment for fingerprint reruns: `jsrt_to_montgomery`
- Endpoints: `predicted_mask`, `mask_free`

Artifacts:

- Per-variant reports under `reports_v2/audits/model_variants/<variant_tag>/`
- Combined ablation table:
  - `reports_v2/audits/model_variants/ABLATION_SUMMARY_jsrt_to_montgomery_2026-02-23.csv`
  - `reports_v2/audits/model_variants/ABLATION_SUMMARY_jsrt_to_montgomery_2026-02-23.md`

## Key results (canonical vs variant, factorized)

### `mask_free` endpoint
- `capacity_only` (`bce_dice`, narrower width):
  - `ΔAUROC +0.0653`, `ΔAUPR +0.1204`, `ΔFPR95 -0.1822`
  - Top-k Jaccard `0.4286`, effect corr `0.6754`
- `loss_only` (`dice`, canonical width):
  - `ΔAUROC +0.0632`, `ΔAUPR +0.1136`, `ΔFPR95 -0.1053`
  - Top-k Jaccard `0.4815`, effect corr `0.7232`
- `combined_change` (`dice` + narrower width):
  - `ΔAUROC +0.0073`, `ΔAUPR +0.0092`, `ΔFPR95 +0.0040`
  - Top-k Jaccard `0.5385`, effect corr `0.7857`

### `predicted_mask` endpoint
- `capacity_only` (`bce_dice`, narrower width):
  - `ΔAUROC -0.0516`, `ΔAUPR -0.0663`, `ΔFPR95 +0.2470`
  - Top-k Jaccard `0.4815`, effect corr `0.5896`
- `loss_only` (`dice`, canonical width):
  - `ΔAUROC -0.0474`, `ΔAUPR -0.0654`, `ΔFPR95 +0.2713`
  - Top-k Jaccard `0.5385`, effect corr `0.7759`
- `combined_change` (`dice` + narrower width):
  - `ΔAUROC -0.0547`, `ΔAUPR -0.0921`, `ΔFPR95 +0.1579`
  - Top-k Jaccard `0.6000`, effect corr `0.7569`

## Interpretation

1. **The endpoint-level pattern is now consistent across all three variants**:
   - `mask_free` improves vs canonical (`ΔAUROC` positive in all three variants)
   - `predicted_mask` degrades vs canonical (`ΔAUROC` negative in all three variants)
2. **Predicted-mask endpoint is configuration-sensitive in a practically important way**:
   - all variants worsen `FPR95` substantially (`+0.158` to `+0.271`)
3. **Mask-free is more robust as a scientific endpoint**, but not invariant:
   - feature overlap and effect correlations remain moderate (not identical)
   - sample-level rankings remain only moderate (`~0.46-0.61`)
4. **Loss and capacity effects are not simply additive**:
   - both single-factor changes improved `mask_free` more than the combined change
   - this suggests interaction effects in the segmentation model -> fingerprint pipeline

## Manuscript / review implication

This supports a stronger, more honest statement:

- attribution fingerprints are **model-configuration-sensitive**, especially for mask-dependent endpoints;
- endpoint choice materially affects robustness conclusions;
- **mask-free** formulations are a stronger default when robustness to segmentation-model details matters.

## Recommended next experiment (highest yield after this)

Move from coarse cross-dataset shifts to a **harder-shift setting** and repeat the paired comparison:

1. `ResNet Mahalanobis` baseline score
2. fingerprint score (`mask_free` primary, `predicted_mask` secondary)
3. combined/complementarity analysis on matched cases

This is the fastest path to an acceptance-relevant claim that is both stronger and more defensible than a pure superiority claim.
