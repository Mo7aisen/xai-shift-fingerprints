# Model Variant Ablation Summary (jsrt_to_montgomery)

Variants found: `3`

## Endpoint: `mask_free`

| Variant | Type | Loss | Features | dAUROC | dAUPR | dFPR95 | TopK Jaccard | Effect Corr | Score Corr ID | Score Corr OOD |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| jsrt_unet_bce_dice_f24_48_96_192_seed314159 | capacity_only | bce_dice | 24,48,96,192 | 0.0653 | 0.1204 | -0.1822 | 0.4286 | 0.6754 | 0.5337 | 0.5287 |
| jsrt_unet_dice_f24_48_96_192_seed314159 | combined_change | dice | 24,48,96,192 | 0.0073 | 0.0092 | 0.0040 | 0.5385 | 0.7857 | 0.4726 | 0.5534 |
| jsrt_unet_dice_f32_64_128_256_seed314159 | loss_only | dice | 32,64,128,256 | 0.0632 | 0.1136 | -0.1053 | 0.4815 | 0.7232 | 0.4568 | 0.5872 |

Takeaway: best mean `dAUROC` for this endpoint is `capacity_only` (0.0653).

## Endpoint: `predicted_mask`

| Variant | Type | Loss | Features | dAUROC | dAUPR | dFPR95 | TopK Jaccard | Effect Corr | Score Corr ID | Score Corr OOD |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| jsrt_unet_bce_dice_f24_48_96_192_seed314159 | capacity_only | bce_dice | 24,48,96,192 | -0.0516 | -0.0663 | 0.2470 | 0.4815 | 0.5896 | 0.6064 | 0.5116 |
| jsrt_unet_dice_f24_48_96_192_seed314159 | combined_change | dice | 24,48,96,192 | -0.0547 | -0.0921 | 0.1579 | 0.6000 | 0.7569 | 0.4994 | 0.5569 |
| jsrt_unet_dice_f32_64_128_256_seed314159 | loss_only | dice | 32,64,128,256 | -0.0474 | -0.0654 | 0.2713 | 0.5385 | 0.7759 | 0.5625 | 0.5773 |

Takeaway: best mean `dAUROC` for this endpoint is `loss_only` (-0.0474).

