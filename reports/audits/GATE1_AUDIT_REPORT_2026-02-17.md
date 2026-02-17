# Gate 1 Audit Report (Data Leakage & Protocol Integrity)

Date: 2026-02-17
Decision: **NO-GO**

## Summary Table
| Check | Status | Evidence |
|---|---|---|
| GT leakage in `data/interim/*.npz` | **FAIL** | `data/interim/*/full/*.npz` contains keys `image` and `mask` (observed directly via inspection). |
| GT mask used in fingerprint extraction | **FAIL** | `src/xfp/fingerprint/runner.py:228` loads `mask = data["mask"]`; `src/xfp/fingerprint/runner.py:256` passes it into `_reduce_attribution`; `src/xfp/fingerprint/runner.py:307-315` computes border/coverage/components using `mask`. |
| Train/test overlap control (patient-level) | **FAIL** | No `patient_id` in cache metadata (`data/interim/*/full/metadata.parquet` columns), no `splits.csv` found under `/home/ubuntu/Datasets`; fallback random image-level split in `scripts/train_unet.py:149-152` and `scripts/train_unet_from_cache.py:47-51`. |
| Protocol freeze | **PASS** | Created `configs/protocol_lock_v1.yaml` with fixed seeds, split policy, endpoints, metrics, and thresholds. |

## Detailed Findings

### 1) GT leakage risk is real in current cache design
- Cache writer stores GT masks directly in NPZ: `src/xfp/data/pipeline.py:63-67`.
- Metadata also stores `mask_coverage` derived from GT: `src/xfp/data/pipeline.py:86`.
- Result: unless explicitly blocked, downstream code can consume GT-derived signals in OOD detection.

### 2) Current fingerprint runner is not deployment-legal
- Direct GT mask load: `src/xfp/fingerprint/runner.py:228`.
- Dice computed against GT: `src/xfp/fingerprint/runner.py:243`.
- GT mask used to compute fingerprint features: `src/xfp/fingerprint/runner.py:256`, `src/xfp/fingerprint/runner.py:307-315`.
- Conclusion: current default fingerprints are analysis-oriented upper-bound, not deployment-safe.

### 3) Split integrity is insufficient for publication-grade claims
- Training script can use patient-level splits if `splits.csv` exists: `scripts/train_unet.py:155-162`.
- In absence of split file, it falls back to random image-level split: `scripts/train_unet.py:149-152`.
- Cached training script always random image-level: `scripts/train_unet_from_cache.py:47-51`.
- No patient identifier currently present in interim metadata to verify disjointness.

## Required Actions Before Gate 2
1. Implement endpoint separation in code (`upper_bound_gt`, `predicted_mask`, `mask_free`) and hard block GT features for deployment endpoint.
2. Introduce patient-level split manifests and propagate `patient_id` through metadata for auditability.
3. Add leakage unit tests that fail if GT-only columns enter deployment detector inputs.
4. Re-run pilot only (small subset, 2 seeds) after fixes; no full GPU run until Gate 1 is re-audited as PASS.
