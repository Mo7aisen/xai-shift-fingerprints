# Attribution Fingerprinting for Dataset Shift Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-red.svg)](https://pytorch.org/)

Official implementation of **"Attribution Fingerprinting for Dataset Shift Detection in Medical Image Segmentation"**

> **Abstract**: Deep learning models for medical image analysis can fail silently when deployed data differs from training distributions. We introduce attribution fingerprinting—a framework for detecting dataset shift through systematic analysis of explanation patterns. Applied to lung segmentation models on chest X-ray datasets (JSRT, Montgomery, NIH ChestX-ray14), our method reveals that models exhibit dramatic attribution instability under distribution shift, with changes invisible to accuracy monitoring alone. We demonstrate that a 121-feature fingerprint achieves 94% accuracy in distinguishing in-distribution from shifted samples, validated on 112,120 NIH ChestX-ray14 images.

---

## Current Status (2026-02-05)

- Canonical manuscript source: `manuscript/`
- Active submission package: `submission_medical_image_analysis/`
- Legacy submission snapshots archived at:
  - `archive/old_submissions_2026-02-05/final_elsevier_submission_2026-01-21.zip`
  - `archive/old_submissions_2026-02-05/journal_submission_bundle_2025-12-23.zip`

### Recently Completed

- ResNet OOD baseline completed on **2026-01-30** (`results/baselines/resnet_ood_auc.csv`)
- Energy/MSP OOD baseline completed on **2026-02-04** (`results/baselines/energy_ood_auc.csv`)
- Canonical manuscript and submission package synchronized and LaTeX-validated on **2026-02-05**

### Pending (Blocked by GPU Occupancy)

- `python scripts/run_rise_segmentation.py --max-samples 200 --n-masks 1000 --mask-batch 32`
- `python scripts/generate_publication_figures_final.py` (regeneration after RISE outputs)

GPU is currently occupied by a separate running project (`/home/ubuntu/UARF-AQA`), so pending GPU steps should be resumed only after that workload finishes.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Reproducing Results](#-reproducing-results)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Key Features

- **Attribution Fingerprinting**: Extract 121 features from Integrated Gradients and Grad-CAM attribution maps
- **Multi-Scale Shift Detection**: Quantify distribution changes using KL divergence, Earth Mover's Distance, and Graph Edit Distance
- **Large-Scale Validation**: Validated on NIH ChestX-ray14 (112,120 samples) with lesion-size stratification
- **Automated QA Framework**: Quality assurance harness for lung mask validation with remediation pipeline
- **Deployment Monitoring**: Production-ready monitoring with tmux/cron integration and configurable alerts
- **Reproducible Pipeline**: Complete workflow from data preparation to publication-ready figures

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU acceleration)
- Poetry for dependency management

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/<org-or-user>/xai-shift-fingerprints.git
cd xai-shift-fingerprints

# Create conda environment
conda env create -f environment.yml
conda activate xai-shift

# Install dependencies with Poetry
poetry install

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Legacy Model Configuration

If using pretrained UNet models from legacy repositories, set the environment variable:

```bash
# Configure legacy model source paths (colon-separated)
export XFP_LEGACY_MODEL_PATHS="/path/to/scba/src:/path/to/legacy/models"
```

The package will automatically search these directories for UNet model definitions.

---

## 🔄 Reproducibility

All experiments use fixed random seeds for reproducibility:

| Component | Seed | Purpose |
|-----------|------|---------|
| Bootstrap CI | 2025 | Confidence interval estimation |
| Train/test splits | 42 | Dataset partitioning |
| Permutation tests | 2025 | Statistical significance |

**Metrics Traceability**: All reported metrics are logged in `manuscript/claims/metrics.json` with source files and reproduction commands.

**Code Availability Statement**: This code is available under MIT license. All experiments can be reproduced using the scripts and configurations provided. The repository URL and archived release DOI will be provided upon acceptance.

---

## ⚡ Quick Start

### 1. Configure Paths

Edit `configs/paths.yaml` to point to your local dataset and model directories:

```yaml
datasets:
  jsrt: /path/to/jsrt
  montgomery: /path/to/montgomery
  nih: /path/to/nih_chestxray14

models:
  jsrt_unet: /path/to/jsrt_model.pth
  montgomery_unet: /path/to/montgomery_model.pth
```

### 2. Run Pilot Experiment

Test the pipeline on a small subset (5 JSRT samples):

```bash
# Prepare data cache
python scripts/prepare_data.py --dataset jsrt --subset pilot5

# Generate attribution fingerprints
python scripts/run_fingerprint.py \
    --experiment jsrt_baseline \
    --subset pilot5 \
    --device cuda

# Compute divergence metrics
python scripts/compute_divergence.py \
    --reference data/fingerprints/jsrt_baseline/jsrt.parquet \
    --target data/fingerprints/jsrt_to_montgomery/montgomery.parquet
```

### 3. View Results

Results are saved in `reports/divergence/`:
- `divergence_summary.md`: Statistical summary with confidence intervals
- `divergence_*.png`: Visualization plots (PCA, violin plots, distribution comparisons)

---

## 📊 Dataset Preparation

### Supported Datasets

| Dataset | Samples | Source | Notes |
|---------|---------|--------|-------|
| **JSRT** | 247 | [Link](http://db.jsrt.or.jp/eng.php) | PA chest X-rays with manual lung masks |
| **Montgomery County** | 138 | [Link](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html) | PA/AP chest X-rays, TB screening |
| **NIH ChestX-ray14** | 112,120 | [Link](https://nihcc.app.box.com/v/ChestXray-NIHCC) | Weakly-supervised with bounding boxes |

### Download & Preprocessing

```bash
# JSRT and Montgomery: Download manually and place in data/raw/

# NIH ChestX-ray14: Check archive integrity
python scripts/check_nih_archives.py

# Extract and prepare manifests
python scripts/prepare_nih_chestxray14.py

# Generate lung masks
python scripts/generate_nih_masks.py --skip-existing

# Run quality assurance
python scripts/run_nih_mask_qc.py --samples 300 --seed 2025
```

**Metadata Overrides** (optional): Populate `data/metadata/jsrt_metadata.csv` or `montgomery_metadata.csv` with pixel spacing, projection type, patient age, and acquisition site for stratified analyses.

---

## 🔬 Reproducing Results

### Complete Experimental Pipeline

Run all four scenarios from the paper (estimated time: ~4 hours on single GPU):

```bash
# Set CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run fingerprinting batch (JSRT baseline, Montgomery baseline, cross-dataset)
bash scripts/run_gpu_fingerprint_batch.sh

# Generate all analyses (divergence, statistics, bbox-stratified, figures)
bash run_all_analyses.sh
```

### Deployment Monitoring Simulation

Simulate weekly deployment monitoring over 12 weeks:

```bash
# Configure monitoring schedule
python scripts/run_deployment_monitor.py \
    --baseline-fingerprints data/fingerprints/jsrt_baseline/jsrt.parquet \
    --deployment-batches data/fingerprints/nih_weekly/ \
    --alert-threshold 0.95 \
    --output reports/deployment/

# Visualize alerts and drift trends
python scripts/plot_deployment_timeline.py \
    --results reports/deployment/monitoring_summary.csv
```

### Statistical Validation

```bash
# Bootstrap confidence intervals (500 iterations)
python scripts/bootstrap_divergence.py \
    --reference data/fingerprints/jsrt_baseline/jsrt.parquet \
    --target data/fingerprints/jsrt_to_montgomery/montgomery.parquet \
    --n-iterations 500

# Permutation tests (1000 permutations)
python scripts/permutation_test.py \
    --fingerprints data/fingerprints/jsrt_baseline/jsrt.parquet \
                   data/fingerprints/jsrt_to_montgomery/montgomery.parquet \
    --n-permutations 1000
```

---

## 📁 Project Structure

```
xai_shift_fingerprints/
├── configs/              # Experiment configurations and paths
│   ├── paths.yaml
│   ├── experiments.yaml
│   └── subsets/
├── data/                 # Data caches (not tracked in git)
│   ├── fingerprints/     # Parquet files with 121-feature fingerprints
│   ├── interim/          # Preprocessed images and masks
│   └── metadata/         # Acquisition metadata (spacing, projection, age)
├── docs/                 # Architecture notes and methodology
├── manuscript/           # LaTeX manuscript and figures
│   ├── main.tex          # Main manuscript file
│   ├── references.bib    # Bibliography
│   ├── figures/          # Publication-quality PDF figures
│   └── tables/           # LaTeX tables
├── notebooks/            # Jupyter notebooks for exploration
├── reports/              # Generated analysis reports (Markdown + plots)
│   ├── divergence/
│   ├── deployment/
│   ├── error_correlation/
│   └── external_validation/
├── results/              # Experimental outputs
│   ├── figures/          # High-res PNG/PDF figures
│   ├── metrics/          # CSV/JSON metric dumps
│   └── tables/           # LaTeX/CSV tables
├── scripts/              # CLI entrypoints for pipeline stages
│   ├── prepare_data.py
│   ├── run_fingerprint.py
│   ├── compute_divergence.py
│   └── run_all_analyses.sh
├── src/xfp/              # Core Python package
│   ├── fingerprints/     # Feature extraction (coverage, border, topology)
│   ├── divergence/       # KL, EMD, GED computation
│   ├── preprocessing/    # Image normalization and mask QA
│   └── utils/            # Logging, plotting, I/O
├── tests/                # Pytest suite
├── environment.yml       # Conda environment specification
├── pyproject.toml        # Poetry dependencies
└── README.md
```

---

## 📖 Attribution Methods

We support multiple attribution backends (configurable via `configs/experiments.yaml`):

- **Integrated Gradients** (default): Axiomatic, path-based attribution with 50 interpolation steps
- **Grad-CAM**: Class activation mapping via gradient backpropagation
- **SmoothGrad**: Variance reduction via noisy sample averaging (coming soon)

**Fingerprint Features** (121 total):
1. **Spatial Statistics** (28): coverage, border ratio, centroid displacement, eccentricity
2. **Distribution Moments** (24): mean, std, skewness, kurtosis, quantiles, entropy
3. **Topological** (16): connected components, Euler characteristic, largest component ratio
4. **Frequency Domain** (18): FFT power spectrum, spectral centroid/spread
5. **Prediction Alignment** (20): Dice overlap, IoU, precision/recall vs mask
6. **Coverage Curves** (15): AUC, coverage at top-k thresholds

---

## 🧪 Quality Assurance

### Automated Mask QA

```bash
python scripts/run_nih_mask_qc.py \
    --samples 300 \
    --stratify-by finding sex age \
    --output reports/external_validation/nih_mask_qc.csv
```

**QA Criteria**:
- Coverage: `0.45 ≤ coverage ≤ 0.88`
- Connected components: `≤ 6`
- Dominant component ratio: `≥ 0.4`
- IoU with bounding box: `≥ 0.5` (when available)

**Automated Remediation**: Flagged masks are re-segmented with adaptive sigmoid thresholds (`τ_high = 0.90`) and morphological cleanup until passing all criteria.

---

## 📈 Deployment Monitoring

### Production Integration

```bash
# Launch monitoring in tmux session
tmux new -s xai_shift_gpu
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/run_gpu_fingerprint_batch.sh

# Configure cron for automated audits
# Add to crontab:
# 0 2 * * 1 /path/to/run_nih_mask_qc.sh >> /var/log/xai_qa.log 2>&1
# 45 2 * * * /path/to/run_deployment_monitor.sh
```

**Alert Thresholds** (configurable):
- KL divergence > 0.05 (95% CI from bootstrap)
- EMD > 0.35
- GED > 400
- Welch's t-test p < 0.01 for lesion-size strata

---

## 🎓 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{mohaisen2026attribution,
  title={Attribution Fingerprinting for Dataset Shift Detection in Medical Image Segmentation},
  author={Mohammed Mohaisen and Gabor Hullam},
  journal={Medical Image Analysis},
  year={2026},
  note={Under review}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets**: JSRT (Japanese Society of Radiological Technology), Montgomery County (US National Library of Medicine), NIH ChestX-ray14 (NIH Clinical Center)
- **Funding**: [Add funding agencies or state "None"]
- **Compute**: [Add computing resources acknowledgment]

---

## 📧 Contact

For questions or collaboration inquiries:
- **Email**: mohammed.mohaisen@edu.bme.hu
- **Issues**: [GitHub Issues](https://github.com/<org-or-user>/xai-shift-fingerprints/issues)

---

## 🔗 Related Work

- [Integrated Gradients](https://arxiv.org/abs/1703.01365) (Sundararajan et al., 2017)
- [Grad-CAM](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2017)
- [Dataset Shift in ML](https://mitpress.mit.edu/books/dataset-shift-machine-learning) (Quiñonero-Candela et al., 2009)
- [Clinician and Dataset Shift](https://www.nejm.org/doi/full/10.1056/NEJMc2104626) (Finlayson et al., 2021)
