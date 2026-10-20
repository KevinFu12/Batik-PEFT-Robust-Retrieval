# Batik PEFT: Frequency Robustness in Motif Retrieval

A research benchmark investigating why adapting foundation vision encoders to fine-grained Indonesian Batik motifs degrades their robustness against real-world frequency corruptions, and how to fix it.

---

## What is this project?

Most Batik recognition benchmarks report over 95% accuracy because rotated copies and cropped regions of the same physical cloth leak across training and test splits. When these near-duplicates are excluded:
- **Clean retrieval is solved:** Frozen foundation models (`SigLIP`, `DINOv2`) achieve a perfect Mean Reciprocal Rank (MRR = 1.000). Clean accuracy can no longer differentiate models.
- **Fine-tuning causes fragility:** Naive parameter-efficient adaptation (LoRA, DoRA, KronA) overfits to ultra-high spatial frequencies (dot spacing, line density). When query images undergo realistic corruptions (blur, down-sampling, compression, low contrast), robust MRR collapses by **0.15 to 0.28**.
- **The remedies:** Adding **Gaussian blur augmentation** prevents high-frequency over-reliance (boosting robust MRR up to 0.890 on DINOv2), while **Fourier amplitude augmentation combined with weight blending (WiSE-FT)** provides the strongest out-of-distribution generalization against unseen noise and contrast degradation.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/vytora04/batik-peft-robust-retrieval.git
cd batik-peft-robust-retrieval

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Setup

Place the four collections into the `data/` directory:

| Dataset | Split Role | Source |
| :--- | :--- | :--- |
| **Batik Nitik 960** | 5-Fold Evaluation (59 classes) | [Mendeley Data](https://doi.org/10.17632/sgh484jxzy.3) |
| **Batik Lasem** | Auxiliary Training (5 classes) | [Mendeley Data](https://doi.org/10.17632/kw5b7d9zgn.5) |
| **Batik Keraton** | Auxiliary Training (4 classes) | [Kaggle](https://www.kaggle.com/datasets/stefaron/dataset-batik-keraton) |
| **Batik Yogyakarta** | Auxiliary Training (3 classes) | [Kaggle](https://www.kaggle.com/datasets/edmundusgigih/batik-keraton-yogyakarta) |

Expected directory layout:
```text
data/
├── BatikNitik960/
├── BatikKeraton/
├── BatikYogyakarta/
└── BatikLasem/
```

### 3. Run Everything

To run the entire pipeline end-to-end (deduplication, calibration, training, evaluation, blending, and export):

```bash
python run_research.py
```

---

## Step-by-Step CLI Usage

You can execute any individual stage from `pipeline/steps/`:

```bash
# 1. Deduplicate images (pHash <= 10) and generate 5-fold class-disjoint splits
python -m pipeline.steps.setup

# 2. Calibrate query degradation parameters (LP, Downsampling, JPEG, Noise)
python -m pipeline.steps.controls

# 3. Train models with Supervised Contrastive Loss
# Arguments:
#   --encoder:      [siglip, dinov2, clip]
#   --method:       [lora, dora, krona, full]
#   --augmentation: [none, blur, fourier]
python -m pipeline.steps.train --encoder siglip --method lora --augmentation blur
python -m pipeline.steps.train --encoder dinov2 --method lora --augmentation fourier

# 4. Evaluate degraded queries vs. clean gallery (MRR across 5 folds)
python -m pipeline.steps.evaluate --encoder siglip --method lora --fold all

# 5. Weight-space blending (WiSE-FT) and spectral band shift analysis
python -m pipeline.steps.generalize --action blend
python -m pipeline.steps.generalize --action spectral_sensitivity

# 6. Export statistical tables (bootstrap 95% CI, Holm correction) & figures
python -m pipeline.steps.export
```

---

## Repository Structure

```text
.
├── pipeline/
│   └── steps/
│       ├── setup.py        # Deduplication (pHash), source-group definitions, fold splits
│       ├── controls.py     # Calibration & query degradation pipelines (LP, down-sampling, JPEG)
│       ├── train.py        # Contrastive training loop (LoRA, DoRA, KronA, Full FT)
│       ├── evaluate.py     # Source-exclusive gallery ranking and reciprocal rank (RR/MRR)
│       ├── generalize.py   # WiSE-FT weight interpolation & 5-band Fourier spectral sensitivity
│       └── export.py       # Bootstrap statistics (10k resamples) and result summaries
├── requirements.txt        # PyTorch, Torchvision, Transformers, PEFT, Scipy
├── run_research.py         # Top-level orchestration script
└── README.md
```

---

## Summary of Results

All models achieve **MRR = 1.000** on clean images once same-cloth leakage is removed. The real divergence appears under frequency degradation:

| Encoder | Adaptation | Robust MRR (Frequency Loss) | Held-out MRR (Noise & Contrast) |
| :--- | :--- | :---: | :---: |
| **SigLIP** | Frozen Baseline | 0.545 | 0.660 |
| | Naive LoRA | 0.329 *(collapses)* | 0.512 |
| | Naive DoRA | 0.306 *(collapses)* | 0.521 |
| | Naive KronA | 0.395 | 0.638 |
| | Full Fine-Tuning | 0.262 | — |
| | LoRA + **Blur Augmentation** | **0.825** | 0.659 |
| | LoRA + **Fourier + Blending** | 0.580 | **0.739** |
| **DINOv2** | Frozen Baseline | 0.571 | 0.741 |
| | Naive LoRA | 0.402 | 0.722 |
| | LoRA + **Blur Augmentation** | **0.890** | 0.876 |
| | LoRA + **Fourier + Blending** | 0.759 | **0.896** |
| **CLIP** | Frozen Baseline | 0.307 | 0.433 |
| | LoRA + **Blur Augmentation** | **0.812** | 0.775 |

---

## License

This project is licensed under the [MIT License](LICENSE).
