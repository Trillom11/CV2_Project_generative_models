# Pix2Pix: Label Map → Realistic Tissue Image (MoNuSeg)

> **Authors:** Javier Crego Fraguela & Raúl Trillo Martínez 
> **Course:** Computer Vision II — Master in Artificial Intelligence  
> **Task:** Image-to-Image Translation using Conditional GANs (Pix2Pix)  
> **Dataset:** [MoNuSeg] — Multi-organ Nucleus Segmentation

---

## Overview

This project implements a **Pix2Pix** Generative Adversarial Network that translates semantic label maps of cell nuclei into realistic histology tissue images. Two architectures are provided:

- **Baseline** — Standard U-Net Generator + PatchGAN Discriminator
- **Improved** — Attention U-Net Generator + Spectral Normalization Discriminator, tuned with Optuna (includes Dropout regularisation, Early Stopping, and ReduceLROnPlateau scheduling)

---

## Repository Structure

```text
Project/
├── infer.py                  # ← Standalone demo script (Task 3)
├── main.ipynb                # Full training & evaluation notebook
├── README.md                 # This file
│
├── src/                      # Modular source code
│   ├── models.py             # Generator (U-Net / Attention) and Discriminator
│   ├── train.py              # Training loop
│   ├── data.py               # Dataset loading and augmentation
│   ├── experiments.py        # Optuna hyperparameter search
│   ├── utils.py              # Metrics (PSNR, SSIM, MAE) and visualisation
│   └── plots.py              # Model comparison charts
│
├── checkpoints/              # Saved model weights and Optuna cache
│   ├── baseline_G.pth        # Baseline generator weights
│   ├── optuna_study.db       # SQLite database with all Optuna trials
│   ├── optuna_results.pkl    # Legacy cache for Optuna
│   ├── optuna_best_trial_G.pth # Best generator from Optuna (used for warm-starting)
│   └── best_model_G.pth      # Best improved generator weights
│
├── samples/                  # Sample input images for the demo
│   ├── sample_01_monuseg.png
│   ├── sample_02_monuseg.png
│   └── sample_03_monuseg.png
│
└── outputs/                  # Generated images are saved here automatically
```

---

## Requirements

**Python 3.8+** is required. Install all dependencies with:

```bash
pip install torch torchvision Pillow matplotlib numpy scikit-image optuna
```

> **GPU recommended.** The script will automatically use CUDA if available, otherwise it falls back to CPU.

---

## Task 3 — Running the Demo (`infer.py`)

### Step 1 — Download the files

Clone or download this repository so the following exist in the same folder:

```
infer.py
src/
checkpoints/best_model_G.pth   ← trained weights
samples/                       ← at least one input image
```

### Step 2 — Run the inference script

```bash
python3 infer.py <path_to_image>
```

**That's it.** The script:
1. Loads the image from `<path_to_image>`.
2. If the image is in MoNuSeg paired format (width ≈ 2× height), it **automatically crops the right half** (the label map).
3. Runs the generator and saves the result to `outputs/<filename>_generated.png`.
4. Opens a **matplotlib window** showing: `Label Map | Generated Image | Ground Truth`.

### Examples

```bash
# Run on a provided sample
python3 infer.py samples/sample_01_monuseg.png

# Run on your own label map
python3 infer.py /path/to/my_label_map.png

# Use the baseline model instead
python3 infer.py samples/sample_01_monuseg.png --model baseline

# Use a custom weights file
python3 infer.py samples/sample_01_monuseg.png --weights checkpoints/baseline_G.pth
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `input` | positional | — | Path to the input image (label map or paired image) |
| `--weights` | optional | `checkpoints/best_model_G.pth` | Path to the generator `.pth` weights file |
| `--model` | optional | `attention` | Architecture: `attention` or `baseline` |

---

## Training Your Own Model

Open `main.ipynb` in Jupyter and run the cells from top to bottom. The notebook covers:

1. **Phase 1** — Data loading and EDA
2. **Phase 2** — Baseline training (U-Net + PatchGAN)
3. **Phase 3** — Quantitative evaluation (PSNR, SSIM, MAE) and loss curves
4. **Phase 4** — Improvements: Attention, Spectral Normalization, Optuna search (with SQLite persistence and Trial Analysis), Early Stopping, and LR Scheduling.
5. **Phase 5** — Demo: calling `infer.py` on sample images
