# Vintage Image Restoration & Colorization using U-Net

## Overview

This repository hosts the implementation of a Deep Learning system aimed at restoring and colorizing **vintage photographs**. Unlike standard colorization tasks that simply predict colors for grayscale images, this project focuses on a **full restoration pipeline**:

* Reversing **Sepia tones**
* Removing **noise / film grain**
* Restoring **natural colors**

The project is organized into **two complementary phases** to balance scientific validation and high-quality visual performance.

---

## Project Phases

### Phase 1 – Simulation & Scalability Analysis

A lightweight U-Net trained on **synthetic data** to:

* Validate the training pipeline
* Analyze scalability
* Study the impact of class diversity on **SSIM** and **L1 Loss**

Experiments are conducted with increasing numbers of object classes:
**1, 10, 100, and 1000 classes**.

### Phase 2 – High-Quality Restoration

An enhanced U-Net architecture trained on the **STL-10 dataset** for visually realistic restoration.

Key upgrades:

* Batch Normalization
* LeakyReLU activations
* Deeper bottleneck (512 channels)

---

## Key Features

* **Dual-Phase Design** – separates theoretical analysis from production-quality training
* **Vintage Degradation Pipeline** (applied on-the-fly):

  * Sepia transformation via matrix multiplication
  * Gaussian noise to simulate film grain
* **Enhanced U-Net Architecture**:

  * Skip connections for spatial detail preservation
  * Deep semantic bottleneck
  * Batch Normalization to avoid color desaturation
* **Automated Dataset Handling**:

  * Phase 1: synthetic data generation
  * Phase 2: automatic STL-10 download via Torchvision

---

## Requirements

Install dependencies using:

```bash
pip install torch numpy matplotlib torchvision
```

---

## Project Structure

```text
│
├── train_phase1_simulation.py   # Phase 1: scalability & synthetic experiments
├── train_phase2_stl10.py        # Phase 2: high-quality restoration (STL-10)
├── Report.pdf                   # Detailed technical report
└── README.md                    # Project documentation
```

---

## How to Run

### Phase 1: Simulation & Scalability Test

This script evaluates how model performance changes with increasing class diversity.

**Command:**

```bash
python train_phase1_simulation.py
```

**Outputs:**

* Training logs for each class configuration
* Graphs showing **L1 Loss** and **SSIM** over epochs
* Generated folders:

  * `/Outputs/Color/` – model predictions
  * `/Outputs/Gray/` – degraded inputs
  * `/Images/train` and `/Images/val`
  * `/Models/` (if `save_model=True`)

---

### Phase 2: High-Quality Restoration

Trains the enhanced U-Net on the STL-10 dataset (96×96 resolution).

**Command:**

```bash
python train_phase2_stl10.py
```

**Outputs:**

* Automatic download of STL-10 into `./data`
* Training of the restoration network
* Visual comparisons:

  * Vintage input
  * Model output
  * Ground truth image

---

## Technical Details

### Degradation Model

The network learns a supervised mapping:

[ I \approx f_\theta(I_{vintage}) ]

where the degraded input is generated as:

[ I_{vintage} = \text{Noise}(\text{Sepia}(I_{clean})) ]

This allows full control over degradation severity and ensures pixel-wise supervision.

---

### Architecture

A classic **U-Net encoder–decoder** structure is used.

**Phase 2 enhancements include:**

* **LeakyReLU (α = 0.2)** to avoid dying neurons
* **Batch Normalization** for stable training and vivid color reconstruction
* **512-channel bottleneck** for rich semantic context

---

## Results

### Phase 1 – Quantitative Analysis

Scalability experiments highlight the trade-off between dataset diversity and convergence speed.

*(Insert SSIM & L1 loss plots here)*

### Phase 2 – Qualitative Results

The model successfully restores natural colors from heavily degraded sepia images.

*(Insert before / after restoration examples here – birds, ships, portraits, etc.)*

---

## Authors

* **El Ouardi Aymane**
* **Majid Bonyadi**

**University of Strasbourg**

---

## Acknowledgments

This project was developed for academic research purposes in image restoration and deep learning, with a focus on explainability, scalability, and visual fidelity.
<img width="1141" height="790" alt="download (5)" src="https://github.com/user-attachments/assets/56ed6a3b-1290-48ae-9254-c013dee4782b" />
<img width="1390" height="590" alt="download (4)" src="https://github.com/user-attachments/assets/34f7403a-5ea2-4013-9267-f6510afaf9ae" />
