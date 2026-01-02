# Vintage Image Restoration & Colorization using U-Net

## Overview
This repository hosts the implementation of a Deep Learning system aimed at restoring and colorizing "vintage" photographs. Unlike standard colorization tasks that simply predict colors for grayscale images, this project focuses on a comprehensive restoration pipeline: **reversing Sepia tones, removing noise, and restoring original colors.**

The project is structured into two distinct execution phases to balance theoretical validation with practical application:

1.  **Phase 1 (Simulation & Scalability):** A lightweight U-Net trained on synthetic data to validate training logic and analyze the impact of class diversity on metrics (SSIM & L1 Loss).
2.  **Phase 2 (High-Quality Restoration):** An enhanced U-Net architecture (incorporating Batch Normalization & LeakyReLU) trained on the **STL-10** dataset for high-fidelity visual results.

## Key Features
- **Dual-Phase Approach:** Separate scripts for algorithmic verification (Phase 1) and production-quality training (Phase 2).
- **Vintage Degradation Pipeline:** Simulates historical image artifacts on-the-fly using:
    - **Sepia Filter:** Applied via matrix multiplication.
    - **Gaussian Noise:** Simulates film grain.
- **Enhanced U-Net Architecture:**
    - Skip connections for spatial detail preservation.
    - Expanded bottleneck (512 channels) for semantic context.
    - Optimized with Batch Normalization to prevent color desaturation.
- **Automated Dataset Handling:** No manual downloads required. Phase 1 uses synthetic generation, and Phase 2 automatically downloads STL-10 via Torchvision.

## Requirements
To install the necessary dependencies, run the following command:

```bash
pip install torch numpy matplotlib torchvision
## How to Train the Model

1. Clone this repository using `git clone https://github.com/williamcfrancis/CNN-Image-Colorization-Pytorch.git`

Download the dataset zip file from https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view?usp=sharing and extract it outside the current directory. The directory structure should look like:

```
│
├── train_phase1_simulation.py   # Script for scalability analysis (Synthetic Data)
├── train_phase2_stl10.py        # Script for high-quality restoration (STL-10)
├── Report.pdf                   # Detailed technical report
└── README.md                    # Project documentation
```

2. Run train.py with the following arguments:

1. Phase 1: Simulation & Scalability Test
This script runs a comparative experiment to analyze how the model behaves when the number of object classes increases (1, 10, 100, 1000). It uses a lightweight U-Net and synthetic data.
python train_phase1_simulation.py
The training creates a `/Outputs/` folder with subfolders `/Color/` and `/Gray/`. Validation results are saved in `/Color/` and inputs in `/Gray/`. The training also creates an `/Images/` folder with `train/val` images separated into different folders. If `save_model` is enabled, the final model is saved in a `/Models/` folder as a .pth file.

بله، بر اساس کدهای شما (که شامل دو بخش شبیه‌سازی و نسخه باکیفیت است) و گزارش فنی‌تان، این فایل README.md را به زبان انگلیسی آماده کردم.

این فایل دقیقاً توضیح می‌دهد که پروژه شما شامل دو فاز است (Phase 1 & Phase 2) و چگونه باید هر کدام را اجرا کرد.

Markdown

# Vintage Image Restoration & Colorization using U-Net

## Overview
This repository hosts the implementation of a Deep Learning system aimed at restoring and colorizing "vintage" photographs. Unlike standard colorization tasks that simply predict colors for grayscale images, this project focuses on a comprehensive restoration pipeline: **reversing Sepia tones, removing noise, and restoring original colors.**

The project is structured into two distinct execution phases to balance theoretical validation with practical application:

1.  **Phase 1 (Simulation & Scalability):** A lightweight U-Net trained on synthetic data to validate training logic and analyze the impact of class diversity on metrics (SSIM & L1 Loss).
2.  **Phase 2 (High-Quality Restoration):** An enhanced U-Net architecture (incorporating Batch Normalization & LeakyReLU) trained on the **STL-10** dataset for high-fidelity visual results.

## Key Features
- **Dual-Phase Approach:** Separate scripts for algorithmic verification (Phase 1) and production-quality training (Phase 2).
- **Vintage Degradation Pipeline:** Simulates historical image artifacts on-the-fly using:
    - **Sepia Filter:** Applied via matrix multiplication.
    - **Gaussian Noise:** Simulates film grain.
- **Enhanced U-Net Architecture:**
    - Skip connections for spatial detail preservation.
    - Expanded bottleneck (512 channels) for semantic context.
    - Optimized with Batch Normalization to prevent color desaturation.
- **Automated Dataset Handling:** No manual downloads required. Phase 1 uses synthetic generation, and Phase 2 automatically downloads STL-10 via Torchvision.

## Requirements
To install the necessary dependencies, run the following command:

```bash
pip install torch numpy matplotlib torchvision
Project Structure
│
├── train_phase1_simulation.py   # Script for scalability analysis (Synthetic Data)
├── train_phase2_stl10.py        # Script for high-quality restoration (STL-10)
├── Report.pdf                   # Detailed technical report
└── README.md                    # Project documentation
How to Run
1. Phase 1: Simulation & Scalability Test
This script runs a comparative experiment to analyze how the model behaves when the number of object classes increases (1, 10, 100, 1000). It uses a lightweight U-Net and synthetic data.

Command:

Bash

python train_phase1_simulation.py
Output:

Training logs for different class subsets.

Comparative Graphs: Displays the evolution of L1 Loss and SSIM over epochs.

2. Phase 2: High-Quality Restoration
This script trains the Enhanced High-Quality U-Net on the STL-10 dataset (96x96 resolution). It focuses on visual performance.

Command:
python train_phase2_stl10.py

Output:

Automatically downloads the STL-10 dataset to ./data.

Trains the model to remove sepia and noise.

Visual Results: Displays a side-by-side comparison of Input (Vintage) vs. AI Output vs. Ground Truth.


Technical DetailsThe Degradation ModelThe model learns a supervised mapping $I \approx f_\theta(I_{vintage})$ where the input is generated mathematically:$$ I_{vintage} = \text{Noise}(\text{Sepia}(I_{clean}))  <img width="1141" height="790" alt="download (5)" src="https://github.com/user-attachments/assets/c5a8b50f-ae47-4eaa-8601-dbea59db77f0" />
<img width="1390" height="590" alt="download (4)" src="https://github.com/user-attachments/assets/f2417d57-c256-4d9d-8290-22408de5135a" />
<img width="1141" height="790" alt="download (5)" src="https://github.com/user-attachments/assets/d42d755d-de4f-482e-a284-f8e6249bfe7a" />
<img width="1390" height="590" alt="download (4)" src="https://github.com/user-attachments/assets/f49b17ac-8627-4555-9c54-634865e84a55" />
$$ArchitectureWe utilize a U-Net architecture with an Encoder-Decoder structure.
 The Phase 2 model is upgraded with:LeakyReLU (0.2): To prevent dying gradients.
Batch Normalization: To stabilize training and ensure vibrant color prediction.
ResultsPhase 1 (Quantitative)Graphs generated by Phase 1 demonstrate the trade-off between dataset diversity and convergence speed.
Phase 2 (Qualitative)The model successfully restores natural colors from heavily degraded sepia inputs.
(You can add your screenshots here, e.g., the bird or ship images)AuthorsEl ouardi AymaneMajid BonyadiUniversity of Strasbourg



## Results
![image](https://user-images.githubusercontent.com/38180831/215289552-d3fd414a-84d9-4eda-9ead-b70abb5e59c5.png)

### Demonstrating Color Temperature Control 
![image](https://user-images.githubusercontent.com/38180831/215289605-c464a3bd-d50a-4a19-9aed-90f9c624e035.png)

