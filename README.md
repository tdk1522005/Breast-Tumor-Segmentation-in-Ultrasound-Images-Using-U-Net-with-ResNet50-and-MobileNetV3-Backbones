# Breast Ultrasound Tumor Segmentation

A deep learning project for **semantic segmentation of breast tumors in ultrasound images**, comparing a baseline U-Net with U-Net variants that use **ResNet50** and **MobileNetV3Large** encoders.

> **Medical disclaimer:** This project is for educational and research purposes only. It is not a clinical diagnostic system and must not be used for medical decision-making.

## Overview

The project uses the **Breast Ultrasound Images Dataset (BUSI)** from Kaggle. The current notebook preprocesses benign and malignant ultrasound images and their ground-truth masks, then trains and evaluates three segmentation architectures.

### Compared Models

- U-Net
- ResNet50-U-Net
- MobileNetV3-U-Net

## Deep Learning Workflow

```text
BUSI ultrasound images + ground-truth masks
                ↓
Image / mask preprocessing
                ↓
Train / validation / test split
                ↓
U-Net / ResNet50-U-Net / MobileNetV3-U-Net
                ↓
Training with BCE + Dice loss
                ↓
Evaluation on test set
                ↓
Loss · Dice Coefficient · IoU
                ↓
Architecture comparison
```

## Dataset

The notebook uses the Kaggle BUSI dataset path:

```text
Dataset_BUSI_with_GT
```

The current preprocessing pipeline uses the `benign` and `malignant` classes and pairs each ultrasound image with its segmentation mask.

Data splitting in the notebook:

- Test split: 15% of the complete processed set
- Validation split: 15% of the remaining training portion
- Random seed: 42

Input sizes used in the notebook:

- U-Net / ResNet50-U-Net: 256 × 256
- MobileNetV3-U-Net: 224 × 224
- Batch size: 8

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Backbones:** ResNet50, MobileNetV3Large
- **Image Processing:** OpenCV
- **Data / Numerical Computing:** NumPy
- **Visualization:** Matplotlib
- **Data Splitting:** Scikit-learn
- **Environment:** Kaggle / Jupyter Notebook

## Loss and Metrics

Training uses a combined **Binary Cross-Entropy + Dice Loss**:

```text
Loss = 0.3 × BCE + 0.7 × Dice Loss
```

Models are compared using:

- Loss
- Dice Coefficient
- Intersection over Union (IoU)

## Test Results

| Model | Loss | Dice | IoU |
|---|---:|---:|---:|
| U-Net | 0.4602 | 0.6798 | 0.5215 |
| **ResNet50-U-Net** | **0.2736** | **0.6910** | **0.5383** |
| MobileNetV3-U-Net | 0.4137 | 0.4886 | 0.3264 |

**ResNet50-U-Net achieved the best test performance** among the three evaluated models, with the highest Dice coefficient and IoU and the lowest test loss.

## Repository Structure

```text
Breast-Tumor-Segmentation-in-Ultrasound-Images-Using-U-Net-with-ResNet50-and-MobileNetV3-Backbones/
├── nhandangdoituong-taduykhanh-2311554980.ipynb  # End-to-end experiment notebook
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tdk1522005/Breast-Tumor-Segmentation-in-Ultrasound-Images-Using-U-Net-with-ResNet50-and-MobileNetV3-Backbones.git
cd Breast-Tumor-Segmentation-in-Ultrasound-Images-Using-U-Net-with-ResNet50-and-MobileNetV3-Backbones
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the BUSI dataset from Kaggle and update the dataset path in the notebook if you are running outside Kaggle.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `nhandangdoituong-taduykhanh-2311554980.ipynb` and execute the cells in order.

## What This Project Demonstrates

- Medical image preprocessing and mask preparation
- Semantic segmentation with U-Net
- Transfer learning / pretrained encoder backbones
- ResNet50 and MobileNetV3 encoder comparison
- Custom Dice and IoU metrics
- Combined BCE + Dice loss
- Training-history and test-result visualization
- Quantitative comparison across segmentation architectures

## Current Limitations

- The project is currently organized primarily as a single experimental notebook.
- Results are based on the current dataset split and training configuration.
- The current experiment processes benign and malignant images; broader validation is needed before making conclusions about generalization.
- No external clinical validation is included.

## Planned Improvements

- Refactor reusable model, preprocessing and evaluation code into a `src/` package.
- Add qualitative prediction examples: input image, ground-truth mask and predicted mask.
- Add automated experiment configuration and reproducibility controls.
- Add cross-validation or repeated experiments to measure result stability.
- Compare computational cost and inference speed across backbones.

## Author

**Ta Duy Khanh**  
AI Engineering student — Machine Learning, Deep Learning & Natural Language Processing
