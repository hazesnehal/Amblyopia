# Amblyopia
# 🧠 Amblyopia Detection Using fMRI Functional Connectivity

> A machine learning pipeline to classify **Amblyopia (lazy eye)** vs **Normal** subjects using resting-state fMRI brain connectivity data — powered by SVM, CNN, and ResNet with majority-vote ensemble prediction.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Known Issues & Fixes](#known-issues--fixes)
- [Future Work](#future-work)
- [Dependencies](#dependencies)

---

## 🔍 Overview

This project uses **resting-state fMRI (rs-fMRI)** data to detect amblyopia by analyzing **functional brain connectivity** patterns. The pipeline extracts ROI-level time series using the Harvard-Oxford cortical atlas, computes correlation-based connectivity matrices, and trains three classifiers — SVM, CNN, and ResNet — with a final majority-vote ensemble decision.

**Task:** Binary classification — `Normal (0)` vs `Amblyopia (1)`

**Input:** 4D fMRI NIfTI files (`restingEyeOpen.nii.gz`)

**Output:** Predicted class + confidence score per subject

---

## 🗂️ Dataset Structure

```
MyDrive/
└── Dataset1/
    ├── Normal/
    │   ├── sbj01/
    │   │   ├── anat.nii.gz
    │   │   ├── pre/
    │   │   │   └── restingEyeOpen.nii.gz
    │   │   └── post1/
    │   └── sbj02/ ...
    └── Amblyopia/
        ├── sbj01/
        │   ├── anat.nii.gz
        │   ├── pre/
        │   │   └── restingEyeOpen.nii.gz
        │   └── post1/
        └── sbj03/ ...
```

- **Total subjects:** 12 (6 Normal, 6 Amblyopia)
- **fMRI shape per subject:** `(64, 64, 32, 300)` — X × Y × Z × Time
- **Session used:** `pre` (pre-treatment resting state, eyes open)

---

## ⚙️ Pipeline

```
Raw fMRI (.nii.gz)
        │
        ▼
Step 1 — Load fMRI with nibabel
        │
        ▼
Step 2 — Preprocessing
        │  Reshape 4D → 2D (voxels × time)
        │  Signal cleaning + z-score standardization (nilearn)
        │
        ▼
Step 3 — ROI Extraction
        │  Harvard-Oxford Cortical Atlas (cort-maxprob-thr25-2mm)
        │  NiftiLabelsMasker → 47 ROI time series per subject
        │
        ▼
Step 4 — Functional Connectivity
        │  Pearson correlation matrix (47 × 47)
        │
        ▼
Step 5 — Feature Vector
        │  Upper triangle of correlation matrix (k=1)
        │  Feature length: 1081 per subject
        │
        ▼
Step 6 — Classification
        │  ┌─────────┐  ┌─────────┐  ┌──────────┐
        │  │   SVM   │  │   CNN   │  │  ResNet  │
        │  └────┬────┘  └────┬────┘  └────┬─────┘
        │       └────────────┴────────────┘
        │              Majority Vote
        ▼
Final Prediction: Normal / Amblyopia
```

---

## 🤖 Models

### 1. SVM (Support Vector Machine)
- Kernel: `linear`
- Input: flattened upper-triangle connectivity vector (1081 features)
- Preprocessing: `StandardScaler`
- Library: `scikit-learn`

### 2. CNN (Convolutional Neural Network)
```
Input (1, 48, 48)
  → Conv2D(1→16, 3×3) + ReLU + MaxPool(2×2)
  → Conv2D(16→32, 3×3) + ReLU + MaxPool(2×2)
  → Flatten → FC(32×12×12 → 64) + ReLU
  → FC(64 → 2)
```
- Optimizer: Adam (lr=0.001), Epochs: 15
- Library: `PyTorch`

### 3. ResNet (Small Residual Network)
```
Input (1, 48, 48)
  → Conv2D(1→32, 3×3) + ReLU
  → ResidualBlock(32) × 2
  → MaxPool(2×2)
  → Flatten → FC(32×24×24 → 2)
```
Each Residual Block: `Conv → ReLU → Conv → Skip Connection`
- Optimizer: Adam (lr=0.001), Epochs: 15
- Library: `PyTorch`

### 4. Ensemble (Majority Vote)
Final prediction is determined by majority vote across all three models:
```python
final_pred = 1 (Amblyopia) if votes.count(1) >= 2 else 0 (Normal)
```

---

## 📊 Results

> ⚠️ Results below are on **synthetic/limited data** (small sample size). Accuracy on real full dataset may vary.

| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| SVM     | 37.5%    | 0.40      | 0.50   | 0.44     |
| CNN     | 62.5%    | 0.67      | 0.50   | 0.57     |
| ResNet  | 50.0%    | 0.50      | 1.00   | 0.67     |

**Sample Prediction Output:**
```
SVM           : Normal
CNN           : Normal
ResNet        : Amblyopia
Final Decision: Normal
```

**Single subject test (sbj01/Amblyopia/pre):**
```
Predicted Class : Amblyopia
Confidence      : 33.65%
```

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/amblyopia-fmri-detection.git
cd amblyopia-fmri-detection
```

### 2. Install dependencies
```bash
pip install nilearn nibabel scikit-learn matplotlib torch torchvision joblib numpy
```

### 3. Mount Google Drive (if using Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🚀 Usage

### Run the full pipeline in Google Colab

Open `Untitled16.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells sequentially.

### Predict on a new fMRI scan
```python
from predict import predict_amblyopia

fmri_path = "/content/drive/MyDrive/Dataset1/Amblyopia/sbj01/pre/restingEyeOpen.nii.gz"
pred, confidence = predict_amblyopia(fmri_path)

if pred == 1:
    print("Predicted Class: Amblyopia")
else:
    print("Predicted Class: Normal")

print(f"Confidence: {round(confidence * 100, 2)}%")
```

### Run ensemble prediction
```python
prediction = predict_amblyopia(connectivity_matrix)  # numpy (48, 48)
for model, result in prediction.items():
    print(model, ":", result)
```

---

## 📁 File Structure

```
amblyopia-fmri-detection/
│
├── Untitled16.ipynb          # Main Colab notebook (full pipeline)
├── README.md                 # This file
│
├── models/
│   ├── svm_amblyopia_model.pkl   # Saved SVM model
│   └── scaler.pkl                # Saved StandardScaler
│
└── src/
    ├── preprocess.py         # fMRI loading + cleaning
    ├── feature_extraction.py # ROI extraction + connectivity
    ├── train_svm.py          # SVM training
    ├── train_cnn.py          # CNN training (PyTorch)
    ├── train_resnet.py       # ResNet training (PyTorch)
    └── predict.py            # Inference function
```

---

## 🐛 Known Issues & Fixes

### Feature size mismatch across subjects
**Problem:** Different subjects produce different ROI counts after resampling, causing feature vector length inconsistency.

**Fix applied:**
```python
# Filter to most common feature length
from collections import Counter
feature_lengths = [len(x) for x in X_all]
most_common_length = Counter(feature_lengths).most_common(1)[0][0]
X_filtered = [x for x in X_all if len(x) == most_common_length]
```

### Too few samples for train-test split
**Problem:** With only 2 usable samples, stratified split fails — `"least populated class has only 1 sample"`.

**Fix:** Use `LeaveOneOut` cross-validation or collect more data per class (minimum 10 subjects per class recommended).

### FutureWarning — zscore deprecation
**Problem:**
```
FutureWarning: boolean values for 'standardize' deprecated.
Use 'zscore_sample' instead of 'True'
```
**Fix:**
```python
# Replace
masker = NiftiLabelsMasker(atlas.maps, standardize=True)
# With
masker = NiftiLabelsMasker(atlas.maps, standardize='zscore_sample')
```

---

## 🔮 Future Work

- [ ] Collect larger dataset (≥ 20 subjects per class for reliable training)
- [ ] Add Graph Neural Network (GNN) on connectivity matrices
- [ ] Apply proper cross-validation (k-fold or leave-one-out)
- [ ] Include post-treatment (`post1`) data for longitudinal analysis
- [ ] Add anatomical MRI (`anat.nii.gz`) as additional modality
- [ ] Deploy as a web app using Flask/FastAPI for clinical use
- [ ] Add Grad-CAM visualization for CNN interpretability
- [ ] Explore transfer learning with pre-trained brain models

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `nilearn` | 0.13.1 | fMRI preprocessing, ROI extraction, atlas |
| `nibabel` | latest | NIfTI file loading |
| `numpy` | ≥1.22.4 | Numerical operations |
| `scikit-learn` | latest | SVM, StandardScaler, metrics |
| `torch` | latest | CNN and ResNet models |
| `matplotlib` | latest | Plotting time series and connectivity matrix |
| `joblib` | ≥1.2.0 | Model saving and loading |
| `pandas` | ≥2.2.0 | Data handling |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Nilearn](https://nilearn.github.io/) — neuroimaging machine learning library
- [Harvard-Oxford Cortical Atlas](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases) — FSL brain atlas
- [NITRC](https://www.nitrc.org/) — Neuroimaging Informatics Tools and Resources Clearinghouse
- Google Colab for GPU/compute support

---

> ⭐ If this project helped you, please consider giving it a star on GitHub!
