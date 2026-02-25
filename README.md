# PTSD Detection Using Machine Learning — Multimodal v2

Upgraded project aligned with your research paper's **multi-faceted framework**: speech signal analysis, facial expression analysis, and text-based behavioural biomarkers. Includes both **binary classification** (PTSD / Non-PTSD) and **severity estimation** (0–10 continuous score).

---

## What Changed From v1

| Feature | v1 | v2 (this version) |
|---|---|---|
| Modalities | Text only | **Text + Audio + Facial** |
| Dataset size | 1 200 | **3 000 per modality (9 000 total)** |
| Class balance | None | **SMOTE oversampling** |
| Models | 4 basic classifiers | 4 classifiers + **Stacking Ensemble + XGBoost** |
| Tuning | None | **GridSearchCV-ready architecture** |
| Evaluation | Accuracy, F1 | + **ROC-AUC, Precision-Recall** |
| Severity | Not included | **Regression model per modality (MAE, R²)** |
| Final decision | Single model | **Majority-vote multimodal fusion** |
| Plots | 2 | **5 (confusion, ROC, comparison, severity scatter, report)** |

---

## Project Structure

```
ptsd_detection_v2/
├── requirements.txt
├── generate_dataset.py          ← creates 3 CSVs (text / audio / facial)
├── train_model.py               ← trains all 3 modalities end-to-end
├── predict.py                   ← multimodal interactive predictor
├── README.md
│
├── data/                        ← auto-created by generate_dataset.py
│   ├── text_dataset.csv         (3 000 rows: text, label, severity)
│   ├── audio_dataset.csv        (3 000 rows: 42 audio features, label, severity)
│   └── facial_dataset.csv       (3 000 rows: 30 facial features, label, severity)
│
├── model/                       ← auto-created by train_model.py
│   ├── text_bundle.pkl          (vectorizer + best classifier + regressor)
│   ├── audio_bundle.pkl         (scaler + best classifier + regressor)
│   └── facial_bundle.pkl        (scaler + best classifier + regressor)
│
└── results/                     ← auto-created by train_model.py
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── cross_modality_comparison.png
    ├── severity_regression.png
    └── classification_report.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate all three datasets
python generate_dataset.py

# 3. Train all models (text + audio + facial)
python train_model.py

# 4. Interactive prediction
python predict.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐   │
│  │  Text    │   │  Audio (.wav)│   │  Video (frames)    │   │
│  │ (typed)  │   │  via librosa │   │  via dlib/OpenCV   │   │
│  └────┬─────┘   └──────┬───────┘   └─────────┬──────────┘   │
│       │                │                     │               │
│ ┌─────▼─────┐  ┌───────▼───────┐   ┌────────▼─────────┐    │
│ │ TF-IDF    │  │ MFCC, Pitch,  │   │ 68-pt Landmarks, │    │
│ │ (1-3gram) │  │ Energy, Jitter│   │ Action Units,    │    │
│ │ 8 000 dim │  │ 42 features   │   │ 30 features      │    │
│ └─────┬─────┘  └───────┬───────┘   └────────┬─────────┘    │
│       │                │                     │               │
│ ┌─────▼─────┐  ┌───────▼───────┐   ┌────────▼─────────┐    │
│ │ Stacking  │  │ Stacking      │   │ Stacking         │    │
│ │ Ensemble  │  │ Ensemble      │   │ Ensemble         │    │
│ │ (LR+NB+RF)│  │ (LR+RF+GB+XGB)│   │ (LR+RF+GB+XGB)  │    │
│ └─────┬─────┘  └───────┬───────┘   └────────┬─────────┘    │
│       │                │                     │               │
│       ▼                ▼                     ▼               │
│   pred + sev       pred + sev           pred + sev           │
│       │                │                     │               │
│       └────────┬───────┴─────────────────────┘               │
│                ▼                                              │
│        ┌───────────────┐                                     │
│        │ FUSION LAYER  │  ← majority vote (classification)   │
│        │               │     average        (severity)       │
│        └───────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Techniques

### SMOTE (Synthetic Minority Over-sampling)
The dataset has a realistic 40/60 class imbalance (PTSD cases are less common). SMOTE generates synthetic PTSD samples so the model doesn't just learn to predict "Non-PTSD" all the time. Applied **only to the training set** — test set stays untouched.

### Stacking Ensemble
Instead of picking one model, a **meta-learner** (Logistic Regression) sits on top of 3–4 base models (LR, Random Forest, Gradient Boosting, XGBoost). Each base model's predictions become features for the meta-learner. This consistently outperforms any single model.

### Severity Regression
Beyond binary yes/no, the paper requires **severity estimation**. A separate regressor (XGBoost for structured data, Ridge for text) predicts a continuous 0–10 score per modality. The final severity is the **average across all modalities**.

### ROC-AUC
Accuracy alone is misleading on imbalanced data. ROC-AUC measures how well the model ranks PTSD cases above non-PTSD cases across all possible thresholds — a much more robust metric for medical screening.

### Multimodal Fusion
The final prediction combines all three modalities via **majority voting** (classification) and **score averaging** (severity). This mirrors the paper's proposed framework of using multiple behavioural biomarkers together.

---

## Severity Scale

| Score | Label |
|---|---|
| 0.0 – 2.0 | Minimal |
| 2.1 – 4.0 | Low |
| 4.1 – 6.0 | Moderate |
| 6.1 – 8.0 | High |
| 8.1 – 10.0 | Severe |

---

## Swapping In Real Data

### Real Text Data
Same as v1 — download from Kaggle, put in `data/`, update the path in `train_text()`.

### Real Audio Data
1. Collect `.wav` files (labelled PTSD / non-PTSD)
2. Run this extraction script on each file:

```python
import librosa
import numpy as np

def extract_audio_features(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {}
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i+1}_std"]  = np.std(mfccs[i])

    pitches, _ = librosa.pipagator.pitch(y, sr=sr)  # or use crepe
    features["pitch_mean"]  = np.nanmean(pitches[pitches > 0])
    features["pitch_std"]   = np.nanstd(pitches[pitches > 0])
    features["pitch_range"] = np.nanptp(pitches[pitches > 0])

    energy = librosa.feature.rms(y=y)[0]
    features["energy_mean"] = np.mean(energy)
    features["energy_std"]  = np.std(energy)
    features["energy_max"]  = np.max(energy)
    # ... add jitter, shimmer, spectral features similarly
    return features
```

3. Compile into a DataFrame with the same column names as `data/audio_dataset.csv`, add `label` and `severity` columns.

### Real Facial Data
1. Collect video files of interviews (labelled)
2. Use `dlib` to extract 68 facial landmarks per frame
3. Compute Action Unit intensities (use OpenFace or similar)
4. Aggregate per-video into the 30 features in `data/facial_dataset.csv`

---

## Disclaimer

This is a **research prototype** for academic purposes only. PTSD is a serious medical condition that must be diagnosed by a qualified mental health professional. No automated system should be used as a substitute for clinical assessment.
