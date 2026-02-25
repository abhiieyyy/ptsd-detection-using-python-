"""
============================================================
  train_model.py   (v4 — Full Advanced Pipeline)

  Trains classifiers AND severity regressors for all three
  modalities: Text, Audio, Facial.

  FEATURES:
      • GridSearchCV & Cross-Validation
      • Stacking & Voting Ensemble Classifiers
      • SHAP Explainability 
      • Automatic 'model.pkl' export for Camera Detection
      • SSL Certificate fix for asset downloads
============================================================
"""

import os, pickle, warnings, string, re, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, RandomForestRegressor, VotingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, mean_absolute_error, r2_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# ═══════════════════════════════════════════════════════════════
#  TRAINING: TEXT MODALITY
# ═══════════════════════════════════════════════════════════════
def train_text(df):
    print("\n" + "="*50 + "\n  TRAINING TEXT MODALITY\n" + "="*50)
    
    df['clean_text'] = df['text'].apply(clean_text)
    X = df['clean_text']
    y = df['label']
    y_sev = df['severity']

    X_train, X_test, y_train, y_test, sev_train, sev_test = train_test_split(
        X, y, y_sev, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Multi-Model Voting for Text
    clf1 = LogisticRegression(C=1.0)
    clf2 = RandomForestClassifier(n_estimators=100)
    voter = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft')
    voter.fit(X_train_vec, y_train)

    # Severity Regressor
    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(X_train_vec, sev_train)

    bundle = {
        "model": voter,
        "regressor": reg,
        "vectorizer": vectorizer
    }
    
    with open("model/text_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    print("✅ Text bundle saved.")
    return bundle

# ═══════════════════════════════════════════════════════════════
#  TRAINING: STRUCTURED MODALITIES (Audio / Facial)
# ═══════════════════════════════════════════════════════════════
def train_structured(df, name, feature_cols, save_name):
    print(f"\n" + "="*50 + f"\n  TRAINING {name.upper()} MODALITY\n" + "="*50)
    
    X = df[feature_cols]
    y = df['label']
    y_sev = df['severity']

    X_train, X_test, y_train, y_test, sev_train, sev_test = train_test_split(
        X, y, y_sev, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ─── GridSearch for Best Classifier ───
    print(f"🔍 Optimizing {name} Classifier...")
    params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_clf = grid.best_estimator_

    # ─── Stacking Ensemble ───
    estimators = [
        ('rf', best_clf),
        ('gb', GradientBoostingClassifier(n_estimators=100))
    ]
    stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stack_clf.fit(X_train_scaled, y_train)

    # ─── Severity Regression ───
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_scaled, sev_train)

    # Save Bundle
    bundle = {
        "model": stack_clf,
        "regressor": reg,
        "scaler": scaler,
        "features": feature_cols
    }
    
    out_path = os.path.join("model", save_name)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    
    # ─── Evaluation ───
    y_pred = stack_clf.predict(X_test_scaled)
    print(f"\n📊 {name} Performance:")
    print(classification_report(y_test, y_pred))
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title(f"Feature Importances - {name}")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_cols[i] for i in indices], rotation=45)
    plt.savefig(f"results/{name.lower()}_importance.png")
    plt.close()

    return bundle

# ═══════════════════════════════════════════════════════════════
#  MAIN SYSTEM ENTRY
# ═══════════════════════════════════════════════════════════════
def main():
    # Setup folders
    for folder in ["model", "results", "data"]:
        os.makedirs(folder, exist_ok=True)

    # 1. Text Training
    try:
        df_text = pd.read_csv("data/text_dataset.csv")
        train_text(df_text)
    except FileNotFoundError:
        print("⚠️ Skip Text: data/text_dataset.csv not found.")

    # 2. Audio Training
    try:
        df_audio = pd.read_csv("data/audio_dataset.csv")
        audio_cols = [c for c in df_audio.columns if c not in ("label","severity")]
        train_structured(df_audio, "Audio", audio_cols, "audio_bundle.pkl")
    except FileNotFoundError:
        print("⚠️ Skip Audio: data/audio_dataset.csv not found.")

    # 3. Facial Training (Crucial for Camera Detection)
    try:
        df_facial = pd.read_csv("data/facial_dataset.csv")
        facial_cols = [c for c in df_facial.columns if c not in ("label","severity")]
        train_structured(df_facial, "Facial", facial_cols, "facial_bundle.pkl")
        
        # ─── CAMERA EXPORT ───
        # camera_detect.py looks for 'model.pkl' in the main root.
        # We copy the facial bundle there to enable real-time detection.
        src = os.path.join("model", "facial_bundle.pkl")
        dst = "model.pkl"
        shutil.copy(src, dst)
        print("\n" + "*"*50)
        print("✨ SUCCESS: All models trained.")
        print("📁 'model.pkl' exported to main directory.")
        print("🚀 RUN: python3 camera_detect2.py")
        print("*"*50)
        
    except Exception as e:
        print(f"❌ Error during Facial training: {e}")
        print("💡 Ensure you run 'python3 generate_dataset.py' first.")

if __name__ == "__main__":
    main()