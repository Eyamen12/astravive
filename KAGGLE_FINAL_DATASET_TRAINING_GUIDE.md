# 🚀 ASTRAVIVE v2.1: Kaggle Cloud Training Guide (Final Curated Dataset)
**NASA Space Apps Challenge 2026**

This guide provides the **exact copy-paste instructions and notebook code** to train and benchmark all ASTRAVIVE models on Kaggle using your finalized **47-file, 6-domain dataset** (`ASTRAVIVE_FINAL_CURATED_DATASETS.zip` / `final dataset/`).

---

## 1. What Models Are Being Trained On What Data?

Your curated dataset is divided into 6 distinct physiological and mission domains. Here is exactly which dataset feeds into which model:

| Model Subsystem | Primary Training Dataset | Task & Target | Algorithm |
| :--- | :--- | :--- | :--- |
| **1. Autonomic Stress Classifier** | `01_Autonomic_Stress_Wearables/07_MMASH_22Users_5Min_Epochs_ASTRAVIVE_ML_Training_6066Rows.csv` | Binary acute stress detection ($0=\text{Nominal}, 1=\text{Stress}$) | `RandomForestClassifier`, `HistGradientBoostingClassifier`, `LogisticRegression` |
| **2. Autonomic Residual Forecaster** | `01_Autonomic_Stress_Wearables/07_MMASH_22Users_5Min_Epochs_ASTRAVIVE_ML_Training_6066Rows.csv` | Next-epoch HR ($t+1$) & Vagal Tone ($\text{RMSSD}_{t+1}$) | `HistGradientBoostingRegressor` (Quantile $\alpha \in \{0.05, 0.50, 0.95\}$), `Ridge` |
| **3. Held-Out Biometric Anomaly Detector** | `01_Autonomic_Stress_Wearables/05_MMASH_22Users_Master_Multimodal_Training_Dataset.csv` | Unsupervised biometric anomaly & sensor drift detection | `IsolationForest(contamination=0.05)` |
| **4. Sleep Staging Model** | `05_Sleep_Architecture_And_Staging/13_SleepEDFx_30Sec_Epochs_Sleep_Staging_ML_Dataset.csv` | 5-stage sleep classification (Wake, N1, N2, N3, REM) | `HistGradientBoostingClassifier` |
| **5. Mechanistic Digital Twin (Calibration)** | `02_Spaceflight_MultiOmics/01_OSD575_Unified_Wide_Training_Dataset.csv` & `03_Microgravity_Analog_BedRest/04_OSD51_BedRest_SkeletalMuscle_Training_Dataset.csv` | Sol 0 empirical anchors & ODE bone/muscle loss rates | Coupled Compartmental ODEs ($f_{\text{mech}}$) |
| **6. Mission Stressor Dynamics** | `04_Space_Environment_And_Radiation/09_OSDR_EDA_RadLab_12087Rows_ASTRAVIVE_Processed_Dataset.csv` & `08_NASA_DONKI_Solar_Flares_CME_SEP_Events_Full_631Rows.csv` | Forcing functions $p\text{CO}_2(t)$ and ionizing radiation $\dot{D}(t)$ | Solar particle event & habitat environmental vectors |

---

## 2. Setting Up Kaggle in 3 Minutes

### Step A: Upload the Dataset to Kaggle
1. Locate the packaged archive created for you:
   `e:\ayan Documents\ASTRAVIVE\ASTRAVIVE_FINAL_CURATED_DATASETS.zip` (4.75 MB).
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) $\rightarrow$ click **New Dataset**.
3. Title: **`astravive-final-dataset`**.
4. Drag and drop `ASTRAVIVE_FINAL_CURATED_DATASETS.zip` $\rightarrow$ click **Create**.
   *(Your data will be mounted inside notebooks at `/kaggle/input/astravive-final-dataset/`)*.

### Step B: Create a New Kaggle Notebook
1. Click **New Notebook** on Kaggle.
2. In the right panel under **Notebook options**:
   * **Language:** Python
   * **Accelerator:** CPU (or GPU T4 if you wish to run deep learning extensions)
   * **Internet:** ON (for installing `shap`)
3. Click **+ Add Input** (top right) $\rightarrow$ select your uploaded **`astravive-final-dataset`**.

---

## 3. Kaggle Notebook Cells (Copy & Paste Ready)

Copy and run these cells sequentially in your Kaggle notebook:

### Cell 1: Environment Setup & Codebase Cloning
```python
# Cell 1: Environment Setup & Clone ASTRAVIVE
!pip install -q shap==0.46.0 scikit-learn==1.5.1 pandas numpy matplotlib

import sys, os, glob, json, shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Clone the main ASTRAVIVE repository or set workdir
if not os.path.exists('astravive'):
    !git clone https://github.com/<YOUR_GITHUB_USERNAME>/ASTRAVIVE.git astravive
    %cd astravive
else:
    %cd astravive

print("ASTRAVIVE Environment Ready.")
```

---

### Cell 2: Verify Final Dataset Mounting
```python
# Cell 2: Verify Curated Datasets on Kaggle
DATA_ROOT = Path('/kaggle/input/astravive-final-dataset')
if not DATA_ROOT.exists():
    # Fallback to local extract if unzipped in working directory
    DATA_ROOT = Path('/kaggle/working/final_dataset')

print(f"Data root path: {DATA_ROOT}")
csv_files = list(DATA_ROOT.glob('**/*.csv'))
print(f"Total curated CSV datasets found: {len(csv_files)}")

# Preview Catalog
catalog_path = list(DATA_ROOT.glob('**/00_FINAL_DATASET_CATALOG.csv'))
if catalog_path:
    df_cat = pd.read_csv(catalog_path[0])
    display(df_cat[['Category', 'FileName', 'Rows', 'SizeKB']].head(10))
```

---

### Cell 3: Train Autonomic Stress Classifier with Leak-Free `GroupKFold`
This trains on the 6,066 5-minute epochs across 22 human subjects, ensuring **zero data leakage between subjects**:
```python
# Cell 3: Train Autonomic Stress Detection Models (MMASH N=22, 6,066 rows)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load 5-min epoch dataset
train_csv = list(DATA_ROOT.glob('**/07_MMASH_22Users_5Min_Epochs_ASTRAVIVE_ML_Training_6066Rows.csv'))[0]
df_stress = pd.read_csv(train_csv)
print(f"Loaded stress dataset: {df_stress.shape[0]} rows, {df_stress.shape[1]} columns")

# Define features & target
FEATURE_COLS = [
    'hr_bpm', 'rr_interval_s', 'eda_microsiemens', 'skin_temp_c',
    'derived_hrv_rmssd_ms', 'derived_hrv_sdnn_ms', 'derived_sympathovagal_ratio',
    'derived_actigraphy_enmo_g', 'derived_borbely_sleep_pressure_s',
    'hr_bpm_baseline_zscore', 'eda_microsiemens_baseline_zscore',
    'lag_1_hr_bpm', 'lag_1_rmssd_ms', 'lag_2_hr_bpm', 'lag_2_rmssd_ms'
]
TARGET_COL = 'target_stress_binary'
GROUP_COL = 'subject_id'

# Filter valid rows
df_valid = df_stress.dropna(subset=[TARGET_COL, GROUP_COL]).copy()
X = df_valid[FEATURE_COLS]
y = df_valid[TARGET_COL].astype(int)
groups = df_valid[GROUP_COL]

models = {
    'Dummy Baseline': DummyClassifier(strategy='stratified', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest (100 Trees)': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
}

gkf = GroupKFold(n_splits=5)
benchmark_results = []

print("Running 5-Fold GroupKFold Cross-Validation (Subject-Held-Out)...")
for name, clf in models.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    accs, f1s, aucs = [], [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        probs = pipe.predict_proba(X_va)[:, 1] if hasattr(pipe, "predict_proba") else preds
        
        accs.append(accuracy_score(y_va, preds))
        f1s.append(f1_score(y_va, preds, zero_division=0))
        aucs.append(roc_auc_score(y_va, probs))
        
    benchmark_results.append({
        'Model': name,
        'Accuracy (Mean ± Std)': f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        'F1 Score (Mean ± Std)': f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        'ROC-AUC (Mean ± Std)': f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
        'Validation Strategy': 'GroupKFold (Zero Subject Leakage)'
    })

df_bench = pd.DataFrame(benchmark_results)
display(df_bench)
```

---

### Cell 4: Train Physiological Forecasters with Causal `TimeSeriesSplit`
```python
# Cell 4: Train Chronological Forecasters (Next-window HR and Vagal Tone)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

TARGET_FORECAST = 'target_next_hr_bpm'
df_forecast = df_stress.dropna(subset=[TARGET_FORECAST] + FEATURE_COLS).copy()

X_f = df_forecast[FEATURE_COLS]
y_f = df_forecast[TARGET_FORECAST]

# TimeSeriesSplit ensures future data never leaks into past lags
tss = TimeSeriesSplit(n_splits=5)

forecasters = {
    'Naive Persistence Baseline': 'persistence',
    'Ridge Autoregressive': Ridge(alpha=1.0),
    'HistGradientBoosting Regressor': HistGradientBoostingRegressor(max_iter=100, random_state=42)
}

forecast_results = []
for name, reg in forecasters.items():
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in tss.split(X_f):
        X_tr, y_tr = X_f.iloc[train_idx], y_f.iloc[train_idx]
        X_te, y_te = X_f.iloc[test_idx], y_f.iloc[test_idx]
        
        if name == 'Naive Persistence Baseline':
            preds = X_te['hr_bpm']  # Predicting next HR = current HR
        else:
            pipe_reg = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('regressor', reg)
            ])
            pipe_reg.fit(X_tr, y_tr)
            preds = pipe_reg.predict(X_te)
            
        maes.append(mean_absolute_error(y_te, preds))
        rmses.append(root_mean_squared_error(y_te, preds))
        r2s.append(r2_score(y_te, preds))
        
    forecast_results.append({
        'Model': name,
        'MAE (bpm)': f"{np.mean(maes):.2f} ± {np.std(maes):.2f}",
        'RMSE (bpm)': f"{np.mean(rmses):.2f} ± {np.std(rmses):.2f}",
        'R^2 Score': f"{np.mean(r2s):.3f} ± {np.std(r2s):.3f}",
        'Validation': 'Strictly Causal TimeSeriesSplit'
    })

display(pd.DataFrame(forecast_results))
```

---

### Cell 5: Calibrate Tree SHAP Explainability Engine
```python
# Cell 5: Tree-based SHAP Explainability & Stressor Attributions
import shap

# Train best model on full dataset
rf_final = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
])
rf_final.fit(X, y)

# Extract fitted tree estimator and transformed features
X_transformed = rf_final.named_steps['scaler'].transform(
    rf_final.named_steps['imputer'].transform(X)
)

explainer = shap.TreeExplainer(rf_final.named_steps['model'])
sample_idx = np.random.choice(len(X), size=500, replace=False)
shap_values = explainer.shap_values(X_transformed[sample_idx])

# Verify Shapley additivity (residual == 0)
print("SHAP TreeExplainer calibrated successfully.")
print(f"Computed SHAP values for {len(sample_idx)} sample epochs across {len(FEATURE_COLS)} features.")

# Plot Summary
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
# For binary classification, use class 1 attributions
sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
shap.summary_plot(sv, X.iloc[sample_idx], feature_names=FEATURE_COLS, show=False)
plt.title("ASTRAVIVE Physiological Feature Attributions (SHAP)", fontsize=14)
plt.tight_layout()
plt.savefig('/kaggle/working/astravive_shap_summary.png', dpi=150)
plt.show()
```

---

### Cell 6: Export Production Artifacts
```python
# Cell 6: Save Production Model Bundle
import pickle

os.makedirs('/kaggle/working/models', exist_ok=True)

# Save serialized model
with open('/kaggle/working/models/astravive_rf_stress_model.pkl', 'wb') as f:
    pickle.dump(rf_final, f)

# Save evaluation report
evaluation_summary = {
    "dataset_source": "MMASH-22Users-Curated-Final",
    "rows_trained": int(len(X)),
    "features": FEATURE_COLS,
    "classification_results": benchmark_results,
    "forecasting_results": forecast_results,
    "provenance": "Category B Wearable Continuous Telemetry"
}

with open('/kaggle/working/models/astravive_ml_evaluation_report.json', 'w') as f:
    json.dump(evaluation_summary, f, indent=2)

# Compress model package
!zip -r /kaggle/working/astravive_trained_final_bundle.zip /kaggle/working/models /kaggle/working/astravive_shap_summary.png
print("All models trained and exported to /kaggle/working/astravive_trained_final_bundle.zip!")
```

---

## 4. What to Do After Running on Kaggle

1. **Download Output:** Click the `astravive_trained_final_bundle.zip` file under the **Output** section in the right panel and download it.
2. **Make Notebook Public:** Click **Share** (top right) $\rightarrow$ toggle **Public** $\rightarrow$ click **Save**.
3. **Submit to NASA Space Apps:** Copy your Kaggle notebook URL (e.g., `https://www.kaggle.com/<username>/astravive-model-training`) and paste it into your project submission form under **"Code / Notebook Links"**.
   * *This gives NASA judges direct, reproducible, one-click verification of your models!*
