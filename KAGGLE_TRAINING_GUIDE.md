# 🚀 ASTRAVIVE v2.1: Kaggle Cloud Training & Benchmark Guide

This guide explains how to run, train, evaluate, and export the **ASTRAVIVE** machine learning pipeline inside a **free Kaggle Notebook** (using CPU or GPU), leverage external datasets, and export trained model bundles.

---

## 1. Why Run on Kaggle?
* **Free Compute:** Access to 30 hours/week of free cloud compute (CPU, GPU P100/T4, or TPU).
* **Direct Access to Open Datasets:** Kaggle hosts copies of wearable datasets (such as WESAD and PhysioNet MMASH derivatives).
* **Reproducible Demonstration:** You can make your Kaggle notebook public and link it directly in your NASA Space Apps project submission under **"Code / Notebook Links"** so judges can verify your cross-validation with a single click.

---

## 2. Kaggle Setup & Environment Configuration

### Step 1: Create a New Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Click **Create** $\rightarrow$ **New Notebook**.
3. In the right-hand panel (**Notebook options**):
   * **Language:** Python
   * **Accelerator:** CPU (or GPU T4 $\times$ 2 if you plan to extend with deep learning)
   * **Persistence:** No persistence (or Files only)
   * **Environment:** Always use latest environment

---

## 3. Kaggle Notebook Cells (Copy-Paste Ready)

### Cell 1: Clone or Download the ASTRAVIVE Codebase
If your repo is public on GitHub:
```python
# Cell 1: Clone Project Repository
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/ASTRAVIVE.git
%cd ASTRAVIVE
!ls -la
```
*If you haven't pushed to GitHub yet, you can upload `ASTRAVIVE_NASA_SpaceApps_2026_Complete.zip` directly to Kaggle via the **"Add Input" -> "Upload Dataset/File"** button and unzip it:*
```python
# Alternative Cell 1: Unzip uploaded package
!unzip -q /kaggle/input/astravive-package/ASTRAVIVE_NASA_SpaceApps_2026_Complete.zip -d .
!ls -la
```

---

### Cell 2: Verify Dependencies
Kaggle comes pre-installed with `scikit-learn`, `pandas`, `numpy`, and `scipy`. Install specific dependencies if needed (e.g., `shap`):
```python
# Cell 2: Install pinned requirements
!pip install -q shap==0.46.0 streamlit==1.38.0
```

---

### Cell 3: Execute Stage 1–4 Data Ingestion & Physiological Bounds Gatekeeper
```python
# Cell 3: Run Data Ingestion & Preprocessing Pipeline
import sys
sys.path.append('.')

from astravive_data_pipeline import AstraviveDataPipeline

print("=== Running ASTRAVIVE Data Pipeline on Kaggle ===")
pipeline = AstraviveDataPipeline(
    output_dir="/kaggle/working/data/processed",
    sqlite_db_path="/kaggle/working/astravive.db",
    allow_synthetic_fallback=True
)

summary = pipeline.run_pipeline()
print(f"Data Pipeline completed successfully with status: {summary.get('status', 'OK')}")
```

---

### Cell 4: Train Models & Run GroupKFold Cross-Validation
This step fits all models under subject-held-out cross-validation to prevent data leakage:
```python
# Cell 4: Execute ML Training Pipeline
from astravive_ml_pipeline import AstraviveMLPipeline

print("=== Training ASTRAVIVE ML Models & Evaluating Cross-Validation ===")
ml_engine = AstraviveMLPipeline(
    data_dir="/kaggle/working/data/processed",
    models_output_dir="/kaggle/working/models"
)

evaluation_report = ml_engine.execute_reproducible_ml_benchmark()

# Display Benchmark Results
import pandas as pd
import json

with open("/kaggle/working/models/astravive_ml_evaluation_report.json", "r") as f:
    report = json.load(f)

print("\n--- Model Benchmark Performance ---")
metrics_table = []
for model_name, res in report.get("classification_benchmark", {}).items():
    metrics_table.append({
        "Model": model_name,
        "Accuracy": f"{res.get('accuracy_mean', 0.0):.3f} ± {res.get('accuracy_std', 0.0):.3f}",
        "F1 Score": f"{res.get('f1_mean', 0.0):.3f} ± {res.get('f1_std', 0.0):.3f}",
        "ROC-AUC": f"{res.get('roc_auc_mean', 0.0):.3f} ± {res.get('roc_auc_std', 0.0):.3f}",
        "Leakage-Free Validation": "GroupKFold (Subject-wise)"
    })

df_metrics = pd.DataFrame(metrics_table)
display(df_metrics)
```

---

### Cell 5: Calibrate Explainable AI (SHAP TreeExplainer)
```python
# Cell 5: Calculate SHAP Feature Attributions
from astravive_xai_engine import AstraviveXAIEngine

print("=== Computing SHAP Explainability & Stressor Attributions ===")
xai_engine = AstraviveXAIEngine(
    models_dir="/kaggle/working/models",
    output_dir="/kaggle/working/models"
)

xai_summary = xai_engine.run_xai_pipeline()
print(f"SHAP explanation completed. Mean residual: {xai_summary.get('residual_mean', 0.0)}")
```

---

### Cell 6: Run Full End-to-End System Integrator & ODE Digital Twin
```python
# Cell 6: System Orchestrator & Multi-Scenario Digital Twin
from astravive_system_integrator import AstraviveSystemIntegrator

integrator = AstraviveSystemIntegrator(
    output_dir="/kaggle/working/data/integrated_bundle",
    sqlite_db_path="/kaggle/working/astravive.db"
)

bundle = integrator.execute_integrated_workflow(mission_duration_sols=90)
print(f"Overall System Status: {bundle.overall_system_status}")
print(f"Data Mode: {bundle.data_mode}")
```

---

### Cell 7: Download Output Artifacts From Kaggle
```python
# Cell 7: Compress outputs for download
!zip -r /kaggle/working/astravive_kaggle_models.zip /kaggle/working/models /kaggle/working/data/integrated_bundle
print("Trained model bundle ready for download in /kaggle/working/astravive_kaggle_models.zip")
```

---

## 4. How to Attach External Datasets on Kaggle

If you want to train on real physiological datasets instead of synthetic demo fallback:

1. In the Kaggle notebook, click **+ Add Input** (top right).
2. Search for:
   * **"WESAD"** (Wearable Stress and Affect Detection)
   * **"MMASH"** (Multilevel Monitoring of Activity and Sleep in Healthy People)
3. Note the input path (e.g., `/kaggle/input/wesad-wearable-stress-and-affect-detection/`).
4. Update `AstraviveDataPipeline` in Cell 3:
   ```python
   pipeline = AstraviveDataPipeline(
       external_data_dir="/kaggle/input/wesad-wearable-stress-and-affect-detection",
       output_dir="/kaggle/working/data/processed",
       allow_synthetic_fallback=False
   )
   ```
5. When trained on the external dataset, `data_mode` switches automatically to **`EXTERNAL_DATASET`**.

---

## 5. Sharing Your Kaggle Notebook with NASA Space Apps Judges

1. Click **Save Version** (top right) $\rightarrow$ Choose **Quick Save** or **Save & Run All (Commit)**.
2. Under **Sharing**, change the visibility from **Private** to **Public**.
3. Copy your public notebook URL:
   `https://www.kaggle.com/<YOUR_KAGGLE_USERNAME>/astravive-model-training`
4. Paste this URL into your NASA Space Apps submission page in the **Additional Links / Project Code** field.
