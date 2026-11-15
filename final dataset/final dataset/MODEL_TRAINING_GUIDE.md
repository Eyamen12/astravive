# 🧠 ASTRAVIVE v2.1: Complete Model Training & ML Pipeline Guide

> **Project:** ASTRAVIVE — Mission-Health Digital Twin  
> **Challenge:** NASA Space Apps Challenge 2026  
> **Key Architecture Principle:** Hybrid Mechanistic Compartmental ODEs ($f_{\text{mech}}$) coupled with small-sample, anti-leakage Machine Learning ($g_{\text{ML}}$).

---

## 1. Why ASTRAVIVE Trains the Way It Does (The Core Philosophy)

Before running training, it is crucial to understand **what** gets trained and **what does not**:

| Data Domain | Datasets | Sample Size | Training Role | Method |
| :--- | :--- | :--- | :--- | :--- |
| **Wearables & Autonomics** | PhysioNet MMASH, UCI WESAD | $N=22$, $N=15$ | **Supervised & Anomaly ML Training** | `RandomForest`, `HistGradientBoosting`, `IsolationForest` |
| **Astronaut Spaceflight** | NASA OSDR `OSD-575`, `OSD-569` | $N=4$ | **Empirical Baseline Anchors** (Not ML training) | Initial values ($S_0$, $BMD_0$) at Sol 0 |
| **Space Analogs** | NASA OSDR `OSD-51` (Bed Rest) | $N=24$ | **Mechanistic Calibration** | Parameter constraints on ODE loss rates |
| **Telemetry & Weather** | OSDR EDA, RadLab, NASA DONKI | Continuous | **Stressor Forcing Functions** | $p\text{CO}_2(t)$, Radiation Dose $\dot{D}(t)$ |

> **Why we DO NOT train deep learning on $N=4$ astronauts:** Training complex neural networks on Inspiration4 ($N=4$) creates catastrophic overfitting and data leakage. Instead, ASTRAVIVE trains lightweight, leak-free Scikit-learn models on terrestrial wearable autonomic streams ($N>35$), using subject-wise **GroupKFold** cross-validation. For long-duration spaceflight (up to 900 Sols), trajectory evolution is governed by **NASA HRP-anchored mechanistic ODEs**.

---

## 2. Models in the Pipeline

ASTRAVIVE trains and benchmarks **four distinct model classes**:

### A. Autonomic Stress & Fatigue Classification
* **Target:** Acute autonomic stress state ($0 = \text{Nominal}$, $1 = \text{Elevated Stress}$)
* **Features:** Heart Rate ($\text{HR}$), Heart Rate Variability ($\text{RMSSD}$, $\text{SDNN}$), Respiration Rate ($\text{RR}$), Skin Temperature, Electrodermal Activity ($\text{EDA}$).
* **Models Evaluated:**
  * `DummyClassifier` (Stratified Baseline)
  * `LogisticRegression` (L2 penalized, balanced weights)
  * `RandomForestClassifier` ($100$ trees, `max_depth=5`)
  * `GradientBoostingClassifier` / `HistGradientBoosting`

### B. Physiological Residual Forecasting (Short-Horizon)
* **Target:** Next-window Heart Rate / Vagal Tone residual ($t+1$)
* **Validation:** Strictly causal `TimeSeriesSplit(n_splits=5)` per subject (no past/future leakage).
* **Models Evaluated:**
  * `Persistence` (Naive baseline: $\hat{y}_{t+1} = y_t$)
  * `Ridge` Regressor (Autoregressive lag features $t-1, t-2, t-3$)
  * `HistGradientBoostingRegressor` with native quantile loss ($\alpha \in \{0.05, 0.50, 0.95\}$)

### C. Held-Out Sensor Anomaly Detection
* **Model:** `IsolationForest(contamination=0.05)`
* **Protocol:** Fitted strictly on nominal baseline observations, tested on out-of-sample perturbed streams (arrhythmias, hypercapnic sensor jumps).

### D. Explainability Engine (XAI)
* **Model:** Tree-based **SHAP** (`shap.TreeExplainer`) with exact baseline Shapley and permutation fallback chains. Computes feature attributions $\phi_j$ without heuristic rescaling.

---

## 3. Step-by-Step Training Execution

### Step 1: Prepare the Environment
Ensure your terminal is in the project root with the virtual environment activated:
```bash
cd "e:\ayan Documents\ASTRAVIVE\ASTRAVIVE v2.1 Complete Package\astravive"
# Activate virtualenv (PowerShell)
.venv\Scripts\Activate.ps1
# Install pinned dependencies
pip install -r requirements.txt
```

### Step 2: Run Data Preprocessing & Feature Engineering
Before model fitting, run the data pipeline to validate bounds, filter ectopic intervals, and build clean feature tables:
```bash
python astravive_data_pipeline.py
```
**What this does:**
1. Validates physiological bounds ($30 \le \text{HR} \le 220\text{ bpm}$, $0.27 \le \text{RR} \le 2.0\text{ s}$).
2. Computes rolling $\text{RMSSD}$, $\text{SDNN}$, and Borbély two-process sleep pressure ($S_t$).
3. Exports processed streams to `data/processed/mmash_wearable_processed.csv` and `.json`.

### Step 3: Run the ML Training & Benchmark Engine
Execute the standalone machine learning training suite:
```bash
python astravive_ml_pipeline.py
```
**Expected Console Output:**
* **GroupKFold Cross-Validation:** Displays out-of-subject validation scores across all 5 folds.
* **Model Benchmark Table:** Evaluates Accuracy, Precision, Recall, F1, and ROC-AUC.
* **Forecasting Metric Comparison:** Displays MAE, RMSE, and $R^2$ comparing persistence against gradient boosting.
* **Saved Artifacts:**
  * `models/astravive_trained_models.pkl` (Serialized model pipeline)
  * `models/astravive_trained_model_bundle.json` (Model metadata and parameters)
  * `models/astravive_ml_evaluation_report.json` (Dynamic validation metrics)

### Step 4: Run Explainability (SHAP / XAI Engine)
Generate feature importance and individual prediction attributions:
```bash
python astravive_xai_engine.py
```
**What this does:**
1. Calculates SHAP summary values ($\phi_j$) for physiological predictors.
2. Verifies additivity ($\sum \phi_j = f(x) - E[f(x)]$) with zero residual.
3. Exports explanations to `models/astravive_xai_explanations.json`.

### Step 5: Run Full End-to-End Orchestrator
Run the full system integrator which unifies data, training, ODE simulations, and bundle packaging:
```bash
python astravive_system_integrator.py
```
This generates the validated `astravive_unified_system_bundle.json` with system status `OPERATIONAL`.

---

## 4. Training on Real External Datasets (Optional)

If you download the raw datasets from PhysioNet or NASA OSDR:

1. **Place raw files in `data/`**:
   * MMASH wearable records $\rightarrow$ `data/mmash/`
   * Inspiration4 metabolic $\rightarrow$ `data/osd_575/`
   * Bed rest analog $\rightarrow$ `data/osd_51/`
2. **Execute with explicit paths**:
   ```python
   from astravive_data_pipeline import AstraviveDataPipeline
   pipeline = AstraviveDataPipeline(
       external_data_dir="data",
       allow_synthetic_fallback=False  # Raises FileNotFoundError if missing
   )
   pipeline.run_pipeline()
   ```
3. When external datasets are found, the system switches `data_mode` from `SYNTHETIC_DEMO` to `EXTERNAL_DATASET`.

---

## 5. Live In-Browser Training & Cross-Validation

You do not need to run Python scripts to demonstrate model training to NASA Space Apps judges:

1. Open `astravive_app.html` or run `streamlit run app.py`.
2. Navigate to **Tab 7: Model Performance**.
3. Click the button: **`Run Live Cross-Validation`**.
4. The browser executes an in-client Leave-One-Subject-Out (LOSO) benchmark across 5 distinct classifiers, proving live computation without pre-baked static metrics.
