> **Archived v2.0 document.** Kept for reference. Where it conflicts with `../AUDIT_REPORT.md` or `../README.md` (v2.1), the v2.1 documents and the code are authoritative.

# ASTRAVIVE — COMPLETE STEP-BY-STEP WORKABILITY & DEPLOYMENT GUIDE

**Project:** ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN  
**Target Event:** NASA Space Apps Challenge 2026

This guide walks you step by step from extracting your project files to running the entire 10-stage Python machine-learning pipeline locally, mapping real downloaded NASA OSDR/PhysioNet datasets, and deploying a live public URL for NASA Space Apps judges in under 10 minutes.

---

## PHASE 1: EXTRACT & ORGANIZE YOUR WORKSPACE (2 Minutes)

### Step 1.1: Download or Extract the Project Files
1. In the Canvas preview of **`astravive_app.html`**, click the green **`DOWNLOAD ALL FILES (.ZIP)`** button in the top-right header bar (or save each file listed below into a folder named `astravive/`).
2. Verify your `astravive/` project folder contains these core files:
```text
astravive/
├── app.py                                 # Streamlit Cloud & local Python web server entry point
├── astravive_app.html                     # Complete interactive 9-tab NASA Mission Control UI
├── astravive_data_pipeline.py             # Stages 1–5: Ingestion, validation, cleaning & features
├── astravive_ml_pipeline.py               # Stage 6: Anti-leakage GroupKFold & TimeSeriesSplit ML
├── astravive_xai_engine.py                # Stage 8: Global & local SHAP explainability engine
├── astravive_digital_twin_simulator.py    # Stages 7 & 9: Scenario A vs. Scenario B ODE simulator
├── astravive_system_integrator.py         # Unified 10-stage orchestrator & fault boundary manager
├── astravive_test_suite.py                # Automated 9-domain Python unittest suite
├── package_astravive_zip.py               # AST syntax verifier & ZIP packager
├── requirements.txt                       # Pinned Python dependencies
└── README.md                              # Complete project documentation
```

---

## PHASE 2: SET UP YOUR LOCAL PYTHON ENVIRONMENT (3 Minutes)

### Step 2.1: Create & Activate a Virtual Environment
Open your terminal (macOS/Linux) or PowerShell/Command Prompt (Windows) inside the `astravive/` folder:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2.2: Install Required Python Packages
Install the pinned dependencies from `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## PHASE 3: CONNECT REAL DOWNLOADED DATASETS (OR USE AUTOMATIC FALLBACK)

ASTRAVIVE works **out of the box with zero manual dataset downloads** because `astravive_data_pipeline.py` automatically generates clearly labeled `SIMULATED DATA` calibrated to published means when raw files are not present.

However, if you have downloaded raw CSV/JSON files from **NASA OSDR** or **PhysioNet**, place them here:

```text
astravive/
└── data/
    └── raw/
        ├── mmash_wearable.csv        # PhysioNet MMASH continuous RR / HR / Actigraphy / Sleep
        ├── osdr_eda_radlab.csv       # NASA OSDR EDA ISS Cabin CO2 & RadLab Dosimetry
        └── osd_575_clinical.csv      # NASA OSDR OSD-575 & OSD-569 Inspiration4 CBC/CMP
```

### Step 3.1: Required Column Headers for Custom CSV/JSON Files
If you pass your own downloaded CSV or JSON file into `pipeline.ingest_dataset()`, ensure its column headers map to the verified schema in `astravive_data_pipeline.py`:

* **For `MMASH-WEARABLE` (`mmash_wearable.csv`):**
  `subject_id, timestamp_utc, rr_interval_s, hr_bpm, actigraphy_axis_x, actigraphy_axis_y, actigraphy_axis_z, sleep_efficiency_pct, waso_minutes`
* **For `OSDR-EDA-RADLAB` (`osdr_eda_radlab.csv`):**
  `subject_id, timestamp_utc, mission_sol, gravity_g, cabin_co2_ppm, cabin_temp_c, cabin_rh_pct, gcr_dose_rate_msv_day, saa_or_spe_dose_rate_msv_day`
* **For `OSD-575` (`osd_575_clinical.csv`):**
  `subject_id, timestamp_utc, timepoint_label, serum_glucose_mg_dl, bun_mg_dl, creatinine_mg_dl, sodium_meq_l, potassium_meq_l, calcium_mg_dl`

To run the pipeline with a real downloaded CSV file, pass `file_path` to `execute_integrated_workflow()`:
```python
from astravive_system_integrator import AstraviveSystemIntegrator

integrator = AstraviveSystemIntegrator()
bundle = integrator.execute_integrated_workflow(
    external_wearable_file="data/raw/mmash_wearable.csv",
    mission_duration_sols=90
)
```

---

## PHASE 4: RUN & VERIFY ALL BACKEND STAGES LOCALLY (3 Minutes)

Run these 6 commands in sequence to verify every module and generate all SQLite, `.pkl`, `.csv`, and `.json` artifacts:

```bash
# 1. Run the 9-domain automated unit & integration test suite (Expect: 6/6 tests OK)
python astravive_test_suite.py

# 2. Run Stages 1–5: Data Ingestion, Validation, Cleaning, Normalization & SQLite Export
python astravive_data_pipeline.py

# 3. Run Stage 6: Anti-Leakage ML Training (GroupKFold + TimeSeriesSplit + IsolationForest)
python astravive_ml_pipeline.py

# 4. Run Stage 8: Global & Individual SHAP Explainable AI Engine
python astravive_xai_engine.py

# 5. Run Stages 7 & 9: Scenario A vs. Scenario B Digital Twin Simulator
python astravive_digital_twin_simulator.py

# 6. Run the Unified 10-Stage System Orchestrator
python astravive_system_integrator.py
```

After running these commands, check your directory—you will now have:
* `data/astravive.db` (Embedded SQLite database with validated tables and audit logs)
* `data/processed/mmash_wearable_processed.csv` & `.json`
* `data/processed/astravive_digital_twin_scenarios_ab.csv` & `.json`
* `models/astravive_trained_models.pkl` & `models/astravive_ml_evaluation_report.json`
* `models/astravive_xai_explanations.json`
* `data/integrated_bundle/astravive_unified_system_bundle.json`

---

## PHASE 5: THREE WAYS TO DEPLOY ASTRAVIVE FOR JUDGES

You can deploy ASTRAVIVE using any of the three production options below depending on your hosting preference:

### Option A: Zero-Backend Static Web Deployment (GitHub Pages / Vercel / Netlify — Fastest, 60 Seconds)
Because **`astravive_app.html`** has its own self-contained in-browser ODE solver, SHAP calculator, and live cross-validation engine, it can be hosted as a static site with **zero server maintenance**:
1. Create a public GitHub repository named `astravive-spaceapps-2026`.
2. Copy `astravive_app.html` to `index.html` in the repository root:
   ```bash
   cp astravive_app.html index.html
   git init
   git add .
   git commit -m "Deploy ASTRAVIVE Mission-Health Digital Twin"
   git branch -M main
   git remote add origin https://github.com/<your-username>/astravive-spaceapps-2026.git
   git push -u origin main
   ```
3. On GitHub, go to **Settings $\rightarrow$ Pages**, select **Branch: `main` (`/root`)**, and click **Save**.
4. Your interactive NASA Mission Control console will be live at:
   `https://<your-username>.github.io/astravive-spaceapps-2026/`

---

### Option B: Full Python + Streamlit Cloud Deployment (Streamlit Community Cloud — Free)
This deploys both the **Python 10-stage backend** (`app.py` + `SQLite` + `Scikit-learn`) and the **9-tab interactive Mission Control UI**:
1. Push your repository (containing `app.py`, `astravive_app.html`, all `.py` modules, and `requirements.txt`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **Create app**, select your repository (`astravive-spaceapps-2026`), set **Main file path** to `app.py`, and click **Deploy!**.
4. To test locally before pushing, run:
   ```bash
   streamlit run app.py
   ```
   This opens `http://localhost:8501` with the Python backend sidebar and full 9-tab console.

---

### Option C: Containerized Docker Deployment (Hugging Face Spaces / AWS / GCP / Render)
If you want a reproducible container for cloud hosting, create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run verification suite during build to guarantee container integrity
RUN python astravive_test_suite.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run the container locally:
```bash
docker build -t astravive-digital-twin .
docker run -p 8501:8501 astravive-digital-twin
```

---

## PHASE 6: PRE-SUBMISSION CHECKLIST FOR NASA SPACE APPS JUDGES

Before submitting your project link to the NASA Space Apps portal, verify these 5 items:
1. **Live Demo Link Works:** Open your deployed URL in an incognito browser window and verify all 9 tabs switch smoothly.
2. **Test the Signature Digital Twin Simulator (Tab 4):** Move the Scenario B sliders (`Sleep Duration`, `Exercise Duration`, `Environmental CO2`) and verify the chart and radar plot update immediately.
3. **Run the In-Browser Verification Suite (Tab 9):** Click **"Execute Live Browser Test Suite"** in Tab 9 and confirm all 9 test domains show `PASS`.
4. **Verify Provenance & Non-Diagnostic Disclaimers:** Confirm the `PUBLIC DATA`, `MODEL PREDICTION`, and `SIMULATED DATA` badges and non-diagnostic disclaimers are visible in Tabs 4, 6, 7, and 8.
5. **Generate Your Final Submission Archive:** Run `python package_astravive_zip.py` to produce `ASTRAVIVE_NASA_SpaceApps_2026_Complete.zip` for your repository release assets.