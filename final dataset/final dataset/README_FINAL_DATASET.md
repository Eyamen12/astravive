# 🚀 ASTRAVIVE: Curated Final Dataset for AI Model Training
**NASA Space Apps Challenge 2026**

This directory contains the cleaned, feature-engineered, and structured datasets ready for direct consumption by the **ASTRAVIVE AI & Mechanistic Digital Twin Pipeline**.

---

## 📁 Directory Structure & Domain Mapping

```
final dataset/
├── 00_FINAL_DATASET_CATALOG.csv           <-- Full index of all datasets, row counts, and sizes
├── MODEL_TRAINING_GUIDE.md               <-- Step-by-step model training architecture
├── README_FINAL_DATASET.md               <-- This documentation file
├── download_and_scrape_supplementary.py  <-- Python scraper for supplementary WESAD/CVES/DONKI
├── download_and_scrape_supplementary.ps1 <-- PowerShell downloader for automated web fetch
├── Nasa_Data_All_50_Processed_CSVs.zip   <-- Single-archive bundle of all datasets
│
├── 01_Autonomic_Stress_Wearables/        <-- Wearable Autonomics & HRV ML Training (N=22)
├── 02_Spaceflight_MultiOmics/             <-- Inspiration4 Multi-Omics Empirical Anchors (N=4)
├── 03_Microgravity_Analog_BedRest/       <-- OSD-51 60-Day Bed Rest Microarray (N=24)
├── 04_Space_Environment_And_Radiation/  <-- ISS Telemetry, RadLab & DONKI Space Weather
├── 05_Sleep_Architecture_And_Staging/   <-- SleepEDFx 197 PSG Recordings & 30s Staging
└── 06_Digital_Twin_Simulations/          <-- 90-Sol Deep-Space Flight Scenarios
```

---

## 🧠 AI / ML Model Training Breakdown

| Model / Subsystem | Target / Task | Key Features | Source Dataset(s) | Folder |
| :--- | :--- | :--- | :--- | :--- |
| **A. Autonomic Stress Classifier** | Binary Acute Stress ($0=\text{Nominal}, 1=\text{Stress}$) | HR, RMSSD, SDNN, pNN50, Steps, Posture, EDA | `07_MMASH_22Users_5Min_Epochs_ASTRAVIVE_ML_Training_6066Rows.csv` | `01_Autonomic_Stress_Wearables/` |
| **B. Residual Forecasting** | Next-window Heart Rate ($t+1$) & Vagal Tone residual | Autoregressive lags ($t-1, t-2, t-3$), Borbély sleep pressure | `06_MMASH_22Users_Hourly_Wearable_HRV_Training_Dataset.csv` | `01_Autonomic_Stress_Wearables/` |
| **C. Sensor Anomaly Detector** | Isolation Forest on out-of-distribution biometric jumps | Multimodal biometric channels, Movement Index | `05_MMASH_22Users_Master_Multimodal_Training_Dataset.csv` | `01_Autonomic_Stress_Wearables/` |
| **D. Sleep Staging Model** | 5-Stage Sleep Classification (Wake, N1, N2, N3, REM) | 30-sec EEG spectral powers, EMG tone, EOG | `13_SleepEDFx_30Sec_Epochs_Sleep_Staging_ML_Dataset.csv` | `05_Sleep_Architecture_And_Staging/` |
| **E. Environmental Stressors** | Forcing functions $p\text{CO}_2(t)$, Radiation $\dot{D}(t)$ | $\text{CO}_2$ ppm, Dose rate ($\mu\text{Gy/h}$), Flare flux | `08_NASA_DONKI_Solar_Flares_CME_SEP_Events_Full_631Rows.csv`<br>`09_OSDR_EDA_RadLab_12087Rows_ASTRAVIVE_Processed_Dataset.csv` | `04_Space_Environment_And_Radiation/` |
| **F. Mechanistic Calibration** | Bone mineral density & muscle atrophy ODE loss rates | Microarray gene expression fold-changes, CMP | `04_OSD51_BedRest_SkeletalMuscle_Training_Dataset.csv`<br>`01_OSD575_Unified_Wide_Training_Dataset.csv` | `02_Spaceflight_MultiOmics/`<br>`03_Microgravity_Analog_BedRest/` |

---

## 🗑️ Clean-up Actions Performed (Unneeded Files Removed)

1. **`combined_dataset.csv` (333 MB) — DELETED**:
   * *Reason:* This was a raw, unaligned concatenation of 153 heterogeneous CSV files resulting in over 90% missing/NaN entries and mismatched column types. It is not suitable for machine learning.
2. **`ALL file/` (99 MB, 153 loose CSVs) — DELETED**:
   * *Reason:* These individual per-user fragments (`Actigraph_*.csv`, `Activity_*.csv`) have already been consolidated, cleaned, and feature-engineered into the structured files in `01_Autonomic_Stress_Wearables/`.
3. **`Processed_Training_CSV_Datasets/` — DELETED**:
   * *Reason:* Legacy intermediate directory containing duplicate copies of the processed datasets.

---

## 🌐 Supplementary Datasets (Web Scraping & Live Fetch)

If you wish to download additional raw datasets directly from the web outside the sandbox:
* Run `download_and_scrape_supplementary.py` (Python) or `download_and_scrape_supplementary.ps1` (PowerShell) located in this folder.
* **WESAD (2.1 GB):** UCI ML Repository (`https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection`).
* **PhysioNet CVES:** Cerebrovascular and Tilt (`https://physionet.org/content/cves/1.0.0/`).
* **NASA DONKI Live API:** Real-time space weather from `https://api.nasa.gov/`.
