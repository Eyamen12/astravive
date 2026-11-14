> **Archived v2.0 document.** Kept for reference. Where it conflicts with `../AUDIT_REPORT.md` or `../README.md` (v2.1), the v2.1 documents and the code are authoritative.

# ASTRAVIVE — DATASET DOWNLOAD, TRAINING & PROVENANCE GUIDE

**Project:** ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN  
**Target Event:** NASA Space Apps Challenge 2026  

---

## 1. How Many Datasets Exist, and Do You Have to Download Them?

There are **12 verified datasets** cataloged in ASTRAVIVE's research registry and Tab 8 of the application. 

**Important Engineering Note:** **You do not need to manually download any datasets to run the ASTRAVIVE prototype.** 
ASTRAVIVE features an **offline-first ingestion architecture with automated synthetic fallback**. When raw individual-level files are offline, restricted (such as raw human genomic reads under IRB), or not present locally, the pipeline (`astravive_data_pipeline.py`) and application automatically generate clearly labeled **SIMULATED DATA** calibrated to published open summaries. This ensures the application runs instantly on any machine with zero manual setup. 

However, if you wish to inspect or run offline analyses using the official raw files, you can download them directly using the links provided in Section 3.

---

## 2. Do You Have to Train Using Them?

Yes, but with a strict functional separation between **Training Datasets** and **Empirical Anchor / Stressor Datasets**:

1. **Machine Learning Training & Cross-Validation Datasets (`MMASH` & `WESAD`):**
   * The ML pipeline (`astravive_ml_pipeline.py`) trains Scikit-learn models (`IsolationForest`, `RandomForestClassifier`, and `HistGradientBoostingRegressor`) using continuous wearable streams from **PhysioNet MMASH** ($N=22$) and **UCI WESAD** ($N=15$). These datasets provide the high-frequency autonomic ($\text{HR}$, $\text{RMSSD}$), electrodermal ($\text{EDA}$), and actigraphy ($\text{ENMO}$) features required to train acute stress classifiers and short-term forecasting models under leak-free subject-wise `GroupKFold` cross-validation.

2. **Empirical Clinical Anchors & Environmental Telemetry (`OSD-575`, `OSD-569`, `EDA`, `RadLab`, `DONKI`):**
   * Spaceflight clinical datasets (`OSD-575` Comprehensive Metabolic Panel and `OSD-569` Complete Blood Count from the SpaceX Inspiration4 mission) and environmental feeds (`OSDR EDA` cabin $\text{CO}_2$, `RadLab` dosimetry, and `DONKI` solar storm APIs) are **not used to train machine-learning classifiers** (because $N=4$ astronauts is too small for supervised training without severe overfitting). Instead, they serve as:
     * **Sol 0 Empirical Baseline Anchors** for the digital twin simulator.
     * **Environmental Stressor Vectors** ($\mathbf{e}_t$) driving hypercapnic SANS and radiation compartmental equations.

---

## 3. Direct Access Links to All 12 Verified Datasets

Below are the official sources and direct access links for all 12 datasets cataloged in ASTRAVIVE:

### Category A — Actual Spaceflight & Astronaut Datasets
1. **SpaceX Inspiration4 Comprehensive Metabolic Panel & Cytokines (`OSD-575`)**
   * *Official Source:* NASA Open Science Data Repository (OSDR) / ALSDA
   * *Direct Link:* [NASA OSDR OSD-575 Repository](https://osdr.nasa.gov/bio/repo/data/studies/OSD-575)
2. **SpaceX Inspiration4 Complete Blood Count Hematology (`OSD-569`)**
   * *Official Source:* NASA OSDR / GeneLab (`GLDS-561`)
   * *Direct Link:* [NASA OSDR OSD-569 Repository](https://osdr.nasa.gov/bio/repo/data/studies/OSD-569)
3. **SpaceX Inspiration4 Multi-Omics & Urine Inflammation (`OSD-656` / SOMA)**
   * *Official Source:* NASA OSDR & Weill Cornell SOMA Portal
   * *Direct Link:* [NASA OSDR Inspiration4 Mission Umbrella](https://osdr.nasa.gov/bio/repo/data/missions/SpaceX%20Inspiration4) | [SOMA Portal](https://soma.weill.cornell.edu)
4. **NASA OSDR Environmental Data Application (EDA) — ISS Habitat Telemetry**
   * *Official Source:* NASA OSDR Visualization Portal
   * *Direct Link:* [NASA OSDR EDA Portal](https://visualization.osdr.nasa.gov/eda/)
5. **NASA OSDR RadLab — Spaceflight Radiation Dosimetry**
   * *Official Source:* NASA OSDR RadLab Portal
   * *Direct Link:* [NASA OSDR RadLab GUI](https://visualization.osdr.nasa.gov/radlab/gui/overview)
6. **NASA DONKI — Space Weather Notifications (Solar Flares & SEPs)**
   * *Official Source:* NASA Goddard / Community Coordinated Modeling Center (CCMC)
   * *Direct Link:* [NASA API Portal (DONKI)](https://api.nasa.gov/)

### Category B — Terrestrial Analogs, Wearables & Physiological Datasets
7. **WISE-2005 60-Day Head-Down Tilt Bed Rest Study (`OSD-51`)**
   * *Official Source:* NASA OSDR (ESA/NASA/CSA/CNES joint study)
   * *Direct Link:* [NASA OSDR OSD-51 Repository](https://osdr.nasa.gov/bio/repo/data/studies/OSD-51)
8. **MMASH — Multilevel Monitoring of Activity and Sleep in Healthy People**
   * *Official Source:* PhysioNet (University of Pisa)
   * *Direct Link:* [PhysioNet MMASH v1.0.0](https://physionet.org/content/mmash/1.0.0/)
9. **WESAD — Wearable Stress and Affect Detection Dataset**
   * *Official Source:* UCI Machine Learning Repository
   * *Direct Link:* [UCI WESAD Dataset 465](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
10. **Sleep-EDF Database Expanded (`sleep-edf-x`)**
    * *Official Source:* PhysioNet (Leiden University Medical Center)
    * *Direct Link:* [PhysioNet Sleep-EDFx v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)
11. **PhysioNet Cerebrovascular & Orthostatic Tilt Responses (`cves`)**
    * *Official Source:* PhysioNet (BIDMC / Harvard Medical School)
    * *Direct Link:* [PhysioNet CVES Dataset](https://physionet.org/content/cves/1.0.0/)

### Category C — Mechanistic Reference Frameworks
12. **NASA Human Research Program (HRP) Evidence Reports & Rate Constants**
    * *Official Source:* NASA Human Research Roadmap & NTRS
    * *Direct Link:* [NASA Human Research Roadmap](https://humanresearchroadmap.nasa.gov/) | [NASA Technical Reports Server (NTRS)](https://ntrs.nasa.gov/)