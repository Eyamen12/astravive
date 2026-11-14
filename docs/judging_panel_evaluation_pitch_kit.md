> **Archived v2.0 document.** Kept for reference. Where it conflicts with `../AUDIT_REPORT.md` or `../README.md` (v2.1), the v2.1 documents and the code are authoritative.

# ASTRAVIVE — NASA SPACE APPS JUDGING EVALUATION & PITCH KIT

**Project Evaluated:** ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN  
**Evaluation Context:** Formal NASA Space Apps Challenge Global Judging Rubric & Pitch Preparation  
**Notice of Independence:** *This evaluation and pitch package is prepared for the NASA Space Apps Challenge 2026 competition. ASTRAVIVE is an independent open-science research prototype and does not claim NASA endorsement, clinical certification, or official agency affiliation.*

---

## PART 1: JUDGING PANEL EVALUATION & COACHING FEEDBACK

**Opening Assessment:** ASTRAVIVE stands out as an exceptionally rigorous, scientifically honest, and visually compelling biomedical mission-control prototype that directly tackles the small-sample challenge of spaceflight physiology through a clever hybrid mechanistic-ODE and machine-learning architecture. **Overall Proficiency Level: Outstanding.**

### 1.1 Evaluation Against the 9 Judging Criteria

| Criterion | Rating | Judging Panel Assessment |
| :--- | :--- | :--- |
| **1. Scientific Credibility** | **Outstanding** | Properly avoids deep learning overfitting on $N=4$ astronaut cohorts (`OSD-575`/`569`) by coupling published NASA HRP differential equations ($f_{\text{mech}}$) with leak-free `GroupKFold` and `TimeSeriesSplit` Scikit-learn models ($g_{\text{ML}}$). |
| **2. Innovation** | **Outstanding** | The **Dual-Scenario (Scenario A vs. Scenario B) Digital Twin Simulator** with explicit temporal provenance transitions (Sol 0 `PUBLIC DATA` $\rightarrow$ Sols 1–7 `MODEL PREDICTION` $\rightarrow$ Sols 8+ `SIMULATED DATA`) is a standout methodological innovation for hackathon digital twins. |
| **3. Technical Implementation** | **Outstanding** | Complete 10-stage Python pipeline (`astravive_data_pipeline.py`, `astravive_ml_pipeline.py`, `astravive_xai_engine.py`, `astravive_digital_twin_simulator.py`, `astravive_system_integrator.py`) paired with a zero-dependency-install HTML5 Mission Control console (`astravive_app.html`) and automated test suite (`astravive_test_suite.py`). |
| **4. Use of NASA Resources** | **Outstanding** | Deep, verified integration of **NASA OSDR (`OSD-575`, `OSD-569`, `OSD-51`, `OSD-195`)**, **NASA OSDR Environmental Data App (EDA)**, **NASA OSDR RadLab**, **NASA DONKI API**, and **NASA HRP Evidence Reports**. |
| **5. Data Integrity** | **Outstanding** | Exemplary 5-tier data classification (`MEASURED DATA`, `PUBLIC DATA`, `SIMULATED DATA`, `DERIVED DATA`, `MODEL PREDICTION`) with automated physiological bound checking and zero silent mixing of synthetic and real observations. |
| **6. User Experience (UX)** | **Good** | Authentic dark aerospace HUD (`#030712`), interactive SVG anatomical body map, and non-blocking custom modals. However, the 9-tab top navigation bar and dense mathematical formulas can feel overwhelming to non-technical judges at first glance. |
| **7. Demonstration Quality** | **Outstanding** | Includes live in-browser cross-validation (`runLiveMLBenchmark()`), live what-if sliders, one-click stress/SEP storm injection, and a live 9-domain browser verification test runner. |
| **8. Relevance to Challenge** | **Outstanding** | Directly addresses astronaut health monitoring, environmental stressor coupling ($\text{CO}_2$ and radiation), and countermeasure optimization for LEO, Artemis Lunar, and Mars transit missions. |
| **9. Communication & Storytelling** | **Developing** | While the technical documentation (`README.md`) is exhaustive and rigorous, the narrative currently reads like a dense systems-engineering manual rather than an emotionally engaging human story for a 3-minute pitch. |

---

### 1.2 Primary Growth Areas & Actionable Coaching Recommendations

To elevate ASTRAVIVE from a strong technical prototype to a Global Winner, focus on these three lowest-scoring criteria (**Communication & Storytelling**, **User Experience Simplification**, and **Mechanistic Uncertainty Calibration**):

* **Growth Area 1 — Communication & Storytelling (Translating Math into Human Impact):**
  * **Specific Text/System Example:** In `README.md` Section 8 (*Digital Twin Mathematical Assumptions*), you present four raw coupled difference equations (e.g., `S_{t+1} = clip(S_t + 1.15 * (24 - sleep_hrs)...)`) without an immediate real-world mission vignette showing how a Flight Surgeon uses this during an emergency.
  * **Why It Matters:** Space Apps judges have only 3 minutes to watch your pitch. Leading with differential equation coefficients before anchoring the human stakes risks losing generalist judges.
  * **Actionable Next Step:** How can you frame your opening 30 seconds around a concrete **"Sol 42 ISS/Mars Flight Surgeon Dilemma"** (e.g., a cabin $\text{CO}_2$ scrubber degradation coinciding with a high-workload EVA week) before revealing the hybrid ODE + ML engine as the solution?

* **Growth Area 2 — User Experience & Cognitive Load across the 9-Tab Console:**
  * **Specific Text/System Example:** In `astravive_app.html`, all 9 tabs (`1. Mission Overview` through `9. Settings`) carry equal visual weight in the top navigation bar, and **Tab 4 (Digital Twin Simulator)** simultaneously displays 10 parameter sliders, 2 KPI cards, a radar chart, a multi-regime line chart, an endpoint table, and 4 equation boxes.
  * **Why It Matters:** During a fast-paced judging review, judges may miss your signature **Scenario A vs. Scenario B Digital Twin Simulator** if they spend too much time exploring secondary tabs first.
  * **Actionable Next Step:** Consider grouping your presentation flow—and optionally adding a visual highlight badge on **Tab 4 (Digital Twin Simulator)** and **Tab 6 (Explainable AI)**—so a first-time user immediately sees a **"Guided 3-Step Flight Surgeon Workflow"** (*Step 1: Detect Anomaly in Tab 1/2 $\rightarrow$ Step 2: Explain Drivers in Tab 6 $\rightarrow$ Step 3: Optimize Countermeasures in Tab 4*).

* **Growth Area 3 — Scientific Nuance on Cross-Domain Generalization (Terrestrial Wearables $\rightarrow$ Microgravity):**
  * **Specific Text/System Example:** In `astravive_ml_pipeline.py` and `README.md` Section 7, the `RandomForestClassifier` and `HistGradientBoostingRegressor` models are trained on terrestrial $1g$ cohorts (`PhysioNet MMASH` $N=22$ and `UCI WESAD` $N=15$) and coupled with $0g$ mechanistic priors.
  * **Why It Matters:** Biomedical judges will appreciate that you did not fabricate astronaut wearable data, but they will scrutinize *covariate shift*—specifically, how cephalad fluid shifts in $0g$ alter baseline heart rate variability ($RMSSD$) compared to $1g$ terrestrial subjects.
  * **Actionable Next Step:** Can you explicitly highlight in your pitch and Tab 8 documentation thatStage 4's **Subject-Wise Pre-Flight Baseline Z-Score Normalization** (`normalize_and_engineer_features`) is the exact mathematical bridge that reduces $1g \rightarrow 0g$ domain shift by evaluating each astronaut relative to their own baseline rather than raw terrestrial population means?

---

## PART 2: 3-MINUTE PITCH PRESENTATION SCRIPT

**Target Duration:** 3 Minutes ($\sim 420\text{ words}$ at $140\text{ wpm}$)  
**Visual Aids:** Slide deck + 45-second Picture-in-Picture live recording of `astravive_app.html`.

### [0:00 – 0:35] The Hook & Problem Statement (35s)
> "Imagine you are the Flight Surgeon supporting a crew 42 days into a deep-space transit. Over the last 48 hours, habitat carbon dioxide has crept up to $3.8\text{ mmHg}$, a solar energetic particle storm is brewing on NASA DONKI, and your crew is sleeping only 6 hours a night to keep up with maintenance.  
>
> In spaceflight, physiological systems don't fail in isolation: elevated $\text{CO}_2$ dilates cerebral blood vessels, microgravity shifts fluid toward the head—elevating Spaceflight-Associated Neuro-Ocular Syndrome (SANS) risk—and sleep debt suppresses autonomic vagal recovery. Yet today, mission planners lack an integrated, open-science sandbox to test how changing a crew's schedule today alters their multi-system readiness 90 days from now."

### [0:35 – 1:15] Our Solution & The Data Honesty Breakthrough (40s)
> "Meet **ASTRAVIVE**—our Mission-Health Digital Twin prototype built to *Predict Health, Optimize Missions, and Protect Astronauts*.  
>
> When we audited NASA's Open Science Data Repository (OSDR), we uncovered a critical reality: while NASA OSDR provides open longitudinal blood, metabolic, and cytokine panels from commercial astronauts (`OSD-575` and `OSD-569`) alongside true ISS cabin $\text{CO}_2$ (`OSDR EDA`) and radiation dosimetry (`RadLab`), continuous astronaut wearable waveforms are privacy-restricted.  
>
> Rather than fabricating data or overfitting deep neural networks on $N=4$ astronauts, we engineered a **Hybrid Mechanistic + Machine Learning Architecture**. We anchor long-horizon bone, radiation, and fluid-shift trajectories in published NASA Human Research Program differential equations, and couple them with leak-free Scikit-learn models calibrated on open human bed-rest analogs (`OSD-51`) and continuous wearable cohorts (`PhysioNet MMASH` and `UCI WESAD`)."

### [1:15 – 2:15] Signature Feature Spotlight: Scenario A vs. B Digital Twin & XAI (60s)
> "Let's look at ASTRAVIVE in action. On our Mission Overview console, the Flight Surgeon clicks our interactive **Anatomical Digital Twin** to inspect five coupled physiological subsystems.  
>
> When an autonomic stress alert triggers, we switch to **Tab 6: Explainable AI**. Using exact SHAP additive attributions, the console shows *why* the alert fired—separating upward stressors like electrodermal activity and cabin $p\text{CO}_2$ from stabilizing vagal tone—while clearly reminding the user that statistical feature attribution is not biological causation or clinical diagnosis.  
>
> Next, in **Tab 4: Our Signature Digital Twin Simulator**, we compare **Scenario A (Baseline Conditions)** against **Scenario B (Modified Countermeasures)** side by side. By increasing sleep from 6 to 8 hours, expanding ARED resistive exercise to 120 minutes, and scrubbing cabin $p\text{CO}_2$ down to $2.0\text{ mmHg}$, we watch projected 90-Sol Composite Readiness jump from $68.4\%$ to $90.2\%$, while preserving femoral bone mineral density and cutting SANS risk."

### [2:15 – 3:00] Zero-Fabrication Verification & Closing (45s)
> "Every single data point in ASTRAVIVE carries one of five mandatory provenance tags—separating `PUBLIC DATA` anchors at Sol 0, `MODEL PREDICTIONS` across Sols 1 to 7, and `SIMULATED DATA` ODE rollouts from Sol 8 onward.  
>
> Even better, judges can open **Tab 7** and **Tab 9** in their browser right now to run live subject-wise `GroupKFold` cross-validation and execute our 9-point automated verification suite with zero hardcoded metrics.  
>
> Independent, transparent, and built on NASA Open Science—this is **ASTRAVIVE**. Thank you!"

---

## PART 3: 5-MINUTE TECHNICAL DEMONSTRATION WALKTHROUGH

| Timestamp | Active Screen / File | Technical Action & Talking Point |
| :--- | :--- | :--- |
| **0:00 – 0:45** | **Tab 1: Mission Overview** (`astravive_app.html`) | Show the top **Global System Integration Banner** (`ALL 10 PIPELINE STAGES INTEGRATED`). Switch the **Subject Twin Selector** between `I4-CREW-01 (OSD-575/569)` and `WISE-2005-BR (OSD-51)` to demonstrate live KPI updates. Click the **CNS/SANS** and **Femoral BMD** nodes on the SVG Anatomical Body Map to display the governing differential equations. |
| **0:45 – 1:30** | **Tab 2 (Health) & Tab 8 (Data & Methodology)** | Show the continuous wearable stream (`MMASH`/`WESAD`) alongside the discrete SpaceX Inspiration4 `OSD-569` CBC and `OSD-575` CMP tables. Switch to **Tab 8** and select `Stream 2: Wearable Stream + Artifact Cleaning Demo` in the **Stage 2–5 Ingestion Inspector** to show how a $295\text{ bpm}$ spike and $0.11\text{ s}$ ectopic RR interval are flagged and linearly imputed with a full audit trail. |
| **1:30 – 2:45** | **Tab 4: Digital Twin Simulator (Signature Feature)** | Walk through **Scenario A (Baseline)** vs. **Scenario B (Modified)**. Adjust the 5 model-supported sliders (*Sleep Duration*, *Recovery Duration*, *Workload Intensity*, *Exercise Duration*, and *Environmental $\text{CO}_2$*). Point out the vertical regime boundary at **Sol 7** separating `MODEL PREDICTION` from `SIMULATED DATA`. Switch the primary chart selector between *Composite Readiness*, *Femoral BMD $\Delta$*, and *SANS Risk Index*, then click **Export JSON**. |
| **2:45 – 3:30** | **Tab 5 (Environmental) & Header Stress Trigger** | Click the red **`INJECT STRESS / SPE EVENT`** button in the top bar. Show the non-blocking Flight Surgeon modal, then open **Tab 5** to show the injected **NASA DONKI `2024-05-10` X5.8 Solar Energetic Particle storm** (+14.8 mSv/day spike) and storm-shelter recommendation. |
| **3:30 – 4:15** | **Tab 6: Explainable AI (XAI)** | Inspect the **Global Feature Importance** bar chart (`mean(|SHAP|)`) and the **Individual Prediction Analysis** waterfall plot for the injected acute stress excursion. Drag the **Wrist EDA** and **Cabin $p\text{CO}_2$** sliders to show real-time local additivity ($\mathbb{E}[f(\mathbf{X})] + \sum \phi_j = f(\mathbf{x})$) and point out the prominent **Non-Causation** and **Non-Diagnostic** banners. |
| **4:15 – 5:00** | **Tab 7 (Model Performance) & Tab 9 (Settings)** | In **Tab 7**, change the random seed dropdown and click **`Run Live Cross-Validation`** to execute live `GroupKFold` and `TimeSeriesSplit` evaluation in the browser. Finally, open **Tab 9** and click **`Execute Live Browser Test Suite`** to show all 9 verification tests (`T1`–`T9`) passing live. |

---

## PART 4: PROJECT POSTER OUTLINE (STANDARD 4-COLUMN SCIENTIFIC LAYOUT)

```
+---------------------------------------------------------------------------------------------------+
| HEADER: ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN: PREDICT HEALTH. OPTIMIZE MISSIONS. PROTECT CREW. |
| Authors & Team | NASA Space Apps Challenge 2026 | Independent Open-Science Research Prototype     |
+------------------------+------------------------+------------------------+------------------------+
| COLUMN 1:              | COLUMN 2:              | COLUMN 3:              | COLUMN 4:              |
| MOTIVATION &           | 10-STAGE HYBRID        | SIGNATURE DIGITAL TWIN | ANTI-LEAKAGE ML, XAI   |
| VERIFIED NASA DATA     | SYSTEM ARCHITECTURE    | SIMULATOR (SCEN A vs B)| & VERIFICATION RESULTS |
|                        |                        |                        |                        |
| 1. Spaceflight Risks   | 1. Hybrid Formulation  | 1. Dual-Scenario Design| 1. Validation Rigor    |
|  - Bone loss: 1-1.5%/mo|  x_{t+1} = f_mech(...) |  - Scenario A: Baseline|  - Subject GroupKFold  |
|  - Plasma vol: -10-17% |    + g_ML(...) + eps_t |    (6h sleep, 3.8 mmHg |  - Chronological       |
|  - SANS + Cabin pCO2   |  - Why No Deep Learning|    pCO2, 45m exercise) |    TimeSeriesSplit     |
|  - 600 mSv Career Limit|    Small-N cohorts     |  - Scenario B: Modified|  - Zero Data Leakage   |
|                        |    (N=4 to 24) require |    (8h sleep, 2.0 mmHg |                        |
| 2. Open Data Audit     |    ODE priors + trees  |    pCO2, 120m ARED)    | 2. Explainable AI (XAI)|
|  - CAT A (Spaceflight):|    to prevent overfit. |                        |  - Global Mean |SHAP|  |
|    OSD-575/569 (I4 N=4)|                        | 2. Multi-Regime Labels |  - Local Waterfall     |
|    OSDR EDA (ISS CO2)  | 2. 10-Stage Pipeline   |  - Sol 0: PUBLIC DATA  |    f(x)=E[f(X)]+sum phi|
|    RadLab & DONKI API  |  [Diagram: Sources ->  |  - Sols 1-7: MODEL     |  - Explicit Non-Causal |
|  - CAT B (Analogs &    |   Ingestion -> Bounds  |    PREDICTION          |    & Non-Diagnostic    |
|    Wearables):         |   -> Z-Score -> Feature|  - Sols 8+: SIMULATED  |    Disclaimers         |
|    OSD-51 Bed Rest     |   -> Scikit-learn ->   |    ODE OUTPUT          |                        |
|    PhysioNet MMASH/CVES|   Forecast -> SHAP ->  |                        | 3. 9-Domain Test Suite |
|    UCI WESAD           |   Twin -> Dashboard]   | 3. Quantitative Impact |  - 100% Pass across    |
|  - CAT C (Mechanistic):|                        |  - Readiness: 68.4% -> |    Bounds, Imputation, |
|    NASA HRP Rate Priors| 3. 5-Tier Provenance   |    90.2% (+21.8% gain) |    ODE Math & Exports  |
+------------------------+------------------------+------------------------+------------------------+
| FOOTER: Open-Source Code, Reproducible Python Pipeline & Interactive HTML5 Console | QR Code Link |
+---------------------------------------------------------------------------------------------------+
```

---

## PART 5: SYSTEM ARCHITECTURE EXPLANATION (JUDGE-READY SUMMARY)

When judges ask *"How does your system architecture work under the hood?"*, use this concise 3-layer explanation:

1. **Layer 1 — Provenance-Locked Ingestion & Preprocessing (Stages 1–5 in `astravive_data_pipeline.py`):**
   * Ingests `.csv` and `.json` files from three strictly separated tiers: **Category A** (NASA OSDR `OSD-575`/`569` Inspiration4 clinical panels, `OSDR EDA` cabin $\text{CO}_2$, `RadLab` dosimetry, `DONKI` solar weather), **Category B** (`OSD-51` 60-day bed rest, `PhysioNet MMASH` 24h beat-to-beat $RR$/actigraphy/hormones, `UCI WESAD`), and **Category C** (NASA HRP rate constants).
   * Enforces physiological range gates (e.g., $HR \in [30, 220]\text{ bpm}$, $p\text{CO}_2 \in [0.1, 15.2]\text{ mmHg}$), filters ectopic cardiac beats ($>25\%$ jump), performs per-subject baseline Z-score normalization without future leakage, and stamps every value with one of 5 mandatory classifications (`MEASURED DATA`, `PUBLIC DATA`, `SIMULATED DATA`, `DERIVED DATA`, `MODEL PREDICTION`).

2. **Layer 2 — Hybrid Mechanistic ODE + Anti-Leakage Scikit-Learn Engine (Stages 6–9 in `astravive_ml_pipeline.py`, `astravive_xai_engine.py`, `astravive_digital_twin_simulator.py`):**
   * **Short-Horizon ML ($g_{\text{ML}}$):** Wraps `SimpleImputer`, `StandardScaler`, and Scikit-learn estimators (`LogisticRegression`, `RandomForest`, `HistGradientBoosting`, `IsolationForest`) inside leak-free pipelines evaluated via subject-wise `GroupKFold` and chronological `TimeSeriesSplit`.
   * **Long-Horizon Mechanistic State-Space ($f_{\text{mech}}$):** Evolves 5 coupled physiological subsystems (Sleep/Fatigue $S_t$, Autonomic $RMSSD_t$, Femoral $\text{BMD}_t$, Neuro-Ocular $\text{SANS}_t$, and Radiation $D_{\text{cum},t}$) across 30 to 900 Sols with $95\%$ Monte Carlo confidence bands.
   * **Explainable AI (Stage 8):** Decomposes every prediction into baseline expectation $\mathbb{E}[f(\mathbf{X})]$ plus signed feature attributions $\sum \phi_j$ with verified additivity ($<10^{-9}$ residual).

3. **Layer 3 — Fault-Tolerant Orchestrator & Interactive Mission Console (Stage 10 in `astravive_system_integrator.py` & `astravive_app.html`):**
   * Wraps every backend and frontend component in defensive error boundaries (`ComponentStatus` in Python and `safeExecute()` in JS). If an external API or library is unreachable in an offline environment, ASTRAVIVE logs an explicit `UNAVAILABLE_EXPLAINED` status rather than fabricating outputs.

---

## PART 6: STEP-BY-STEP LIVE DEMO SCRIPT (CLICK-BY-CLICK)

Use this exact checklist during live Q&A or booth judging:

1. **Pre-Flight Check (10 seconds before judges arrive):**
   * Open `astravive_app.html` in the browser.
   * Click **Tab 4 (`4. Digital Twin Simulator`)** and click **`Reset Scenarios`**, then return to **Tab 1 (`1. Mission Overview`)**.
2. **Action 1 — Anatomical Inspection (Tab 1):**
   * *Click:* The glowing amber node on the astronaut's femur (`Node 5: Femoral Neck / Lumbar BMD`).
   * *Say:* "Here in Tab 1, clicking any anatomical node reveals its exact differential equation and empirical anchor—in this case, `OSD-51` WISE-2005 Bed Rest and NASA HRP bone mineral density priors."
3. **Action 2 — Triggering a Spaceflight Emergency (Top Header $\rightarrow$ Tab 5):**
   * *Click:* The red **`INJECT STRESS / SPE EVENT`** button in the top-right header, then click **`Acknowledge`** on the modal and switch to **Tab 5 (`5. Environmental`)**.
   * *Say:* "We just injected a real historical Solar Energetic Particle storm from NASA DONKI (`2024-05-10 X5.8 Flare`). Notice how the radiation dose rate spikes on the chart and triggers a $35\text{ g/cm}^2$ storm-shelter recommendation."
4. **Action 3 — Auditing the AI Explanation (Tab 6):**
   * *Click:* **Tab 6 (`6. Explainable AI`)** and drag the **Wrist EDA** slider from `6.4 µS` down to `1.2 µS`, or click **`2. High Physical Workload (ARED)`**.
   * *Say:* "Watch what happens when high heart rate is accompanied by high actigraphy workload (`ENMO = 0.76 g`): our SHAP attribution turns green (`-φ_j`) for actigraphy because physical exercise contextualizes elevated heart rate and reduces false-positive psychological stress alerts."
5. **Action 4 — Running the Signature Scenario A vs. B Simulator (Tab 4):**
   * *Click:* **Tab 4 (`4. Digital Twin Simulator`)**. Drag Scenario B's **Environmental CO2** slider from `2.0 mmHg` up to `4.5 mmHg`, then back down to `1.8 mmHg`.
   * *Say:* "In real time, our coupled state-space equations recompute both the 90-Sol trajectory and the 5-axis subsystem radar plot, while keeping Sol 0 (`PUBLIC DATA`), Sols 1–7 (`MODEL PREDICTION`), and Sols 8+ (`SIMULATED DATA`) strictly separated."
6. **Action 5 — Proving Zero Fabrication Live (Tab 7 & Tab 9):**
   * *Click:* **Tab 7 (`7. Model Performance`)** $\rightarrow$ **`Run Live Cross-Validation`**, then **Tab 9 (`9. Settings`)** $\rightarrow$ **`Execute Live Browser Test Suite`**.
   * *Say:* "None of our metrics are hardcoded strings. Clicking these two buttons runs out-of-subject `GroupKFold` cross-validation and executes 9 automated runtime assertions right in the browser."

---

## PART 7: POTENTIAL JUDGES' QUESTIONS & DEFENSIBLE ANSWERS (Q&A DEFENSE)

### Q1 (Biomedical / Space Physiology Judge): *"NASA OSDR `OSD-575` and `OSD-569` only have $N=4$ astronauts from a 3-day commercial mission, and your continuous wearable datasets (`MMASH`, `WESAD`) were collected in $1g$ on Earth. How can you claim your model predicts astronaut health over a 180-day ISS or 900-day Mars mission?"*
* **Answer:** "That is the exact scientific constraint that shaped our entire architecture. We explicitly **refuse** to train supervised ML classifiers on $N=4$ astronauts, and we never extrapolate terrestrial wearable ML models beyond 7 Sols. Instead, our ML models (`RandomForest` and `HistGradientBoosting`) operate strictly on short-horizon autonomic residuals normalized via **per-subject pre-flight baseline Z-scores**—which mitigates $1g$-to-$0g$ baseline shift. Beyond Sol 7, trajectory evolution is governed by published NASA Human Research Program compartmental differential equations ($f_{\text{mech}}$) calibrated against 60-day Head-Down Tilt Bed Rest analogs (`OSD-51`, $N=24$) and meta-analytic astronaut cohorts ($N>300$), and every point from Sol 8 onward is transparently labeled as `SIMULATED DATA`."

### Q2 (AI / Machine Learning Judge): *"Why didn't you use Deep Learning—like LSTMs, Transformers, or Physics-Informed Neural Networks (PINNs) in PyTorch—for your digital twin?"*
* **Answer:** "We evaluated that directly in Section 0.2 of our architecture specification. In verified open spaceflight and analog cohorts, sample sizes range from $N=4$ (`OSD-575`) to $N=24$ (`OSD-51`) and $N=22$ (`MMASH`). Deep neural networks have thousands to millions of parameters and severely overfit small-sample physiological tables while producing uncalibrated long-horizon drift over a 900-Sol rollout. By combining explicit `NumPy` mechanistic differential equations with `Scikit-learn` histogram gradient boosting (`loss='quantile'` for $\alpha \in \{0.05, 0.50, 0.95\}$) and `IsolationForest`, we achieve fast ($<20\text{ ms}$) CPU execution, exact tree-based SHAP explainability, bounded 900-Sol asymptotic stability, and better out-of-subject generalization under `GroupKFold`."

### Q3 (Data Engineering / Integrity Judge): *"How do you prevent data leakage in time-series wearable data, and how do you handle corrupted sensor spikes?"*
* **Answer:** "We enforce three anti-leakage guardrails in `astravive_data_pipeline.py` and `astravive_ml_pipeline.py`:
  1. **Per-Subject Baseline Locking:** Z-score normalization uses only the initial 25% baseline calibration window of each subject so future shifts never leak backward.
  2. **In-Fold Scaling & Splitting:** `SimpleImputer` and `StandardScaler` are wrapped inside a `scikit-learn` `Pipeline` evaluated with `GroupKFold(groups=subject_id)` for classification and strictly causal past lags (`t-1, t-2, t-3`) with `TimeSeriesSplit` for forecasting.
  3. **Transparent Artifact Gatekeeping:** Out-of-bounds spikes (like $HR > 220\text{ bpm}$ or ectopic $RR$ jumps $>25\%$) are flagged in `row_invalid_flag`, logged in `transformation_audit_trail`, and linearly interpolated per subject."

### Q4 (Flight Operations / Medical Ethics Judge): *"Could a crew member or mission controller misinterpret your SHAP plots as proving that a specific factor caused a medical condition?"*
* **Answer:** "We built explicit safeguards against both causal misinterpretation and clinical misuse. Every SHAP view in **Tab 6** and every JSON export from `astravive_xai_engine.py` carries two mandatory banners: first, that SHAP attributions ($\phi_j$) measure statistical association within the model rather than biological causation; and second, that ASTRAVIVE is an exploratory research prototype—not an FDA-cleared or NASA-certified medical diagnostic tool—designed to support, never replace, qualified Flight Surgeons."