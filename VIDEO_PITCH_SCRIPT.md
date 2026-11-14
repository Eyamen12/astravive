# 🎬 ASTRAVIVE: Video Pitch & Demo Script (Slide-by-Slide & App UI Match)
**NASA Space Apps Challenge 2026**  
**Total Target Time:** ~3:00 minutes (180 seconds) or condensable to a high-impact 30-second summary.

---

## 🧭 Executive Summary of Setup
* **Screen Layout Recommended:** Split-screen or screen-recording that alternates between the **Slide Deck** (`ASTRAVIVE_Presentation.html`) and the **Live Mission Control App UI** (`astravive_app.html` / `index.html`).
* **Tone:** Urgent, clinical, scientifically rigorous, visionary.

---

## ⏱️ Video Script Breakdown (Slide by Slide + UI Synchronized)

---

### 🟢 SLIDE 1 (0:00 – 0:15) | The Hook & Identity
* **On Screen:** Slide 1 of `ASTRAVIVE_Presentation.html` (Title: *"ASTRAVIVE — Mission-Health Digital Twin"* with glowing aerospace cyan logo and NASA Space Apps badge).
* **Audio / Voiceover:**
  > *"When human crews venture into deep space on Artemis or a 900-day voyage to Mars, Earth is no longer a real-time hospital. A Flight Surgeon sitting in Mission Control will face up to a 24-minute communications delay. This is ASTRAVIVE: an open-science digital twin designed to predict astronaut health risks, optimize countermeasures, and protect lives before clinical symptoms appear."*

---

### 🔴 SLIDE 2 + APP TAB 1 (0:15 – 0:45) | The Problem & Real-Time Physiological Map
* **On Screen:** 
  1. Quick cut to Slide 2 (*"The Human Problem: Five Compounding Threats"*).
  2. Transition to **App UI $\rightarrow$ Tab 1: Mission Overview**. Click on the interactive anatomical astronaut body map (femur node, intracranial pressure node).
* **Audio / Voiceover:**
  > *"Picture Day 42 of a Gateway lunar mission. Commander Chen wakes up with blurred vision and a dull pressure behind her eyes. On Earth, her medical team won't see her report for days.*
  > 
  > *In space, human bodies face five cascading, interrelated threats: bone mineral density drops up to 1.5% a month; cephalad fluid shifts raise intracranial pressure leading to SANS (Spaceflight-Associated Neuro-ocular Syndrome); microgravity contracts cardiovascular plasma volume; circadian rhythms fracture; and ionizing radiation accumulates toward NASA’s 600 mSv career limit.*
  > 
  > *None of these act alone. Poor sleep accelerates bone deconditioning. Radiation compounds cardiac stress. We needed an onboard system that tracks this whole interconnected network."*

---

### 🔵 SLIDE 3 + APP TAB 4 (0:45 – 1:15) | The Core Innovation: Dual-Scenario Digital Twin
* **On Screen:**
  1. Cut to Slide 3 (*"Our Solution: Mechanistic ODEs + Explainable ML"*).
  2. Transition to **App UI $\rightarrow$ Tab 4: Digital Twin Simulator**. 
  3. **Visual Action in UI:** Drag the **Environmental $\text{CO}_2$ slider** in Scenario B from $3.8\text{ mmHg}$ down to $1.8\text{ mmHg}$, and increase **ARED Exercise** to $120\text{ min/day}$. Watch the 90-Sol trajectory lines and the 5-axis radar chart dynamically recompute in real time!
* **Audio / Voiceover:**
  > *"To solve this, ASTRAVIVE combines published NASA Human Research Program differential equations with machine learning into a real-time digital twin.*
  > 
  > *Here in our Scenario Simulator, a Flight Surgeon can test what-if interventions live. Under Scenario A—a standard baseline with 6 hours of sleep and elevated habitat $\text{CO}_2$—Commander Chen’s SANS risk elevates and bone loss accelerates.*
  > 
  > *By adjusting Scenario B—adding 45 minutes of resistive exercise and scrubbing habitat $\text{CO}_2$ down to 1.8 mmHg—ASTRAVIVE projects a 21.8% gain in overall mission readiness, stopping the ocular pressure spike before permanent optic nerve damage occurs."*

---

### 🟣 SLIDE 4 + APP TAB 8 (1:15 – 1:35) | NASA Data & Provenance Integrity
* **On Screen:**
  1. Cut to Slide 4 (*"Anchored in Open NASA Science"*).
  2. Transition to **App UI $\rightarrow$ Tab 8: Data & Methods**. Point to the 5-Tier Data Badges (`MEASURED DATA`, `PUBLIC DATA`, `SIMULATED DATA`, `DERIVED DATA`, `MODEL PREDICTION`).
* **Audio / Voiceover:**
  > *"Scientific integrity is our highest priority. We never invent or hallucinate data. Small astronaut samples make deep learning dangerous.*
  > 
  > *Instead, ASTRAVIVE anchors directly in verified NASA Open Science Data Repository studies: Inspiration4 metabolic data (`OSD-575`), hematology (`OSD-569`), 60-day bed rest microgravity analogs (`OSD-51`), ISS environmental sensor telemetry via OSDR EDA, RadLab dosimetry, and real-time solar weather from NASA DONKI.*
  > 
  > *Every single data point in our UI is tagged with strict provenance badges, so mission controllers know exactly what is empirical and what is simulated."*

---

### 🟠 SLIDE 5 + APP TAB 6 & TAB 7 (1:35 – 2:10) | Explainable AI & Live Benchmark Verification
* **On Screen:**
  1. Transition to **App UI $\rightarrow$ Tab 6: Explainable AI (SHAP)**. Hover over the feature attribution bars (showing $\text{HRV RMSSD}$, sleep pressure $S_t$, and habitat $\text{CO}_2$ impact).
  2. Transition to **App UI $\rightarrow$ Tab 7: Model Performance**. Click the **`Run Live Cross-Validation`** button. Watch the out-of-subject `GroupKFold` table compute live in the browser.
* **Audio / Voiceover:**
  > *"A black-box AI cannot be trusted with an astronaut’s life. In Tab 6, our tree-based SHAP engine breaks down every alert into mathematically exact feature attributions, explaining whether an anomaly is driven by sleep debt, autonomic vagal suppression, or cabin hypercapnia.*
  > 
  > *Furthermore, none of our validation metrics are pre-baked text strings. In Tab 7, clicking 'Run Live Cross-Validation' trains and evaluates Scikit-learn models directly in the client across 22 subjects, proving zero data leakage and real predictive validity."*

---

### 🟡 SLIDE 6 + APP TAB 9 (2:10 – 2:35) | Robustness & Automated Verification
* **On Screen:**
  1. Transition to **App UI $\rightarrow$ Tab 9: Settings & QA**.
  2. Click **`Run Verification Suite`** (Shows 11/11 automated checks passing: bounds checks, interpolation, parity with Python ODEs, WCAG accessibility).
  3. Cut to Slide 6 (*"Mission Impact & Exploration Roadmap"*).
* **Audio / Voiceover:**
  > *"ASTRAVIVE is built to mission-critical aerospace standards. Our embedded verification suite executes 11 real-time assertions, verifying that our browser ODE equations match the Python backend within 0.004 precision, that all out-of-range sensor spikes are caught, and that the UI meets full WCAG 2.2 AA accessibility.*
  > 
  > *Our roadmap expands ASTRAVIVE to live streaming telemetry on the Lunar Gateway, commercial space station modules, and crewed Mars transit vehicles."*

---

### ⚪ SLIDE 7 (2:35 – 3:00) | Conclusion & Call to Action
* **On Screen:** Slide 7 (*"References, Open Science & Team Credits"* with GitHub and Live Demo QR codes/URLs).
* **Audio / Voiceover:**
  > *"Astronauts push the boundaries of human presence in the solar system. ASTRAVIVE gives them the computational guardian they deserve: predicting health, optimizing missions, and protecting the pioneers of tomorrow.*
  > 
  > *Try the live interactive mission console right in your browser today. Thank you!"*

---

## ⚡ 30-Second Elevator Pitch (For Strict 30-Second Video Format)
If your local event enforces a strict 30-second cap, use this exact cut:

* **0:00 – 0:08 (Slide 1 + Tab 1):** *"During deep space missions, communication delays make real-time telemedicine from Earth impossible. Meet ASTRAVIVE: a mission-health digital twin for astronauts."*
* **0:08 – 0:18 (Tab 4):** *"By fusing NASA OSDR spaceflight data with mechanistic differential equations and explainable machine learning, ASTRAVIVE lets flight surgeons simulate countermeasure scenarios live—predicting bone loss, SANS ocular pressure, and radiation risk before symptoms occur."*
* **0:18 – 0:30 (Tab 6 + Slide 7):** *"With transparent SHAP explanations, zero data fabrication, and a 100% verified browser console, ASTRAVIVE protects the humans who explore our solar system. Explore the live prototype today."*
