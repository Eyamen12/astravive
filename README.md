# 🚀 ASTRAVIVE v2.1 — Mission-Health Digital Twin

**NASA Space Apps Challenge 2026** · *Predict Health. Optimize Missions. Protect Astronauts.*

---

## 🎯 Try It Now

> 👉 **[Open Live Dashboard](https://YOUR_USERNAME.github.io/ASTRAVIVE/)** — no install needed, runs entirely in your browser
>
> 📊 **[View Presentation Slides](ASTRAVIVE_Presentation.html)** — 7-slide project overview

---

## The Problem We Solve

On Day 42 of a Gateway mission, Commander Chen wakes with blurred vision. It could be the start of **SANS** — Spaceflight-Associated Neuro-Ocular Syndrome, the leading risk for future crewed missions. Her Flight Surgeon on Earth has a 1.3-second signal delay and won't see the first clinical sign for 72 hours.

**What if the computer already knew — 18 hours before symptoms appeared?**

ASTRAVIVE is a browser-native astronaut health digital twin that pairs **mechanistic ODE-based physiology** (sleep pressure, bone density, cardiovascular fitness, SANS/ICP risk, radiation dose) with **explainable machine learning** and **Monte Carlo forecasting** — built entirely on **NASA open-science data** (OSDR, RadLab, DONKI).

It empowers Flight Surgeons and mission planners to:
- 🔍 **Detect** early health decline across 5 physiological subsystems
- 📈 **Forecast** multi-horizon trajectories with 95% confidence intervals
- ⚖️ **Compare** countermeasure strategies in real-time (Scenario A vs. B simulator)
- 🧠 **Explain** every prediction with SHAP feature attributions
- ✅ **Verify** every metric live in-browser (11-check verification suite)

> **Not a medical device.** This is an exploratory research prototype. Unless you load your own external datasets, every number comes from simulation or synthetic demo data. The UI and the exported bundle say so (`data_mode: SYNTHETIC_DEMO`).

---

## NASA Data & Resources Used

| Source | Dataset ID | What It Provides |
|---|---|---|
| NASA OSDR | OSD-575 | Inspiration4 crew wearable physiology (N=4) |
| NASA OSDR | OSD-569 | Inspiration4 blood/CBC clinical data |
| NASA OSDR | OSD-51 | WISE-2005 Head-Down Tilt Bed Rest analog (N=24) |
| NASA OSDR | OSD-195 | Additional spaceflight analog data |
| NASA OSDR EDA | — | Environmental sensor data (CO₂, temperature, humidity) |
| NASA OSDR RadLab | — | Radiation exposure data and dosimetry |
| NASA DONKI | API | Space weather events (solar particle events, CMEs) |
| NASA-STD-3001 | — | Career radiation dose limit (600 mSv) |
| NASA HRP | Evidence Reports | Human Research Program physiological models |

**Scientific References:**
- Zeitlin et al. 2013, *Science* — MSL-RAD Mars transit dose (1.84 ± 0.30 mSv/day) — [DOI: 10.1126/science.1235989](https://doi.org/10.1126/SCIENCE.1235989)
- Zhang et al. 2020, *Science Advances* — Chang'E-4 lunar surface dose (~1.37 mSv/day) — [DOI: 10.1126/sciadv.aaz1334](https://www.science.org/doi/10.1126/sciadv.aaz1334)
- JAXA ISS Space Medicine — LEO dose rate (0.5–1 mSv/day) — [Reference](https://iss.jaxa.jp/med/health_promotion.pdf)

---

## Files

| File | Role |
|---|---|
| `astravive_app.html` | Self-contained dashboard (9 ARIA tabs, in-browser twin, Monte Carlo forecast, exact Shapley, LOSO benchmark, CSV/JSON ingestion, 11-check verification suite) |
| `app.py` | Streamlit entry point. Runs the integrator and embeds the HTML |
| `astravive_data_pipeline.py` | Ingestion, physiological bounds, imputation, provenance tags |
| `astravive_ml_pipeline.py` | Subject-wise GroupKFold classifiers, chronological forecasters, held-out anomaly detector |
| `astravive_xai_engine.py` | SHAP TreeExplainer, with permutation and exact-baseline Shapley fallbacks |
| `astravive_digital_twin_simulator.py` | Scenario A/B mechanistic twin (equations in the docstring match the code; see test_07) |
| `astravive_system_integrator.py` | Runs every stage and writes the system bundle with safe error summaries |
| `astravive_test_suite.py` | 9 verification tests |
| `package_astravive_zip.py` | Compiles, scans for secrets and forbidden patterns, writes a SHA-256 manifest, builds the ZIP |
| `requirements.txt` | Pinned dependency ranges |
| `data/`, `models/` | Example processed stream, scenario output, model bundle and XAI explanations |
| `docs/` | Archived v2.0 documents (superseded where they conflict with v2.1) |
| `AUDIT_REPORT.md` | Every finding with severity, fix and verification |
| `STEP_BY_STEP_GUIDE.md` | Setup → test → package → run → deploy |
| `CHANGELOG.md` | v2.0 → v2.1 changes |

### Upload → canonical name map

| Uploaded as | Canonical |
|---|---|
| `astravive_streamlit_web_server_entry_point.py` | `app.py` |
| `astravive_data_ingestion_preprocessing_pipeline-1.py` | `astravive_data_pipeline.py` |
| `astravive_reproducible_ml_pipeline-1.py` | `astravive_ml_pipeline.py` |
| `astravive_explainable_ai_engine-1.py` | `astravive_xai_engine.py` |
| `astravive_digital_twin_simulator_engine(-1).py` (identical copies) | `astravive_digital_twin_simulator.py` |
| `astravive_unified_system_orchestrator_integration_engine(-1).py` (identical copies) | `astravive_system_integrator.py` |
| `astravive_automated_verification_test_suite.py` | `astravive_test_suite.py` |
| `astravive_project_zip_packager_syntax_validator.py` | `package_astravive_zip.py` |
| `python_dependencies.txt` | `requirements.txt` |
| `code-astravive.txt` | `astravive_app.html` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python astravive_test_suite.py          # expect: Ran 9 tests ... OK
streamlit run app.py
```

You can also open `astravive_app.html` directly in a browser. Then go to Settings & QA → Run verification suite (expect 11/11 PASS). See `STEP_BY_STEP_GUIDE.md` for details.

## Architecture

```
[1. DATA SOURCES] → [2. INGESTION] → [3. VALIDATION] → [4. PREPROCESSING]
       → [5. FEATURE ENGINEERING] → [6. AI/ML ENGINE] → [7. FORECASTING]
       → [8. XAI (SHAP)] → [9. DIGITAL TWIN] → [10. DASHBOARD]
```

| Component | Technology |
|---|---|
| ML Models | scikit-learn (RandomForest, HistGradientBoosting, IsolationForest) with GroupKFold |
| XAI | TreeExplainer SHAP → Permutation → exact baseline Shapley (chain of fallbacks) |
| Digital Twin | Coupled ODEs: sleep pressure, bone density, cardiovascular fitness, SANS/ICP, radiation |
| Dashboard | Self-contained HTML5, 9 ARIA tabs, WCAG 2.2 AA, zero external dependencies at runtime |
| Backend | Python 3.10+ / Streamlit / Plotly |

## Environment dose anchors (used by the twin)

| Environment | Dose-equivalent rate | Source |
|---|---|---|
| Deep space / Mars transit | 1.84 ± 0.30 mSv/day | MSL-RAD cruise, Zeitlin et al. 2013, Science ([paper](https://doi.org/10.1126/SCIENCE.1235989)) |
| Lunar surface | ~1.37 mSv/day | Chang'E-4 LND, Zhang et al. 2020, Sci. Adv. ([paper](https://www.science.org/doi/10.1126/sciadv.aaz1334)) |
| ISS (LEO) | ~0.5–1 mSv/day (0.5 used) | [JAXA space medicine](https://iss.jaxa.jp/med/health_promotion.pdf) |
| Career limit | 600 mSv effective dose | NASA-STD-3001 ([NASA OCHMO brief](https://www.nasa.gov/wp-content/uploads/2023/03/radiation-protection-technical-brief-ochmo.pdf)) |

## Honest limitations

- With synthetic data, the classifiers reach AUC 1.0 because the demo labels are a deterministic function of the features. That shows the pipeline works, not that the model performs.
- For HR forecasting, naive persistence (MAE 4.67) beats every learned model on the synthetic stream. The selector only compares learned models, and the benchmark table shows this.
- The twin coefficients are hand-set, literature-inspired values, not fitted to crew data. `g_ML` (an ML correction term) is documented but not applied.
- The in-browser twin matches the Python twin within |Δ| ≤ 4.1e-3. It is a port, so keep both in sync when you edit the equations.

## Technologies & Licenses

| Library | Version | License |
|---|---|---|
| Streamlit | ≥1.38.0 | Apache 2.0 |
| scikit-learn | ≥1.5.1 | BSD-3 |
| SHAP | ≥0.46.0 | MIT |
| pandas / numpy / scipy | latest stable | BSD-3 |
| three.js | r149 | MIT |
| Chart.js | 4.4.1 | MIT |
| Font Awesome | 6.5.1 | SIL OFL 1.1 + MIT |
| Plotly | ≥5.24.0 | MIT |

## License

This project is an open-science research prototype developed for the NASA Space Apps Challenge 2026. All source code is provided for educational and research purposes.
