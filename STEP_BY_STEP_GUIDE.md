# ASTRAVIVE v2.1: Step-by-Step Guide

Each step lists what to run and what you should see. Commands assume a terminal opened in the project folder (the one with `app.py`).

## Step 1: Requirements

- Python 3.10 or newer (tested on 3.14) and a modern browser (Chrome, Edge, Firefox or Safari).
- An internet connection the first time the dashboard opens: it loads pinned, integrity-checked libraries from jsDelivr and cdnjs.

## Step 2: Create a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of the prompt.

## Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

You should see `Successfully installed streamlit … shap …`. `xgboost` is optional and not required.

## Step 4: Run the test suite

```bash
python astravive_test_suite.py
```

You should see `Ran 9 tests … OK`. What the tests cover:

| Test | Checks |
|---|---|
| 01 | HTML structure: 9 tab panels, responsive grid classes |
| 02 | Invalid measurements → NaN → imputation; LINEAR/MEDIAN audit |
| 03 | CSV/JSON ingestion, provenance tags, export |
| 04 | Subject-wise ML benchmark; SHAP additivity |
| 05 | Twin scenario A vs B, extreme-input clamping, duration alignment |
| 06 | Integrator bundle, safe error summaries, `data_mode` |
| 07 | Documented twin equations == code coefficients |
| 08 | Missing-path warning, `FileNotFoundError` when fallback is disabled, held-out anomaly split |
| 09 | Front-end integrity: no unseeded RNG, no secrets or Firebase/postMessage/Tailwind CDN, ARIA tabs, focus-visible, reduced motion, SRI on every script |

## Step 5 (optional): Run each engine on its own

```bash
python astravive_data_pipeline.py
python astravive_ml_pipeline.py
python astravive_xai_engine.py
python astravive_digital_twin_simulator.py
python astravive_system_integrator.py
```

Each exits with code 0. WARNING lines about out-of-range `hr_bpm`/`rr_interval_s`/`skin_temp_c` are expected: the synthetic demo injects one outlier per channel to prove the bounds checks work.

## Step 6 (optional): Use real datasets

Put the files in `data/` (see `docs/dataset_download_training_provenance_guide.md` for the OSDR, PhysioNet and UCI sources) and pass their paths to the pipeline. If a path you give does not exist, the pipeline logs a loud warning. With `allow_synthetic_fallback=False` it stops with `FileNotFoundError` instead of silently using demo data. The bundle's `data_mode` then reads `EXTERNAL_DATASET`.

## Step 7: Build the submission ZIP

```bash
python package_astravive_zip.py
```

The packager:
1. confirms every required file exists,
2. compiles every `.py`,
3. scans for secrets and forbidden patterns (API keys, Firebase, `postMessage`, unseeded RNG, Tailwind CDN),
4. writes `MANIFEST.txt` with SHA-256 hashes,
5. writes `ASTRAVIVE_NASA_SpaceApps_2026_Complete.zip`.

It exits with code 1 and lists the problem if any check fails.

## Step 8: Run the app

```bash
streamlit run app.py
# taller embed if your screen clips content:
ASTRAVIVE_IFRAME_HEIGHT=3000 streamlit run app.py      # macOS / Linux
$env:ASTRAVIVE_IFRAME_HEIGHT=3000; streamlit run app.py # Windows
```

Open http://localhost:8501. A blue info box saying SYNTHETIC_DEMO is expected without external data. A yellow warning means a stage degraded: check the terminal log.

You can also open `astravive_app.html` by double-clicking it. It needs no server.

## Step 9: Verify in the browser

1. Go to **Settings & QA** → **Run verification suite**. Expect **11/11 PASS** (tabs, Python twin parity, bounds, interpolation, ingestion, leakage-free benchmark, Shapley additivity, charts, no overflow, export, seeded RNG).
2. Keyboard check: press Tab once to reveal the skip link, then use ← → Home End in the tab bar. Organ hotspots are reachable with Tab, and Esc closes dialogs.
3. Fault tests on the same tab: HR spike, pCO₂ 8.2 mmHg, 20 % missing data, empty subject ID, WebGL off. Each shows a labelled result and the UI keeps working.

## Step 10: Deploy

Streamlit Community Cloud:
1. Push the project folder to a public GitHub repo. Do not commit `.venv/`, `*.db` or secrets.
2. On share.streamlit.io choose **New app**, pick the repo, set main file `app.py`, and choose Python 3.11 or newer under Advanced settings.
3. Optional: under **Secrets / environment**, set `ASTRAVIVE_IFRAME_HEIGHT`.
4. Deploy, then repeat Step 9 on the public URL.

Static only (no Python): host `astravive_app.html` on GitHub Pages or Netlify. The in-browser twin, forecast, XAI and benchmark all run client-side.

Docker (any cloud):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Blank 3D panel with a text note | WebGL unavailable | Expected fallback. Enable hardware acceleration |
| Icons or fonts missing | CDN blocked by network | Allow `cdn.jsdelivr.net` and `cdnjs.cloudflare.com` |
| Packager exits 1 | Missing file or forbidden pattern | Read the listed problem, fix it, re-run |
| `PARTIAL_DEGRADATION` in app | XAI or ML stage failed | Read the server log (tracebacks are logged there, not exported) |
