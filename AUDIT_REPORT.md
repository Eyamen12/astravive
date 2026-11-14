# ASTRAVIVE v2.1: Audit Report

Scope: all 17 uploaded files (6 Python modules, Streamlit entry point, packager, test suite, v2.0 HTML, requirements, 4 docs and the submission ZIP). Audit date: 2026-11-09.
Method: read every file line by line, run every module's `__main__` with `-W default`, run the test suite, check the dashboard in headless Chromium at 1440 px and 390 px on all 9 tabs, and diff the documented equations against the code.

Severity: **Critical** = security or data-integrity breach · **High** = wrong or fabricated results · **Medium** = functional bug or accessibility blocker · **Low** = polish or maintainability.

## 1. Findings

### 1.1 Security

| # | Sev | File | Issue | Fix | Verified by |
|---|---|---|---|---|---|
| S1 | Critical | ZIP `index.html` | A live-format Firebase API key (`AIzaSy…`) was embedded in the page | Removed. The packager now blocks `AIzaSy[0-9A-Za-z_-]{10,}` | Packager scan; test_09 |
| S2 | Critical | ZIP `index.html` | Injected Gemini/Firebase scripts and `postMessage(..., '*')` (sends data to any origin) | All removed. The page has no network calls (`connect-src 'none'`) | test_09 greps for `firebase`, `postMessage(` |
| S3 | High | v2.0 HTML | Runtime Tailwind CDN plus unpinned CDNs without SRI | Tailwind precompiled and inlined. three r149, Chart.js 4.4.1, Font Awesome 6.5.1 and Fontsource pinned with `integrity` + `crossorigin` | test_09 checks every `<script src>` has `integrity` |
| S4 | High | v2.0 HTML | `innerHTML` templating with data values (XSS risk once users upload files) | All dynamic content is built with `textContent` and `createElement` | `grep -c innerHTML` on the script = 0 |
| S5 | Medium | HTML | No Content-Security-Policy | CSP meta added: `default-src 'none'`, scripts only from jsDelivr, `object-src 'none'`, `base-uri 'none'`, `form-action 'none'` | Page loads with 0 CSP violations |
| S6 | Medium | `astravive_system_integrator.py` | Full Python tracebacks written into the exported bundle | `_safe_error_summary()` exports `Type: message`. The traceback goes only to the server log | test_06 |
| S7 | Medium | HTML upload | No size or type limit on user files | 5 MB cap, `.csv`/`.json` only, parsed locally, physiological bounds checked | Diagnostics check 5 |
| S8 | Low | `astravive_ml_pipeline.py` | Pickled model artefacts loaded without warning | Added a docstring note: only load pickles you created. The shipped bundle is JSON | Review |

### 1.2 Scientific integrity (fabricated or inconsistent numbers)

| # | Sev | File | Issue | Fix | Verified by |
|---|---|---|---|---|---|
| D1 | High | v2.0 HTML | Benchmark table generated with `Math.random`; KPIs hardcoded | Real in-browser LOSO benchmark (Dummy, LogReg, Gaussian NB, ARX OLS, z-distance anomaly) on seeded data | Diagnostics check 6 (train/test subjects disjoint, LR AUC 0.881 vs Dummy 0.500) |
| D2 | High | ZIP HTML | "Ridge/GBDT" predictions used hand-set coefficients | Removed. Forecast tab is a Monte Carlo of the twin (500 seeded trajectories) | Review |
| D3 | High | v2.0 HTML | "PUBLIC DATA" badges on simulated values | Every value is tagged SIMULATED, DERIVED or ENCODED ANCHOR. The Data tab says no live NASA data is loaded | Visual check, all tabs |
| D4 | High | v2.0 HTML | CBC chart showed non-existent "FD+" timepoints | Removed. OSD-569 is now described only as a registry entry | Review |
| D5 | Medium | v2.0 HTML | Cumulative dose 128.4 mSv disagreed with rate × Sol; Sol display not zero-padded | Dose computed as Σ rate. Sol shown as `042 / 090` | Overview: 75.6 mSv at Sol 42 on the Gateway |
| D6 | High | twin + HTML + docs | Documented equations did not match the code; `c_fatigue_icp` typo | Docstring rewritten 1:1 with the code. The Data & Methods tab shows the same equations | **test_07** parses the docstring and compares coefficients |
| D7 | Medium | twin | Sols 1–7 labelled MODEL PREDICTION, but `g_ML` is never applied | Relabelled SIMULATED_DATA. The docstring states `g_ML` is not applied | test_05, test_07 |
| D8 | Medium | v2.0 HTML | An "Immune" subsystem was displayed but not modelled | Removed. Radar shows the 5 modelled axes | Visual |
| D9 | Medium | twin | Scenario B silently used a different duration from A | B is aligned to A's duration with a logged warning | test_05 |
| D10 | High | `astravive_ml_pipeline.py` | Anomaly detector evaluated on its own training rows | Subject-held-out GroupKFold, fitted on nominal training rows only. `split_scheme` records "held-out" | **test_08** |
| D11 | Medium | `astravive_xai_engine.py` | SHAP values rescaled to force additivity | Chain: TreeExplainer (interventional) → Permutation → exact baseline Shapley. No rescaling | test_04 residual 0.0 |
| D12 | Info (open) | ML | Synthetic labels are a deterministic function of the features, so classifier AUC is 1.0 | Documented in README and UI as a pipeline check, not evidence of performance | `python astravive_ml_pipeline.py` |
| D13 | Info (open) | ML | Naive persistence (MAE 4.67) beats every learned HR forecaster (best 6.70) | Documented. The selector compares learned models only | Same run |

### 1.3 Functional bugs

| # | Sev | File | Issue | Fix | Verified by |
|---|---|---|---|---|---|
| F1 | Medium | v2.0 HTML | `tickWaveforms()` returned early while paused and never rescheduled, so waveforms froze for good | Loop always reschedules; pausing (tab hidden) only skips drawing | Code review (`tickWaveforms`) |
| F2 | Medium | test_01 | Expected `tab-*` ids, HTML used `panel-*` | Panels are `id="tab-<name>"`, buttons `id="tabbtn-<name>"` | test_01 |
| F3 | Low | v2.0 HTML | three.js 0.160 deprecation warning for `build/three.min.js` | Pinned r149 (last UMD build without the warning). WebGL failure shows a text fallback | 0 console errors |
| F4 | Medium | data pipeline | `sqlite3` connection not closed (ResourceWarning) | `contextlib.closing()` | `python -W default` clean |
| F5 | Low | data pipeline | `pd.to_datetime` format-inference warning | `format="ISO8601"` | Same |
| F6 | High | data pipeline | An explicit file path that was missing fell back to synthetic data **silently** | Loud WARNING. With `allow_synthetic_fallback=False` it raises `FileNotFoundError` | **test_08** |
| F7 | Low | data pipeline | Imputation audit labelled everything LINEAR | Separate LINEAR and MEDIAN counts | test_02 |
| F8 | Medium | integrator | XAI failure still reported OPERATIONAL | Sets `PARTIAL_DEGRADATION`. Adds `data_mode` (SYNTHETIC_DEMO / EXTERNAL_DATASET) | test_06 |
| F9 | Low | xai | `RANDOM_SEED_XAI` duplicated the global seed | Uses `RANDOM_SEED` | Review |
| F10 | Medium | `app.py` | Fixed iframe height clipped content; status not surfaced | `ASTRAVIVE_IFRAME_HEIGHT` (default 2200, clamped 600–6000). Shows a warning if not OPERATIONAL and an info note in SYNTHETIC_DEMO | Run locally |
| F11 | High | packager | Listed files that did not exist and skipped them silently | Fails with exit 1 on any missing file, compile error or forbidden pattern. Writes a SHA-256 `MANIFEST.txt` | `python package_astravive_zip.py` |
| F12 | Low | `app.py` | Emoji page icon | `:material/monitor_heart:` | Review |

### 1.4 Accessibility and UX (WCAG 2.2 AA)

| # | Sev | Issue | Fix |
|---|---|---|---|
| A1 | Medium | Tabs were plain buttons with no ARIA | `role="tablist"`, 9 `role="tab"` / `role="tabpanel"`, `aria-selected`, `aria-controls`, roving tabindex, ← → Home End keys |
| A2 | Medium | Organ hotspots were `div`s (no keyboard access, 18 px targets) | Real `<button>`s with a 44×44 px hit area and labels shown on hover or focus |
| A3 | Medium | Modals had no focus management | Focus moves into the dialog and is trapped; Esc closes; focus returns to the trigger |
| A4 | Medium | Text at /50–/60 opacity and `--text-mute #6b8590` fell below 4.5:1 | Raised to ≥ /80 and `#8aa3ad` (≥ 5.5:1 on panels) |
| A5 | Medium | No visible focus | Global `:focus-visible` 2 px cyan ring. Skip link to main content |
| A6 | Low | Buttons, selects and tabs below 44 px | `min-height: 44px` everywhere, including small buttons |
| A7 | Low | Motion not reducible | `prefers-reduced-motion` stops pulses, waveforms and the 3D spin |
| A8 | Low | 390 px layout: SHAP rows and tabs overflowed | Responsive SHAP grid, tab strip scrolls horizontally. Measured `scrollWidth == innerWidth` at 390 and 1440 |
| A9 | Low | Charts or WebGL failing left blank boxes | `.chart-fallback` / `.holo-fallback` text |

## 2. Verification results (2026-11-09)

| Check | Result |
|---|---|
| `python astravive_test_suite.py` | **9/9 OK** |
| Browser verification suite (Settings & QA) at 1440 px and 390 px | **11/11 PASS**, 0 page errors, 0 console errors (only headless-GPU WebGL notices) |
| Twin parity JS ↔ Python (default A/B, 90 Sol, 0 g) | max \|Δ\| = 4.1e-3. Readiness A90 43.09, B90 89.12 |
| Shapley additivity (in-browser) | residual 0.0 |
| Each module's `__main__` with `-W default` | exit 0, no Python warnings (only expected data-validation logs for deliberately injected outliers) |
| `python -m py_compile *.py` | OK |
| Packager forbidden-pattern scan + manifest | OK (see `MANIFEST.txt` in the ZIP) |

## 3. Not covered

- No real NASA OSDR or PhysioNet files were downloaded, so `EXTERNAL_DATASET` mode was only tested with fixtures.
- Screen-reader testing was a structural ARIA review, not a session with NVDA or VoiceOver.
- No penetration test of a deployed Streamlit instance.
