# Changelog

## v2.1 (2026-11-09): audit and polish release

### Security
- Removed a leaked Firebase API key, injected Gemini/Firebase scripts and the `postMessage('*')` bridge from the ZIP HTML.
- Replaced the Tailwind runtime CDN with precompiled inline CSS. Pinned all CDNs with SRI. Added a CSP that blocks all network connections.
- Replaced `innerHTML` templating with DOM APIs. Upload limits are 5 MB, CSV/JSON only.
- Exported bundles carry `Type: message` error summaries; tracebacks stay in the server log.

### Scientific integrity
- Replaced the `Math.random` benchmark and hardcoded KPIs with a real, seeded LOSO benchmark and live twin outputs.
- Removed fake PUBLIC DATA badges, fake CBC timepoints, the unmodelled "Immune" subsystem and the hand-set "Ridge/GBDT" forecast.
- Twin equations are documented 1:1 with the code (new test_07). Sols 1–7 relabelled SIMULATED_DATA.
- Anomaly detector evaluated on held-out subjects (new test_08). SHAP no longer rescaled.
- Documented the AUC = 1.0 on synthetic labels and that the persistence baseline beats the learned HR forecasters.

### Fixes
- Waveform freeze after pause, sqlite ResourceWarning, `to_datetime` warning, silent synthetic fallback, imputation labels, XAI-failure status, Scenario B duration mismatch, test_01 id mismatch, packager silently skipping missing files.

### UI/UX
- ARIA tabs with keyboard navigation, skip link, `:focus-visible` rings, 44 px targets, button organ hotspots, focus-trapped modals.
- Contrast raised to ≥ 4.5:1. `prefers-reduced-motion` support. No horizontal scroll at 390 px.
- New: Monte Carlo forecast, exact Shapley XAI, CSV/JSON ingestion, mission configuration, CSV/JSON/SQL export, in-page 11-check verification suite, fault-injection buttons.
- Streamlit: configurable iframe height, status and SYNTHETIC_DEMO notices, emoji-free icon.

### Packaging and docs
- Packager: required-file check, `py_compile`, forbidden-pattern scan, SHA-256 manifest, exit 1 on failure.
- Added README, AUDIT_REPORT, STEP_BY_STEP_GUIDE and CHANGELOG. Archived the v2.0 docs in `docs/`.

## v2.0: original submission (uploaded)
