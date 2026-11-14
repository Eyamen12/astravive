"""
ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN
Module: Automated End-to-End Technical Review & Verification Test Suite
Target: NASA Space Apps Challenge 2026

PURPOSE:
Provides deterministic unit and integration tests covering all 9 technical review domains:
  1. Application functionality & HTML/JS structural integrity
  2. Invalid input handling (out-of-bounds sensor spikes, invalid timestamps, extreme sliders)
  3. Missing data handling (NaN telemetry dropouts, all-NaN columns, empty subject_id)
  4. Dataset loading (CSV, JSON, and clearly labeled synthetic fallback)
  5. Model predictions (GroupKFold, TimeSeriesSplit, single-subject fallback, IsolationForest)
  6. Digital twin calculations (Scenario A vs. Scenario B ODE stability & provenance tags)
  7. Visualization & XAI additivity (|E[f(X)] + sum(phi_j) - f(x)| < 1e-6)
  8. Responsive design & zero-alert() UI compliance
  9. Export functionality (CSV, JSON, SQLite persistence)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from astravive_data_pipeline import (
    AstraviveDataPipeline,
    ClearlyLabeledSyntheticGenerator,
    DataClassification,
)
from astravive_digital_twin_simulator import (
    AstronautDigitalTwinSimulator,
    ScenarioParameters,
)
from astravive_ml_pipeline import (
    AstraviveMLPipeline,
    RANDOM_SEED,
    perform_exploratory_data_analysis,
    prepare_ml_dataset_without_leakage,
)
from astravive_system_integrator import AstraviveSystemIntegrator
from astravive_xai_engine import AstraviveXAIEngine


class TestAstraviveTechnicalReview(unittest.TestCase):
    """Comprehensive verification suite across all 9 ASTRAVIVE test categories."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.db_path = self.work_dir / "test_astravive.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # ------------------------------------------------------------------
    # TEST 1 & 8: Application Functionality, Responsive Layout & No alert()
    # ------------------------------------------------------------------
    def test_01_html_application_structure_and_responsive_compliance(self) -> None:
        html_path = Path("astravive_app.html")
        if not html_path.exists():
            self.skipTest("astravive_app.html not found in current working directory.")
        content = html_path.read_text(encoding="utf-8")

        # Verify all 9 required mission-control tab sections exist
        required_tabs = [
            'id="tab-overview"',
            'id="tab-health"',
            'id="tab-forecast"',
            'id="tab-twin"',
            'id="tab-environment"',
            'id="tab-xai"',
            'id="tab-performance"',
            'id="tab-data"',
            'id="tab-settings"',
        ]
        for tab_marker in required_tabs:
            self.assertIn(tab_marker, content, f"Missing required tab section: {tab_marker}")

        # Verify zero usage of blocking browser alert() or confirm()
        self.assertNotIn("window.alert(", content)
        self.assertNotIn("window.confirm(", content)

        # Verify responsive Tailwind grid classes exist
        self.assertIn("lg:grid-cols-12", content)
        self.assertIn("sm:grid-cols-2", content)

    # ------------------------------------------------------------------
    # TEST 2 & 3: Invalid Input Handling & Missing Data Handling
    # ------------------------------------------------------------------
    def test_02_invalid_measurements_and_missing_data_resilience(self) -> None:
        pipeline = AstraviveDataPipeline(sqlite_db_path=self.db_path)
        raw_df = ClearlyLabeledSyntheticGenerator.generate_wearable_demo_stream(
            n_subjects=2, epochs_per_subject=30, inject_test_artifacts=True, random_seed=RANDOM_SEED
        )

        # Inject additional edge cases: corrupted timestamp, empty subject_id, and out-of-range HR
        raw_df.loc[0, "timestamp_utc"] = "NOT_A_VALID_TIMESTAMP"
        raw_df.loc[1, "subject_id"] = "   "
        raw_df.loc[2, "hr_bpm"] = -45.0

        clean_df, report = pipeline.validate_and_clean(raw_df, dataset_id="MMASH-WEARABLE")

        # Verify zero NaNs remain in numeric columns after transparent imputation
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertEqual(int(clean_df[col].isna().sum()), 0, f"Unimputed NaN remained in {col}")

        # Verify out-of-bounds HR spike (295.0 and -45.0) was clamped/imputed into [30, 220] bpm
        self.assertTrue((clean_df["hr_bpm"] >= 30.0).all() and (clean_df["hr_bpm"] <= 220.0).all())
        self.assertIn("hr_bpm", report.invalid_measurements_detected)
        self.assertIn("timestamp_utc", report.invalid_measurements_detected)
        self.assertIn("subject_id", report.invalid_measurements_detected)

    # ------------------------------------------------------------------
    # TEST 4 & 9: Dataset Loading (CSV/JSON) & Multi-Format Export
    # ------------------------------------------------------------------
    def test_03_csv_json_ingestion_provenance_and_export(self) -> None:
        pipeline = AstraviveDataPipeline(sqlite_db_path=self.db_path)
        raw_df = ClearlyLabeledSyntheticGenerator.generate_wearable_demo_stream(
            n_subjects=2, epochs_per_subject=20, inject_test_artifacts=False
        )
        clean_df, _ = pipeline.validate_and_clean(raw_df, dataset_id="MMASH-WEARABLE")
        feat_df = pipeline.normalize_and_engineer_features(clean_df, dataset_id="MMASH-WEARABLE")

        exports = pipeline.export_processed_dataset(
            feat_df, dataset_id="MMASH-WEARABLE", output_dir=self.work_dir
        )
        self.assertTrue(Path(exports["csv_path"]).exists())
        self.assertTrue(Path(exports["json_path"]).exists())
        self.assertTrue(Path(exports["sqlite_db_path"]).exists())

        # Re-ingest from the exported CSV and JSON files and verify provenance classification
        reloaded_csv = pipeline.ingest_dataset(
            dataset_id="MMASH-WEARABLE", file_path=exports["csv_path"]
        )
        reloaded_json = pipeline.ingest_dataset(
            dataset_id="MMASH-WEARABLE", file_path=exports["json_path"]
        )
        self.assertEqual(len(reloaded_csv), len(feat_df))
        self.assertEqual(len(reloaded_json), len(feat_df))
        self.assertIn("derived_hrv_rmssd_ms", reloaded_csv.columns)
        self.assertEqual(
            reloaded_csv["derived_features_classification"].iloc[0],
            DataClassification.DERIVED_DATA.value,
        )

    # ------------------------------------------------------------------
    # TEST 5 & 7: Anti-Leakage ML Predictions & Exact SHAP Additivity
    # ------------------------------------------------------------------
    def test_04_ml_pipeline_and_shap_additivity(self) -> None:
        pipeline = AstraviveDataPipeline(sqlite_db_path=self.db_path)
        raw_df = ClearlyLabeledSyntheticGenerator.generate_wearable_demo_stream(
            n_subjects=6, epochs_per_subject=45, inject_test_artifacts=True, random_seed=RANDOM_SEED
        )
        clean_df, _ = pipeline.validate_and_clean(raw_df, dataset_id="MMASH-WEARABLE")
        feat_df = pipeline.normalize_and_engineer_features(clean_df, dataset_id="MMASH-WEARABLE")
        ml_df = prepare_ml_dataset_without_leakage(feat_df)

        features = [
            "hr_bpm",
            "derived_hrv_rmssd_ms",
            "derived_hrv_sdnn_ms",
            "derived_sympathovagal_ratio",
            "eda_microsiemens",
            "skin_temp_c",
            "derived_actigraphy_enmo_g",
            "derived_borbely_sleep_pressure_s",
            "rolling_past_workload_enmo",
            "autonomic_recovery_velocity",
        ]

        ml_engine = AstraviveMLPipeline(model_output_dir=self.work_dir)
        clf_metrics, best_clf = ml_engine.evaluate_classification_models(
            ml_df, feature_cols=features, target_col="target_stress_binary"
        )
        self.assertTrue(len(clf_metrics) >= 4)
        self.assertTrue(best_clf != "")

        # Verify XAI local additivity: |E[f(X)] + sum(phi_j) - f(x)| < 1e-6
        clf_bundle = ml_engine.trained_pipelines["health_state_classifier"]
        xai = AstraviveXAIEngine(
            trained_pipeline=clf_bundle["pipeline"],
            feature_cols=features,
            background_df=ml_df,
            target_col="target_stress_binary",
        )
        local_exp = xai.explain_individual_observation(ml_df.iloc[0], observation_id="TEST-OBS-01")
        self.assertLess(local_exp.additivity_check_residual, 1e-6)

    # ------------------------------------------------------------------
    # TEST 6: Digital Twin Scenario A vs. Scenario B Calculations & Bounds
    # ------------------------------------------------------------------
    def test_05_digital_twin_scenario_comparison_and_extreme_bounds(self) -> None:
        sim = AstronautDigitalTwinSimulator()
        scen_a, scen_b = sim.get_default_scenarios(mission_duration_sols=90)
        comp = sim.run_comparison(scen_a, scen_b)

        # Under default countermeasures, Scenario B must outperform Scenario A
        deltas = comp.side_by_side_summary["net_effect_b_minus_a"]
        self.assertGreater(deltas["composite_readiness_delta_pct"], 10.0)
        self.assertGreater(deltas["hrv_rmssd_delta_ms"], 5.0)
        self.assertGreater(deltas["bmd_preservation_delta_pct"], 1.5)
        self.assertLess(deltas["sans_risk_reduction_pct"], -5.0)

        # Verify explicit provenance separation: Sol 0 anchor vs. simulated rollout (no ML is applied)
        rec0 = comp.trajectory_records[0]
        rec5 = comp.trajectory_records[5]
        rec30 = comp.trajectory_records[30]
        self.assertEqual(rec0["data_classification"], DataClassification.PUBLIC_DATA.value)
        self.assertEqual(rec5["data_classification"], DataClassification.SIMULATED_DATA.value)
        self.assertEqual(rec30["data_classification"], DataClassification.SIMULATED_DATA.value)

        # Test extreme out-of-range inputs are safely clamped by validate_bounds()
        extreme_scen = ScenarioParameters(
            scenario_label="EXTREME_TEST",
            mission_duration_sols=5000,   # Should clamp to 900
            gravity_g=-2.0,              # Should clamp to 0.0
            sleep_duration_hrs=0.5,      # Should clamp to 4.0
            recovery_duration_hrs=99.0,  # Should clamp to 4.0
            workload_intensity_pct=999.0,# Should clamp to 100.0
            exercise_duration_min=-50.0, # Should clamp to 0.0
            cabin_co2_mmhg=45.0,         # Should clamp to 6.0
        )
        extreme_comp = sim.run_comparison(extreme_scen, scen_b)
        self.assertEqual(extreme_comp.scenario_a_params["mission_duration_sols"], 900)
        self.assertEqual(extreme_comp.scenario_a_params["sleep_duration_hrs"], 4.0)
        self.assertEqual(extreme_comp.scenario_a_params["cabin_co2_mmhg"], 6.0)
        # Both scenarios must share one horizon so terminal deltas compare the same Sol
        self.assertEqual(extreme_comp.scenario_b_params["mission_duration_sols"], 900)
        self.assertEqual(len(extreme_comp.trajectory_records), 901)

    # ------------------------------------------------------------------
    # TEST 10: End-to-End Unified System Orchestrator
    # ------------------------------------------------------------------
    def test_06_unified_system_integrator_bundle(self) -> None:
        integrator = AstraviveSystemIntegrator(
            output_dir=str(self.work_dir), sqlite_db_path=str(self.db_path)
        )
        bundle = integrator.execute_integrated_workflow(mission_duration_sols=60)
        self.assertEqual(bundle.overall_system_status, "OPERATIONAL")
        self.assertEqual(len(bundle.component_health_checks), 4)
        self.assertTrue(Path(bundle.exported_files["unified_bundle_json"]).exists())

    # ------------------------------------------------------------------
    # TEST 11: Documented twin equations reproduce the simulator (doc/code parity)
    # ------------------------------------------------------------------
    def test_07_twin_documented_equations_match_code(self) -> None:
        import math
        sim = AstronautDigitalTwinSimulator()
        p, _ = sim.get_default_scenarios(mission_duration_sols=14)
        traj = sim._simulate_single_scenario(p)
        w, g, sl, rec, ex, co2 = (p.workload_intensity_pct / 100, p.gravity_g, p.sleep_duration_hrs,
                                  p.recovery_duration_hrs, p.exercise_duration_min, p.cabin_co2_mmhg)
        S, R, B = 22.0, 48.2, 0.0
        for t in range(3):  # step the DOCUMENTED equations independently
            S = min(95, max(6, S + 0.22 * (1.15 * ((24 - sl) - 16) + 8.5 * (w - 0.55) - 2.65 * (sl - 7.5)
                                          - 1.40 * (rec - 1.5) + 1.25 * max(0, co2 - 2.5))))
            target = 48.2 - 14.0 * (w - 0.5) + 3.6 * (rec - 1.5) - 0.24 * (S - 22) + 0.04 * min(ex, 90)
            R = min(82, max(14, R + 0.28 * (target - R)))
            B = B - 0.0433 * (1 - g) + 0.0330 * (ex / 120) - 0.012 * max(0, w - 0.70) * max(0, (8 - sl) / 4)
            sans = min(96, max(5, 10 + 24 * (1 - math.exp(-(t + 1) / 18)) * (1 - g)
                               + 8.4 * max(0, co2 - 2.0) ** 1.25 - 4.5 * (rec / 3) + 0.15 * max(0, S - 40)))
            rec_t = traj[t + 1]
            self.assertAlmostEqual(rec_t["sleep_pressure_index"], S, places=1)
            self.assertAlmostEqual(rec_t["hrv_rmssd_ms"], R, places=1)
            self.assertAlmostEqual(rec_t["bmd_delta_pct"], B, places=2)
            self.assertAlmostEqual(rec_t["sans_risk_index"], sans, places=1)
        doc = sim.doc["documented_equations_and_coefficients"]
        self.assertNotIn("c_ fatigue", json.dumps(doc))
        self.assertIn("0.22", doc["1_sleep_and_fatigue_dynamics"]["equation"])

    # ------------------------------------------------------------------
    # TEST 12: Out-of-sample anomaly evaluation & explicit-path fallback warning
    # ------------------------------------------------------------------
    def test_08_anomaly_out_of_sample_and_missing_path_warning(self) -> None:
        pipeline = AstraviveDataPipeline(sqlite_db_path=self.db_path)
        with self.assertLogs("ASTRAVIVE_DataPipeline", level="WARNING") as logs:
            df = pipeline.ingest_dataset(dataset_id="MMASH-WEARABLE", file_path=str(self.work_dir / "nope.csv"))
        self.assertTrue(any("EXPLICIT PATH NOT FOUND" in m for m in logs.output))
        self.assertTrue(bool(df["is_synthetic"].astype(bool).all()))
        with self.assertRaises(FileNotFoundError):
            pipeline.ingest_dataset(dataset_id="MMASH-WEARABLE", file_path=str(self.work_dir / "nope.csv"),
                                    allow_synthetic_fallback=False)

        clean_df, _ = pipeline.validate_and_clean(df, dataset_id="MMASH-WEARABLE")
        feat_df = pipeline.normalize_and_engineer_features(clean_df, dataset_id="MMASH-WEARABLE")
        ml_df = prepare_ml_dataset_without_leakage(feat_df)
        engine = AstraviveMLPipeline(model_output_dir=self.work_dir)
        m = engine.train_and_evaluate_anomaly_detector(ml_df, anomaly_features=["hr_bpm", "eda_microsiemens"])
        self.assertIn("held-out", m.split_scheme.lower())

    # ------------------------------------------------------------------
    # TEST 13: Front-end integrity (no fabricated metrics, no injected scripts, a11y, SRI)
    # ------------------------------------------------------------------
    def test_09_frontend_integrity_and_accessibility(self) -> None:
        html_path = Path("astravive_app.html")
        if not html_path.exists():
            self.skipTest("astravive_app.html not found in current working directory.")
        content = html_path.read_text(encoding="utf-8")
        self.assertFalse("Math.random" in content, "UI must use the seeded RNG, never Math.random")
        for forbidden in ("AIzaSy", "firebase", "postMessage(", "cdn.tailwindcss.com"):
            self.assertFalse(forbidden in content, f"forbidden token in HTML: {forbidden}")
        self.assertIn('role="tablist"', content)
        self.assertEqual(content.count('role="tab"'), 9)
        self.assertEqual(content.count('role="tabpanel"'), 9)
        self.assertIn("prefers-reduced-motion", content)
        self.assertIn(":focus-visible", content)
        import re
        for tag in re.findall(r"<script[^>]+src=[^>]+>", content):
            self.assertIn("integrity=", tag, f"CDN script without SRI: {tag}")


if __name__ == "__main__":
    unittest.main(verbosity=2)