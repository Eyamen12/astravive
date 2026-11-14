"""
ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN
Module: Unified 10-Stage System Orchestrator & Health-Check Integrator
Target: NASA Space Apps Challenge 2026

PURPOSE:
Integrates all developed ASTRAVIVE backend modules into a single deterministic,
fault-tolerant workflow:
  1. astravive_data_pipeline.py        (Stages 1-5: Ingestion, Validation, Preprocessing, Features)
  2. astravive_ml_pipeline.py          (Stage 6: Anti-Leakage ML Training & Cross-Validation)
  3. astravive_xai_engine.py           (Stage 8: Global & Local SHAP Explainable AI)
  4. astravive_digital_twin_simulator.py (Stages 7 & 9: Scenario A vs. Scenario B Digital Twin)

ERROR HANDLING & ZERO-FABRICATION GUARANTEE:
- Every stage is wrapped in a structured try/except boundary (`StageExecutionStatus`).
- If any external file, optional library, or upstream component is unavailable, the
  integrator NEVER fabricates fake accuracy or clinical validation numbers; instead,
  it logs an explicit `COMPONENT_UNAVAILABLE` or `SYNTHETIC_DEMO_FALLBACK` status with
  a human-readable explanation.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from astravive_xai_engine import AstraviveXAIEngine, MANDATORY_XAI_DISCLAIMERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | ASTRAVIVE-INTEGRATOR | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("ASTRAVIVE_SystemIntegrator")


@dataclass
class ComponentStatus:
    """Tracks operational health, provenance classification, and error state of each subsystem."""
    component_id: str
    component_name: str
    status: str  # "OPERATIONAL", "DEGRADED_FALLBACK", "UNAVAILABLE_ERROR"
    data_classification: str
    execution_latency_ms: float
    explanation: str
    error_trace: Optional[str] = None  # AUDIT FIX: "<ExceptionType>: <message>" only; full traceback goes to logs


def _safe_error_summary(exc: BaseException) -> str:
    """Log the full traceback server-side; return only a short, non-sensitive summary for export."""
    logger.error("Component failure (full traceback follows)", exc_info=exc)
    return f"{type(exc).__name__}: {str(exc)[:300]}"


@dataclass
class UnifiedSystemStateBundle:
    """Complete end-to-end state bundle exported by the integrated ASTRAVIVE pipeline."""
    system_name: str = "ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN"
    integration_timestamp_utc: str = ""
    overall_system_status: str = "OPERATIONAL"
    zero_fabrication_lock: bool = True
    # AUDIT NOTE: with no local dataset the data stage reports DEGRADED_FALLBACK (synthetic input) while
    # overall_system_status stays OPERATIONAL, because every stage ran. `data_mode` makes this explicit.
    data_mode: str = "UNKNOWN"  # "SYNTHETIC_DEMO" | "EXTERNAL_DATASET"
    component_health_checks: List[ComponentStatus] = field(default_factory=list)
    ingestion_and_validation_summary: Dict[str, Any] = field(default_factory=dict)
    ml_evaluation_summary: Dict[str, Any] = field(default_factory=dict)
    xai_explanation_summary: Dict[str, Any] = field(default_factory=dict)
    digital_twin_scenario_comparison: Dict[str, Any] = field(default_factory=dict)
    exported_files: Dict[str, str] = field(default_factory=dict)


class AstraviveSystemIntegrator:
    """
    Coordinates all 10 stages of the ASTRAVIVE architecture with defensive error handling,
    provenance verification, and unified JSON/CSV/SQLite export.
    """

    def __init__(
        self,
        output_dir: str = "data/integrated_bundle",
        sqlite_db_path: str = "data/astravive.db",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path = sqlite_db_path
        self.bundle = UnifiedSystemStateBundle(
            integration_timestamp_utc=datetime.now(timezone.utc).isoformat()
        )

    def execute_integrated_workflow(
        self,
        external_wearable_file: Optional[str] = None,
        mission_duration_sols: int = 90,
    ) -> UnifiedSystemStateBundle:
        """
        Executes all stages sequentially, passing validated outputs from Stage 2-5
        into Stage 6 (ML), Stage 8 (XAI), and Stage 7/9 (Digital Twin Simulator).
        """
        data_pipe = AstraviveDataPipeline(sqlite_db_path=self.sqlite_db_path)
        ml_df = None
        clf_bundle = None

        # ======================================================================
        # CHECK 1 & 2: DATA INGESTION & VALIDATION (Stages 1 to 5)
        # ======================================================================
        t0 = datetime.now(timezone.utc)
        try:
            raw_wearable_df = data_pipe.ingest_dataset(
                dataset_id="MMASH-WEARABLE",
                file_path=external_wearable_file,
                allow_synthetic_fallback=True,
            )
            clean_df, val_report = data_pipe.validate_and_clean(
                raw_wearable_df, dataset_id="MMASH-WEARABLE"
            )
            feat_df = data_pipe.normalize_and_engineer_features(
                clean_df, dataset_id="MMASH-WEARABLE"
            )
            export_paths = data_pipe.export_processed_dataset(
                feat_df, dataset_id="MMASH-WEARABLE", output_dir=self.output_dir
            )
            ml_df = prepare_ml_dataset_without_leakage(feat_df)

            is_synth = bool(feat_df["is_synthetic"].iloc[0])
            self.bundle.data_mode = "SYNTHETIC_DEMO" if is_synth else "EXTERNAL_DATASET"
            cls_tag = str(feat_df["data_classification"].iloc[0])
            dt_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0

            self.bundle.ingestion_and_validation_summary = {
                "dataset_id": "MMASH-WEARABLE",
                "rows_ingested": val_report.input_row_count,
                "rows_validated": val_report.output_row_count,
                "invalid_measurements_detected": val_report.invalid_measurements_detected,
                "ectopic_rr_beats_filtered": val_report.ectopic_rr_beats_filtered,
                "missing_values_imputed": val_report.missing_values_imputed,
                "data_classification": cls_tag,
                "is_synthetic_fallback": is_synth,
                "exports": export_paths,
            }
            self.bundle.component_health_checks.append(
                ComponentStatus(
                    component_id="STAGE_1_TO_5_DATA_PIPELINE",
                    component_name="Data Ingestion, Validation, Normalization & Feature Engineering",
                    status="DEGRADED_FALLBACK" if is_synth else "OPERATIONAL",
                    data_classification=cls_tag,
                    execution_latency_ms=round(dt_ms, 2),
                    explanation=(
                        "External CSV/JSON file not provided; transparently generated clearly labeled "
                        "SIMULATED DATA calibrated to PhysioNet MMASH/WESAD distributions and verified "
                        "artifact/missing-value handling."
                        if is_synth
                        else "Successfully ingested and validated external PUBLIC/MEASURED dataset."
                    ),
                )
            )
        except Exception as exc:
            dt_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
            self.bundle.overall_system_status = "PARTIAL_DEGRADATION"
            self.bundle.component_health_checks.append(
                ComponentStatus(
                    component_id="STAGE_1_TO_5_DATA_PIPELINE",
                    component_name="Data Ingestion, Validation, Normalization & Feature Engineering",
                    status="UNAVAILABLE_ERROR",
                    data_classification="NONE_COMPONENT_UNAVAILABLE",
                    execution_latency_ms=round(dt_ms, 2),
                    explanation=(
                        f"Data ingestion/validation component encountered an error: {exc}. "
                        "Downstream ML and XAI stages are halted rather than fabricating metrics."
                    ),
                    error_trace=_safe_error_summary(exc),
                )
            )

        # ======================================================================
        # CHECK 3: AI/ML PREDICTIONS & ANTI-LEAKAGE CROSS-VALIDATION (Stage 6)
        # ======================================================================
        t1 = datetime.now(timezone.utc)
        feature_cols = [
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
        forecast_cols = [
            "hr_bpm",
            "lag_1_hr_bpm",
            "lag_2_hr_bpm",
            "lag_3_hr_bpm",
            "derived_hrv_rmssd_ms",
            "lag_1_rmssd_ms",
            "lag_2_rmssd_ms",
            "derived_actigraphy_enmo_g",
            "lag_1_enmo_g",
            "eda_microsiemens",
            "derived_borbely_sleep_pressure_s",
            "rolling_past_workload_enmo",
        ]

        if ml_df is not None and len(ml_df) >= 20:
            try:
                ml_engine = AstraviveMLPipeline(model_output_dir=self.output_dir)
                eda_res = perform_exploratory_data_analysis(
                    ml_df, feature_cols=feature_cols, target_col="target_stress_binary"
                )
                clf_metrics, best_clf = ml_engine.evaluate_classification_models(
                    ml_df, feature_cols=feature_cols, target_col="target_stress_binary"
                )
                reg_metrics, best_reg = ml_engine.evaluate_forecasting_models(
                    ml_df, feature_cols=forecast_cols, target_col="target_next_hr_bpm"
                )
                anom_metrics = ml_engine.train_and_evaluate_anomaly_detector(
                    ml_df, anomaly_features=feature_cols
                )
                clf_bundle = ml_engine.trained_pipelines.get("health_state_classifier")

                dt_ms = (datetime.now(timezone.utc) - t1).total_seconds() * 1000.0
                self.bundle.ml_evaluation_summary = {
                    "selected_classifier": best_clf,
                    "selected_forecaster": best_reg,
                    "exploratory_data_analysis": asdict(eda_res),
                    "classification_groupkfold_metrics": [asdict(m) for m in clf_metrics],
                    "forecasting_timeseriessplit_metrics": [asdict(m) for m in reg_metrics],
                    "anomaly_detection_metrics": asdict(anom_metrics),
                }
                self.bundle.component_health_checks.append(
                    ComponentStatus(
                        component_id="STAGE_6_ML_ENGINE",
                        component_name="Anti-Leakage ML Engine (GroupKFold & TimeSeriesSplit)",
                        status="OPERATIONAL",
                        data_classification=DataClassification.MODEL_PREDICTION.value,
                        execution_latency_ms=round(dt_ms, 2),
                        explanation=(
                            f"Trained and evaluated candidate models without data leakage. "
                            f"Selected classifier: '{best_clf}'; Selected forecaster: '{best_reg}'."
                        ),
                    )
                )
            except Exception as exc:
                dt_ms = (datetime.now(timezone.utc) - t1).total_seconds() * 1000.0
                self.bundle.overall_system_status = "PARTIAL_DEGRADATION"
                self.bundle.component_health_checks.append(
                    ComponentStatus(
                        component_id="STAGE_6_ML_ENGINE",
                        component_name="Anti-Leakage ML Engine (GroupKFold & TimeSeriesSplit)",
                        status="UNAVAILABLE_ERROR",
                        data_classification="NONE_COMPONENT_UNAVAILABLE",
                        execution_latency_ms=round(dt_ms, 2),
                        explanation=f"ML pipeline unavailable due to runtime exception: {exc}. No fake metrics generated.",
                        error_trace=_safe_error_summary(exc),
                    )
                )
        else:
            self.bundle.component_health_checks.append(
                ComponentStatus(
                    component_id="STAGE_6_ML_ENGINE",
                    component_name="Anti-Leakage ML Engine (GroupKFold & TimeSeriesSplit)",
                    status="UNAVAILABLE_ERROR",
                    data_classification="NONE_COMPONENT_UNAVAILABLE",
                    execution_latency_ms=0.0,
                    explanation="Upstream feature store unavailable or has < 20 rows; ML evaluation skipped transparently.",
                )
            )

        # ======================================================================
        # CHECK 4: EXPLAINABLE AI (XAI) ENGINE (Stage 8)
        # ======================================================================
        t2 = datetime.now(timezone.utc)
        if ml_df is not None and clf_bundle is not None:
            try:
                xai_engine = AstraviveXAIEngine(
                    trained_pipeline=clf_bundle["pipeline"],
                    feature_cols=feature_cols,
                    background_df=ml_df,
                    target_col="target_stress_binary",
                )
                global_imp = xai_engine.compute_global_feature_importance(ml_df)
                stress_sample = ml_df.loc[ml_df["target_stress_binary"] == 1].iloc[0]
                local_exp = xai_engine.explain_individual_observation(
                    stress_sample, observation_id="INTEGRATED-STRESS-WINDOW-01"
                )
                dt_ms = (datetime.now(timezone.utc) - t2).total_seconds() * 1000.0

                self.bundle.xai_explanation_summary = {
                    "explainer_method": xai_engine.explainer_method_name,
                    "mandatory_disclaimers": MANDATORY_XAI_DISCLAIMERS,
                    "global_feature_importance": [asdict(g) for g in global_imp],
                    "sample_individual_explanation": asdict(local_exp),
                }
                self.bundle.component_health_checks.append(
                    ComponentStatus(
                        component_id="STAGE_8_XAI_ENGINE",
                        component_name="Explainable AI (SHAP & Permutation Attribution)",
                        status="OPERATIONAL",
                        data_classification=DataClassification.MODEL_PREDICTION.value,
                        execution_latency_ms=round(dt_ms, 2),
                        explanation=(
                            f"Computed global feature importance and individual SHAP explanation "
                            f"(additivity residual = {local_exp.additivity_check_residual:.2e})."
                        ),
                    )
                )
            except Exception as exc:
                dt_ms = (datetime.now(timezone.utc) - t2).total_seconds() * 1000.0
                self.bundle.overall_system_status = "PARTIAL_DEGRADATION"  # AUDIT FIX: consistent with other stages
                self.bundle.component_health_checks.append(
                    ComponentStatus(
                        component_id="STAGE_8_XAI_ENGINE",
                        component_name="Explainable AI (SHAP & Permutation Attribution)",
                        status="UNAVAILABLE_ERROR",
                        data_classification="NONE_COMPONENT_UNAVAILABLE",
                        execution_latency_ms=round(dt_ms, 2),
                        explanation=f"XAI engine encountered an error: {exc}.",
                        error_trace=_safe_error_summary(exc),
                    )
                )
        else:
            self.bundle.component_health_checks.append(
                ComponentStatus(
                    component_id="STAGE_8_XAI_ENGINE",
                    component_name="Explainable AI (SHAP & Permutation Attribution)",
                    status="UNAVAILABLE_ERROR",
                    data_classification="NONE_COMPONENT_UNAVAILABLE",
                    execution_latency_ms=0.0,
                    explanation="Trained ML classifier unavailable; XAI attribution skipped transparently.",
                )
            )

        # ======================================================================
        # CHECK 5: DIGITAL TWIN RESEARCH SIMULATOR — SCENARIO A vs. SCENARIO B
        # ======================================================================
        t3 = datetime.now(timezone.utc)
        try:
            twin_sim = AstronautDigitalTwinSimulator()
            scen_a, scen_b = twin_sim.get_default_scenarios(
                mission_duration_sols=mission_duration_sols
            )
            comp_result = twin_sim.run_comparison(scen_a, scen_b)
            sim_exports = twin_sim.export_simulation_results(
                comp_result, output_dir=str(self.output_dir)
            )
            dt_ms = (datetime.now(timezone.utc) - t3).total_seconds() * 1000.0

            self.bundle.digital_twin_scenario_comparison = asdict(comp_result)
            self.bundle.exported_files.update(sim_exports)
            self.bundle.component_health_checks.append(
                ComponentStatus(
                    component_id="STAGE_7_AND_9_DIGITAL_TWIN",
                    component_name="Interactive Digital-Twin Simulator (Scenario A vs. Scenario B)",
                    status="OPERATIONAL",
                    data_classification=(
                        "Sol 0: PUBLIC DATA | Sols 1-7: MODEL PREDICTION | Sols 8+: SIMULATED DATA"
                    ),
                    execution_latency_ms=round(dt_ms, 2),
                    explanation=comp_result.scenario_explanation,
                )
            )
        except Exception as exc:
            dt_ms = (datetime.now(timezone.utc) - t3).total_seconds() * 1000.0
            self.bundle.overall_system_status = "PARTIAL_DEGRADATION"
            self.bundle.component_health_checks.append(
                ComponentStatus(
                    component_id="STAGE_7_AND_9_DIGITAL_TWIN",
                    component_name="Interactive Digital-Twin Simulator (Scenario A vs. Scenario B)",
                    status="UNAVAILABLE_ERROR",
                    data_classification="NONE_COMPONENT_UNAVAILABLE",
                    execution_latency_ms=round(dt_ms, 2),
                    explanation=f"Digital Twin Simulator failed with error: {exc}.",
                    error_trace=_safe_error_summary(exc),
                )
            )

        # Save unified integration report JSON
        unified_json_path = self.output_dir / "astravive_unified_system_bundle.json"
        self.bundle.exported_files["unified_bundle_json"] = str(unified_json_path)
        with open(unified_json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.bundle), f, indent=2)

        logger.info(f"Saved complete integrated ASTRAVIVE system bundle to: {unified_json_path}")
        return self.bundle


if __name__ == "__main__":
    print("=" * 92)
    print("ASTRAVIVE — END-TO-END 10-STAGE SYSTEM INTEGRATION & HEALTH CHECK")
    print("=" * 92)
    integrator = AstraviveSystemIntegrator()
    state_bundle = integrator.execute_integrated_workflow(mission_duration_sols=90)
    print(f"Overall System Status: {state_bundle.overall_system_status}")
    for chk in state_bundle.component_health_checks:
        print(
            f" - [{chk.status:17s}] {chk.component_name:62s} | "
            f"{chk.execution_latency_ms:6.2f} ms | Class: {chk.data_classification}"
        )