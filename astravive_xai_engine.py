"""
ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN
Module: Explainable AI (XAI) Engine — Global & Local SHAP Attribution (Stage 8)
Target: NASA Space Apps Challenge 2026

SCIENTIFIC & REGULATORY RESPONSIBILITY MANDATE:
1. Association vs. Causation: Feature contributions (Shapley values phi_j and permutation
   importances) quantify how input variables shift a mathematical model's output relative to
   its baseline expectation E[f(X)]. They DO NOT establish biological or clinical causation.
   Collinear physiological signals (e.g., Heart Rate and RMSSD, or Cabin pCO2 and Sleep WASO)
   share statistical attribution without proving a direct causal mechanism.
2. Research Prototype (Not a Medical Diagnostic Tool): ASTRAVIVE is a decision-support and
   digital-twin research prototype developed for the NASA Space Apps Challenge 2026. It is NOT
   an FDA-cleared or NASA-certified clinical medical diagnostic device and must never be used
   to diagnose, treat, cure, or prevent disease.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from astravive_data_pipeline import (
    AstraviveDataPipeline,
    ClearlyLabeledSyntheticGenerator,
)
from astravive_ml_pipeline import (
    AstraviveMLPipeline,
    RANDOM_SEED,
    prepare_ml_dataset_without_leakage,
)

# Optional native SHAP library import with exact analytical fallback
try:
    import shap  # type: ignore
    SHAP_LIBRARY_AVAILABLE = True
except ImportError:
    SHAP_LIBRARY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | ASTRAVIVE-XAI-ENGINE | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("ASTRAVIVE_XAIEngine")


# ==============================================================================
# 1. MANDATORY SCIENTIFIC DISCLAIMERS (NON-CAUSATION & NON-DIAGNOSTIC)
# ==============================================================================

MANDATORY_XAI_DISCLAIMERS: Dict[str, str] = {
    "non_causation_statement": (
        "CRITICAL EPISTEMIC LIMITATION — CORRELATION VS. CAUSATION: "
        "SHAP feature attributions (phi_j) and permutation importance scores measure how much "
        "each input variable shifts the statistical model's prediction relative to the training "
        "distribution baseline E[f(X)]. Feature contributions DO NOT establish biological or "
        "clinical causation. Observational associations in wearable and habitat telemetry can "
        "be confounded by unmeasured variables or shared autonomic pathways."
    ),
    "non_diagnostic_statement": (
        "REGULATORY & CLINICAL NOTICE — NOT A MEDICAL DIAGNOSTIC TOOL: "
        "ASTRAVIVE is an exploratory biomedical research prototype and digital-twin simulation "
        "sandbox. Model predictions, risk indices, and feature explanations do NOT constitute "
        "medical diagnosis or clinical advice, and must never replace qualified aerospace medicine "
        "physicians (Flight Surgeons) or validated clinical laboratory diagnostics."
    ),
}


FEATURE_CLINICAL_METADATA: Dict[str, Dict[str, str]] = {
    "hr_bpm": {
        "display_name": "Heart Rate (bpm)",
        "unit": "bpm",
        "subsystem": "Cardiovascular / Autonomic",
        "provenance": "PUBLIC/SIMULATED (PhysioNet MMASH / WESAD)",
    },
    "derived_hrv_rmssd_ms": {
        "display_name": "HRV RMSSD Vagal Tone (ms)",
        "unit": "ms",
        "subsystem": "Autonomic Parasympathetic",
        "provenance": "DERIVED DATA (from beat-to-beat RR_interval_s)",
    },
    "derived_hrv_sdnn_ms": {
        "display_name": "HRV SDNN Total Variability (ms)",
        "unit": "ms",
        "subsystem": "Autonomic Nervous System",
        "provenance": "DERIVED DATA (rolling 5-epoch SDNN)",
    },
    "derived_sympathovagal_ratio": {
        "display_name": "Sympathovagal Index (HR / RMSSD)",
        "unit": "bpm/ms",
        "subsystem": "Autonomic Stress Balance",
        "provenance": "DERIVED DATA (HR / RMSSD ratio)",
    },
    "eda_microsiemens": {
        "display_name": "Electrodermal Activity EDA (uS)",
        "unit": "uS",
        "subsystem": "Sympathetic Sudomotor",
        "provenance": "PUBLIC/SIMULATED (UCI WESAD Wrist/Chest)",
    },
    "skin_temp_c": {
        "display_name": "Peripheral Skin Temp (degC)",
        "unit": "degC",
        "subsystem": "Thermoregulatory / Vascular",
        "provenance": "PUBLIC/SIMULATED (UCI WESAD / MMASH)",
    },
    "derived_actigraphy_enmo_g": {
        "display_name": "Wrist Actigraphy ENMO (g)",
        "unit": "g",
        "subsystem": "Locomotor / Physical Workload",
        "provenance": "DERIVED DATA (Euclidean Norm Minus One)",
    },
    "derived_borbely_sleep_pressure_s": {
        "display_name": "Borbely Sleep Pressure (S_t)",
        "unit": "index [0-1]",
        "subsystem": "Circadian / Homeostatic Sleep",
        "provenance": "DERIVED DATA (Sleep Eff & WASO composite)",
    },
    "rolling_past_workload_enmo": {
        "display_name": "Cumulative Past Physical Workload (g)",
        "unit": "g",
        "subsystem": "Musculoskeletal / Exertion",
        "provenance": "DERIVED DATA (Causal 3-lag ENMO mean)",
    },
    "autonomic_recovery_velocity": {
        "display_name": "Post-Exertion RMSSD Rebound (ms/epoch)",
        "unit": "ms/epoch",
        "subsystem": "Autonomic Recovery",
        "provenance": "DERIVED DATA (RMSSD_t - RMSSD_{t-1})",
    },
}


# ==============================================================================
# 2. DATA STRUCTURES FOR GLOBAL & INDIVIDUAL EXPLANATIONS
# ==============================================================================

@dataclass
class GlobalFeatureImportanceItem:
    """Represents global importance of a single input feature across the cohort."""
    rank: int
    feature_key: str
    display_name: str
    subsystem: str
    mean_abs_shap: float
    relative_importance_pct: float
    permutation_importance_mean: float
    permutation_importance_std: float
    direction_of_association: str


@dataclass
class LocalFeatureContribution:
    """Represents a single feature's signed contribution to an individual prediction."""
    feature_key: str
    display_name: str
    observed_value: float
    reference_baseline_mean: float
    unit: str
    shap_contribution: float
    contribution_pct_of_total_shift: float
    direction: str  # "INCREASES_PREDICTED_RISK" or "DECREASES_PREDICTED_RISK"


@dataclass
class IndividualPredictionExplanation:
    """Complete local explanation package for a single astronaut/subject observation."""
    observation_id: str
    subject_id: str
    timestamp_utc: str
    target_task: str
    explainer_method: str
    baseline_expected_value: float
    final_model_prediction: float
    total_additive_shift: float
    additivity_check_residual: float
    top_positive_drivers: List[LocalFeatureContribution]
    top_negative_stabilizers: List[LocalFeatureContribution]
    all_feature_contributions: List[LocalFeatureContribution]
    natural_language_explanation: str
    non_causation_disclaimer: str = MANDATORY_XAI_DISCLAIMERS["non_causation_statement"]
    non_diagnostic_disclaimer: str = MANDATORY_XAI_DISCLAIMERS["non_diagnostic_statement"]


# ==============================================================================
# 3. CORE EXPLAINABLE AI (XAI) ENGINE
# ==============================================================================

class AstraviveXAIEngine:
    """
    Computes:
      1. Important input features & Global feature importance (`mean(|SHAP|)` + Permutation Importance)
      2. Signed feature contribution vectors (`phi_j`) for individual predictions
      3. Exact additivity verification (`E[f(X)] + sum(phi_j) == f(x)`)
      4. Human-readable prediction explanations with explicit non-causation & non-diagnostic notices
    """

    def __init__(
        self,
        trained_pipeline: Pipeline,
        feature_cols: List[str],
        background_df: pd.DataFrame,
        target_col: str = "target_stress_binary",
        task_description: str = "Acute Autonomic Stress State Probability",
    ) -> None:
        self.pipeline = trained_pipeline
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.task_description = task_description

        # Extract fitted preprocessor steps and underlying model from the leak-free Pipeline
        self.imputer = self.pipeline.named_steps["imputer"]
        self.scaler = self.pipeline.named_steps["scaler"]
        self.model = self.pipeline.named_steps["model"]

        # Precompute transformed background distribution for baseline expectation E[f(X)]
        self.background_X_raw = background_df[self.feature_cols].copy()
        self.background_y = background_df[self.target_col].to_numpy()
        self.background_X_scaled = self._transform_features(self.background_X_raw)
        self.feature_means_raw = {
            col: float(self.background_X_raw[col].mean()) for col in self.feature_cols
        }

        # Initialize SHAP explainer (TreeExplainer / LinearExplainer if `shap` is installed,
        # otherwise exact analytical Linear/Marginal Shapley estimator)
        self.explainer_method_name = self._initialize_shap_explainer()

    def _transform_features(self, X_df: pd.DataFrame) -> np.ndarray:
        X_imp = self.imputer.transform(X_df[self.feature_cols])
        return np.asarray(self.scaler.transform(X_imp), dtype=float)

    def _predict_scalar_output(self, X_scaled: np.ndarray) -> np.ndarray:
        """Returns log-odds or probability/regression output on scaled matrix."""
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X_scaled)[:, 1], dtype=float)
        return np.asarray(self.model.predict(X_scaled), dtype=float)

    def _initialize_shap_explainer(self) -> str:
        """
        AUDIT FIX: explanations are produced directly in the same output space that
        `_predict_scalar_output` returns (probability for classifiers, raw value for
        regressors). No post-hoc proportional rescaling is applied, so signs cannot flip
        and additivity holds by construction.
        """
        self._native_shap_explainer = None
        self._native_kind = None
        model_cls_name = self.model.__class__.__name__
        rng = np.random.default_rng(RANDOM_SEED)
        n_bg = min(100, len(self.background_X_scaled))
        bg_idx = rng.choice(len(self.background_X_scaled), size=n_bg, replace=False)
        self._shap_bg = self.background_X_scaled[bg_idx]

        if SHAP_LIBRARY_AVAILABLE:
            is_tree = any(k in model_cls_name for k in ("Forest", "Boosting", "XGB", "Tree"))
            if is_tree:
                try:
                    self._native_shap_explainer = shap.TreeExplainer(
                        self.model,
                        data=self._shap_bg,
                        model_output="probability" if hasattr(self.model, "predict_proba") else "raw",
                        feature_perturbation="interventional",
                    )
                    self._native_kind = "tree"
                    return f"SHAP TreeExplainer · interventional, probability space ({model_cls_name})"
                except Exception as exc:  # e.g. unsupported model_output for this estimator
                    logger.warning(f"TreeExplainer unavailable for {model_cls_name}: {exc}")
            try:
                masker = shap.maskers.Independent(self._shap_bg)
                self._native_shap_explainer = shap.explainers.Permutation(self._predict_scalar_output, masker)
                self._native_kind = "permutation"
                return f"SHAP PermutationExplainer · probability space ({model_cls_name})"
            except Exception as exc:
                logger.warning(f"PermutationExplainer init failed: {exc}")

        return f"Permutation-Sampled Baseline Shapley (reference = background mean) ({model_cls_name})"

    def compute_shap_matrix(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Computes the (N_samples, N_features) matrix of signed SHAP contributions `phi_ij`
        and the scalar baseline `E[f(X)]` (in model-output space) such that
            f(x_i) = E[f(X)] + sum_j phi_ij     (checked; residual logged)
        """
        X_scaled = self._transform_features(X_df)
        sample_preds = self._predict_scalar_output(X_scaled)

        if self._native_kind == "tree":
            raw_shap = self._native_shap_explainer.shap_values(X_scaled, check_additivity=False)
            if isinstance(raw_shap, list):
                raw_shap = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            shap_arr = np.asarray(raw_shap, dtype=float)
            if shap_arr.ndim == 3:
                shap_arr = shap_arr[:, :, -1]
            ev = np.atleast_1d(np.asarray(self._native_shap_explainer.expected_value, dtype=float))
            expected_value = float(ev[-1])
        elif self._native_kind == "permutation":
            explanation = self._native_shap_explainer(X_scaled, max_evals=max(2 * X_scaled.shape[1] + 1, 200))
            shap_arr = np.asarray(explanation.values, dtype=float)
            expected_value = float(np.atleast_1d(explanation.base_values)[0])
        else:
            shap_arr, expected_value = self._permutation_baseline_shapley(X_scaled, sample_preds)

        residual = float(np.max(np.abs(expected_value + shap_arr.sum(axis=1) - sample_preds))) if len(X_scaled) else 0.0
        if residual > 1e-4:
            logger.warning(f"SHAP additivity residual {residual:.2e} exceeds 1e-4 ({self.explainer_method_name})")
        return shap_arr, expected_value

    def _permutation_baseline_shapley(
        self, X_scaled: np.ndarray, sample_preds: np.ndarray, n_permutations: int = 64
    ) -> Tuple[np.ndarray, float]:
        """
        Dependency-free fallback: Shapley values of the baseline game v(S) = f(x_S, ref_~S)
        with ref = background mean. Every sampled permutation path sums exactly to
        f(x) - f(ref), so the average is exactly additive (efficiency axiom) and
        unbiased for the exact Shapley value.
        """
        ref = np.mean(self.background_X_scaled, axis=0)
        expected_value = float(self._predict_scalar_output(ref.reshape(1, -1))[0])
        n_samples, n_features = X_scaled.shape
        rng = np.random.default_rng(RANDOM_SEED)
        phi = np.zeros((n_samples, n_features), dtype=float)
        for i in range(n_samples):
            for _ in range(n_permutations):
                order = rng.permutation(n_features)
                path = np.tile(ref, (n_features + 1, 1))
                for k, j in enumerate(order):
                    path[k + 1:, j] = X_scaled[i, j]
                vals = self._predict_scalar_output(path)
                phi[i, order] += np.diff(vals)
            phi[i] /= n_permutations
        return phi, expected_value

    # --------------------------------------------------------------------------
    # REQUIREMENT 1 & 5: Global Feature Importance (`mean(|SHAP|)` + Permutation)
    # --------------------------------------------------------------------------
    def compute_global_feature_importance(self, eval_df: pd.DataFrame) -> List[GlobalFeatureImportanceItem]:
        """
        Computes global feature importance across the dataset using both:
          1. Mean Absolute SHAP Value: `mean(|phi_j|)`
          2. Out-of-sample Permutation Importance (`sklearn.inspection.permutation_importance`)
        """
        shap_matrix, _ = self.compute_shap_matrix(eval_df)
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        total_abs_shap = float(np.sum(mean_abs_shap)) or 1.0

        # Compute permutation importance on the full pipeline
        X_eval = eval_df[self.feature_cols]
        y_eval = eval_df[self.target_col].to_numpy()
        perm_result = permutation_importance(
            self.pipeline, X_eval, y_eval, n_repeats=8, random_state=RANDOM_SEED
        )

        X_scaled = self._transform_features(eval_df)
        items: List[GlobalFeatureImportanceItem] = []

        for idx, col in enumerate(self.feature_cols):
            meta = FEATURE_CLINICAL_METADATA.get(
                col,
                {"display_name": col, "subsystem": "Physiological", "unit": ""},
            )
            # Determine overall direction of association (sign of corr(feature_value, shap_value))
            if np.std(X_scaled[:, idx]) > 1e-9 and np.std(shap_matrix[:, idx]) > 1e-9:
                corr = float(np.corrcoef(X_scaled[:, idx], shap_matrix[:, idx])[0, 1])
            else:
                corr = 0.0

            direction_str = (
                "Higher feature values associate with HIGHER predicted score (+)"
                if corr >= 0
                else "Higher feature values associate with LOWER predicted score (-)"
            )

            items.append(
                GlobalFeatureImportanceItem(
                    rank=0,
                    feature_key=col,
                    display_name=meta["display_name"],
                    subsystem=meta["subsystem"],
                    mean_abs_shap=round(float(mean_abs_shap[idx]), 5),
                    relative_importance_pct=round(float((mean_abs_shap[idx] / total_abs_shap) * 100.0), 2),
                    permutation_importance_mean=round(float(perm_result.importances_mean[idx]), 5),
                    permutation_importance_std=round(float(perm_result.importances_std[idx]), 5),
                    direction_of_association=direction_str,
                )
            )

        # Sort descending by mean(|SHAP|) and assign rank
        items.sort(key=lambda item: item.mean_abs_shap, reverse=True)
        for rank_idx, item in enumerate(items, start=1):
            item.rank = rank_idx

        return items

    # --------------------------------------------------------------------------
    # REQUIREMENT 2, 3 & 4: Individual Prediction Analysis & Natural-Language Summary
    # --------------------------------------------------------------------------
    def explain_individual_observation(
        self,
        obs_row: Union[pd.Series, pd.DataFrame],
        observation_id: str = "OBS-001",
    ) -> IndividualPredictionExplanation:
        """
        Generates a complete local SHAP attribution breakdown for a single observation,
        separating positive risk-elevating drivers from negative risk-reducing stabilizers,
        verifying exact additivity, and generating a transparent natural-language explanation.
        """
        single_df = obs_row.to_frame().T if isinstance(obs_row, pd.Series) else obs_row.iloc[[0]].copy()
        shap_matrix, expected_val = self.compute_shap_matrix(single_df)
        phi_vec = shap_matrix[0]

        X_scaled = self._transform_features(single_df)
        pred_val = float(self._predict_scalar_output(X_scaled)[0])
        total_shift = float(np.sum(phi_vec))
        additivity_residual = abs((expected_val + total_shift) - pred_val)
        sum_abs_phi = float(np.sum(np.abs(phi_vec))) or 1.0

        contributions: List[LocalFeatureContribution] = []
        for idx, col in enumerate(self.feature_cols):
            meta = FEATURE_CLINICAL_METADATA.get(
                col,
                {"display_name": col, "unit": "", "subsystem": "General"},
            )
            obs_val = float(single_df[col].iloc[0])
            phi_val = float(phi_vec[idx])
            contributions.append(
                LocalFeatureContribution(
                    feature_key=col,
                    display_name=meta["display_name"],
                    observed_value=round(obs_val, 4),
                    reference_baseline_mean=round(self.feature_means_raw[col], 4),
                    unit=meta["unit"],
                    shap_contribution=round(phi_val, 5),
                    contribution_pct_of_total_shift=round(float((abs(phi_val) / sum_abs_phi) * 100.0), 2),
                    direction=(
                        "INCREASES_PREDICTED_SCORE" if phi_val >= 0 else "DECREASES_PREDICTED_SCORE"
                    ),
                )
            )

        # Sort by magnitude of impact |phi_j|
        contributions.sort(key=lambda c: abs(c.shap_contribution), reverse=True)
        pos_drivers = [c for c in contributions if c.shap_contribution > 0][:3]
        neg_stabilizers = [c for c in contributions if c.shap_contribution < 0][:3]

        subj_id = str(single_df["subject_id"].iloc[0]) if "subject_id" in single_df.columns else "UNKNOWN_SUBJECT"
        ts_utc = str(single_df["timestamp_utc"].iloc[0]) if "timestamp_utc" in single_df.columns else datetime.now(timezone.utc).isoformat()

        # Construct transparent, non-causal, non-diagnostic natural language explanation
        top_pos_text = (
            ", ".join(
                f"{c.display_name} ({c.observed_value} {c.unit}, SHAP={c.shap_contribution:+.3f})"
                for c in pos_drivers
            )
            if pos_drivers
            else "no upward feature excursions"
        )
        top_neg_text = (
            ", ".join(
                f"{c.display_name} ({c.observed_value} {c.unit}, SHAP={c.shap_contribution:+.3f})"
                for c in neg_stabilizers
            )
            if neg_stabilizers
            else "no downward stabilizing offsets"
        )

        nl_summary = (
            f"For subject '{subj_id}' at {ts_utc}, the model's baseline expectation across the "
            f"reference cohort is E[f(X)] = {expected_val:.3f}, and the final predicted output for "
            f"'{self.task_description}' is f(x) = {pred_val:.3f} (net additive shift = {total_shift:+.3f}). "
            f"The primary features associated with increasing the predicted score are: {top_pos_text}. "
            f"The primary features associated with reducing/stabilizing the predicted score are: "
            f"{top_neg_text}. NOTE: These SHAP attributions reflect statistical model associations "
            f"only and do not establish biological causation or a clinical medical diagnosis."
        )

        return IndividualPredictionExplanation(
            observation_id=observation_id,
            subject_id=subj_id,
            timestamp_utc=ts_utc,
            target_task=self.task_description,
            explainer_method=self.explainer_method_name,
            baseline_expected_value=round(expected_val, 5),
            final_model_prediction=round(pred_val, 5),
            total_additive_shift=round(total_shift, 5),
            additivity_check_residual=round(additivity_residual, 9),
            top_positive_drivers=pos_drivers,
            top_negative_stabilizers=neg_stabilizers,
            all_feature_contributions=contributions,
            natural_language_explanation=nl_summary,
        )


# ==============================================================================
# 4. END-TO-END REPRODUCIBLE XAI REPORT GENERATOR
# ==============================================================================

def run_complete_xai_pipeline(output_dir: str = "models") -> Dict[str, Any]:
    """
    Trains the leak-free ASTRAVIVE classifier and forecaster, computes:
      1. Global Feature Importance (`mean(|SHAP|)` + Permutation Importance)
      2. Individual Prediction Explanations for both a Nominal Baseline window and an
         Acute Autonomic Stress Excursion window
    and saves the complete JSON artifact to `models/astravive_xai_explanations.json`.
    """
    data_pipe = AstraviveDataPipeline()
    raw_df = ClearlyLabeledSyntheticGenerator.generate_wearable_demo_stream(
        n_subjects=8, epochs_per_subject=75, inject_test_artifacts=True, random_seed=RANDOM_SEED
    )
    clean_df, _ = data_pipe.validate_and_clean(raw_df, dataset_id="MMASH-WEARABLE")
    feat_df = data_pipe.normalize_and_engineer_features(clean_df, dataset_id="MMASH-WEARABLE")
    ml_df = prepare_ml_dataset_without_leakage(feat_df)

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

    ml_engine = AstraviveMLPipeline(model_output_dir=output_dir)
    ml_engine.evaluate_classification_models(
        ml_df, feature_cols=feature_cols, target_col="target_stress_binary"
    )
    clf_bundle = ml_engine.trained_pipelines["health_state_classifier"]

    xai_engine = AstraviveXAIEngine(
        trained_pipeline=clf_bundle["pipeline"],
        feature_cols=feature_cols,
        background_df=ml_df,
        target_col="target_stress_binary",
        task_description="Acute Autonomic Stress Probability P(Stress | x)",
    )

    # 1. Global Feature Importance
    global_importance = xai_engine.compute_global_feature_importance(ml_df)

    # 2. Individual Prediction Analysis: (A) Nominal Baseline Window vs (B) Acute Stress Window
    nominal_row = ml_df.loc[ml_df["target_stress_binary"] == 0].iloc[0]
    stress_row = ml_df.loc[ml_df["target_stress_binary"] == 1].iloc[0]

    nominal_explanation = xai_engine.explain_individual_observation(
        nominal_row, observation_id="INDIV-NOMINAL-WINDOW-01"
    )
    stress_explanation = xai_engine.explain_individual_observation(
        stress_row, observation_id="INDIV-STRESS-EXCURSION-02"
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xai_json_path = out_dir / "astravive_xai_explanations.json"

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": clf_bundle["selected_model_name"],
        "explainer_method": xai_engine.explainer_method_name,
        "mandatory_disclaimers": MANDATORY_XAI_DISCLAIMERS,
        "global_feature_importance": [asdict(item) for item in global_importance],
        "individual_prediction_explanations": {
            "nominal_baseline_observation": asdict(nominal_explanation),
            "acute_stress_excursion_observation": asdict(stress_explanation),
        },
    }

    with open(xai_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.info(f"Saved complete XAI explanations to: {xai_json_path}")
    return report_payload


if __name__ == "__main__":
    print("=" * 92)
    print("ASTRAVIVE — STAGE 8 EXPLAINABLE AI (XAI) ENGINE VERIFICATION")
    print("=" * 92)
    xai_report = run_complete_xai_pipeline()
    print("\n[MANDATORY DISCLAIMERS]")
    for k, v in xai_report["mandatory_disclaimers"].items():
        print(f" - {k}: {v}\n")

    print("--- GLOBAL FEATURE IMPORTANCE (Top 6 by Mean |SHAP|) ---")
    for item in xai_report["global_feature_importance"][:6]:
        print(
            f"Rank #{item['rank']} | {item['display_name']:38s} | "
            f"Mean|SHAP|={item['mean_abs_shap']:.4f} ({item['relative_importance_pct']:5.1f}%) | "
            f"PermImp={item['permutation_importance_mean']:.4f}"
        )

    stress_exp = xai_report["individual_prediction_explanations"]["acute_stress_excursion_observation"]
    print("\n--- INDIVIDUAL PREDICTION EXPLANATION (ACUTE STRESS EXCURSION) ---")
    print(f"Baseline E[f(X)] = {stress_exp['baseline_expected_value']:.4f} -> Prediction f(x) = {stress_exp['final_model_prediction']:.4f}")
    print(f"Additivity Check Residual |E[f(X)] + sum(phi) - f(x)| = {stress_exp['additivity_check_residual']:.9f}")
    print(f"Summary: {stress_exp['natural_language_explanation']}")