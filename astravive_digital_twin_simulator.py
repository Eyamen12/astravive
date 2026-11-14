"""
ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN
Module: Interactive Astronaut Health Digital-Twin Research Simulator (Stage 7 & Stage 9)
Target: NASA Space Apps Challenge 2026

SCIENTIFIC & REGULATORY RESPONSIBILITY NOTICE:
1. NOT A CLINICALLY VALIDATED ASTRONAUT PHYSIOLOGICAL TWIN:
   This simulator is an exploratory biomedical research prototype and mathematical what-if
   sandbox developed for the NASA Space Apps Challenge 2026. It is NOT a clinically validated
   astronaut physiological twin, nor is it certified for operational flight-surgeon medical
   decision-making.
2. STRICT SEPARATION OF PROVENANCE CATEGORIES:
   Every point in the simulator output is explicitly tagged to distinguish:
   - PUBLIC / MEASURED ANCHORS (`PUBLIC DATA`): Pre-flight/post-flight empirical anchor points
     from NASA OSDR (OSD-575/569, OSD-51) and terrestrial baseline priors (PhysioNet MMASH).
   - MODEL PREDICTIONS (`MODEL PREDICTION`): Short-horizon (Sol 1 to Sol 7) autoregressive and
     tree-ensemble predictions from the trained Scikit-learn pipeline (`g_ML`).
   - SIMULATED OUTPUTS (`SIMULATED DATA`): Multi-sol parametric state-space ODE/difference
     equation rollouts (`f_mech`) driven by documented simulation assumptions.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from astravive_data_pipeline import DataClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | ASTRAVIVE-TWIN-SIM | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("ASTRAVIVE_DigitalTwinSimulator")


# ==============================================================================
# 1. EXPLICIT EQUATIONS, COEFFICIENTS, ASSUMPTIONS & LIMITATIONS REGISTRY
# ==============================================================================

DIGITAL_TWIN_MATHEMATICAL_DOCUMENTATION: Dict[str, Any] = {
    "scientific_disclaimer": (
        "RESEARCH PROTOTYPE DISCLAIMER: This system is an exploratory mathematical digital-twin "
        "research simulator and NOT a clinically validated astronaut physiological twin or medical "
        "diagnostic device. All long-horizon trajectories beyond empirical observation windows "
        "are parametric simulations governed by the equations and assumptions documented below."
    ),
    "governing_hybrid_state_equation": (
        "x_{t+1} = f_mech(x_t, u, e; theta)   [this build: mechanistic ODE only; the ML residual "
        "term w_ML(t) * g_ML(...) is NOT applied, so every Sol > 0 is a parametric SIMULATION]"
    ),
    "state_variables": {
        "S_t": "Homeostatic Sleep Pressure / Fatigue Index in [0, 100]% (lower is better rested)",
        "RMSSD_t": "Autonomic Vagal Tone (HRV RMSSD in ms; higher indicates stronger parasympathetic recovery)",
        "BMD_delta_pct_t": "Cumulative Weight-Bearing Bone Mineral Density Change (% relative to L-3 pre-flight)",
        "SANS_proxy_t": "Neuro-Ocular / Cephalad Fluid Congestion Risk Proxy in [0, 100]%",
        "Readiness_t": "Composite Mission-Health Readiness Score in [0, 100]%",
    },
    "adjustable_parameters_supported": {
        "sleep_duration_hrs": {
            "range": [4.0, 10.0],
            "default_scenario_a": 6.0,
            "default_scenario_b": 8.0,
            "unit": "hours/day",
            "supporting_source": "PhysioNet MMASH (sleep.csv) & Sleep-EDFx Borbely Two-Process Priors",
        },
        "recovery_duration_hrs": {
            "range": [0.5, 4.0],
            "default_scenario_a": 1.0,
            "default_scenario_b": 2.5,
            "unit": "hours/day",
            "supporting_source": "Post-exertion parasympathetic rebound time constant (MMASH Activity.csv + RR.csv)",
        },
        "workload_intensity_pct": {
            "range": [20.0, 100.0],
            "default_scenario_a": 80.0,
            "default_scenario_b": 55.0,
            "unit": "% of peak EVA/operational schedule load",
            "supporting_source": "UCI WESAD & MMASH ENMO physical/cognitive strain coupling",
        },
        "exercise_duration_min": {
            "range": [0.0, 180.0],
            "default_scenario_a": 45.0,
            "default_scenario_b": 120.0,
            "unit": "minutes/day (ARED resistive + treadmill protocol)",
            "supporting_source": "NASA OSDR OSD-51 (WISE-2005 Bed Rest Exercise Countermeasure) & NASA HRP BMD priors",
        },
        "cabin_co2_mmhg": {
            "range": [0.5, 6.0],
            "default_scenario_a": 3.8,
            "default_scenario_b": 2.0,
            "unit": "mmHg partial pressure (1 mmHg ≈ 1,316 ppm at 760 mmHg)",
            "supporting_source": "NASA OSDR EDA ISS Cabin CO2 Telemetry & PhysioNet CVES Cerebral CO2 Reactivity",
        },
    },
    "documented_equations_and_coefficients": {
        # AUDIT FIX: equations below are transcribed 1:1 from `_simulate_single_scenario`.
        # w = workload_pct / 100, g = gravity_g, wake = 24 - sleep_hrs. Verified by test_07.
        "1_sleep_and_fatigue_dynamics": {
            "equation": (
                "S_{t+1} = clip(S_t + 0.22 * [1.15 * (wake - 16) + 8.5 * (w - 0.55) - 2.65 * (sleep - 7.5) "
                "- 1.40 * (rec - 1.5) + 1.25 * max(0, pCO2 - 2.5)], 6, 95);  S_0 = 22"
            ),
            "coefficients": {"gain": 0.22, "alpha_wake": 1.15, "k_workload": 8.5, "beta_sleep": 2.65,
                             "gamma_rec": 1.40, "delta_co2_sleep": 1.25},
            "assumption": (
                "Discrete daily Borbely-style homeostatic balance: wake time and workload above the "
                "reference schedule (16 h wake, 55% load, 7.5 h sleep, 1.5 h recovery) accumulate sleep "
                "pressure; cabin pCO2 > 2.5 mmHg adds a sleep-fragmentation penalty."
            ),
        },
        "2_autonomic_hrv_rmssd_dynamics": {
            "equation": (
                "target = 48.2 - 14.0 * (w - 0.50) + 3.6 * (rec - 1.5) - 0.24 * (S_{t+1} - 22) "
                "+ 0.04 * min(ex, 90);  RMSSD_{t+1} = clip(RMSSD_t + 0.28 * (target - RMSSD_t), 14, 82);  RMSSD_0 = 48.2"
            ),
            "coefficients": {"RMSSD_base": 48.2, "k_revert": 0.28, "k_work": 14.0, "k_rec": 3.6,
                             "k_fatigue": 0.24, "k_aero": 0.04},
            "assumption": (
                "Vagal tone relaxes toward a load-dependent target: suppressed by workload and sleep "
                "pressure, restored by recovery blocks and up to 90 min/day of conditioning."
            ),
        },
        "3_musculoskeletal_bmd_dynamics": {
            "equation": (
                "BMD_{t+1} = BMD_t - 0.0433 * (1 - g) + 0.0330 * (ex / 120) "
                "- 0.012 * max(0, w - 0.70) * max(0, (8 - sleep) / 4);  BMD_0 = 0"
            ),
            "coefficients": {"r_ug_daily": 0.0433, "r_ex_gain": 0.0330, "r_overwork_catabolic": 0.012},
            "assumption": (
                "0.0433 %/day ~= 1.3 %/month unmitigated weight-bearing BMD loss (within the NASA HRP "
                "1.0-1.5 %/month range); 120 min/day exercise offsets ~76% of that loss."
            ),
        },
        "4_neuro_ocular_sans_proxy_dynamics": {
            "equation": (
                "SANS_{t+1} = clip(10 + 24 * (1 - exp(-(t+1) / 18)) * (1 - g) + 8.4 * max(0, pCO2 - 2.0)^1.25 "
                "- 4.5 * (rec / 3) + 0.15 * max(0, S_{t+1} - 40), 5, 96);  SANS_0 = 10"
            ),
            "coefficients": {"SANS_0": 10.0, "c_ug": 24.0, "tau_fluid": 18.0, "c_co2": 8.4,
                             "c_rec_drain": 4.5, "c_fatigue_icp": 0.15},
            "assumption": (
                "Exponential cephalad fluid-shift saturation (tau = 18 sols) plus a power-law hypercapnia "
                "term above 2.0 mmHg. This is a risk PROXY, not a measured optic-disc metric."
            ),
        },
        "5_composite_readiness_index": {
            "equation": (
                "Readiness_t = 0.25 * (100 - S_t) + 0.25 * clip(RMSSD_t / 55 * 100, 20, 100) "
                "+ 0.25 * clip(100 - |BMD_t| * 8, 20, 100) + 0.25 * (100 - SANS_t)"
            ),
        },
        "6_derived_resting_heart_rate": {
            "equation": "HR_t = 64 + 0.55 * (48.2 - RMSSD_t) + 14 * (w - 0.5) + 0.12 * (S_t - 22)",
        },
    },
    "limitations": [
        "1. Population-Level Priors: Mechanistic rate constants represent published group means and do not capture patient-specific genetic or anatomical variance (e.g., optic nerve sheath anatomy in SANS).",
        "2. Absence of Continuous In-Flight Wearable Validation: Open NASA repositories (OSDR) release discrete blood/omics panels rather than continuous in-flight wearable waveforms.",
        "3. First-Order Coupling Approximation: Cross-subsystem interactions (such as hypercapnia affecting sleep fragmentation and cerebral blood flow) are modeled via first-order linear/power-law terms.",
    ],
}


# ==============================================================================
# 2. SCENARIO PARAMETER & TRAJECTORY DATA STRUCTURES
# ==============================================================================

@dataclass
class ScenarioParameters:
    """User-configurable input parameters for Scenario A (Baseline) or Scenario B (Modified)."""
    scenario_label: str
    mission_duration_sols: int = 90
    gravity_g: float = 0.0
    sleep_duration_hrs: float = 6.0
    recovery_duration_hrs: float = 1.0
    workload_intensity_pct: float = 80.0
    exercise_duration_min: float = 45.0
    cabin_co2_mmhg: float = 3.8

    def validate_bounds(self) -> None:
        """Enforces documented parameter domain bounds."""
        self.mission_duration_sols = int(np.clip(self.mission_duration_sols, 14, 900))
        self.gravity_g = float(np.clip(self.gravity_g, 0.0, 1.0))
        self.sleep_duration_hrs = float(np.clip(self.sleep_duration_hrs, 4.0, 10.0))
        self.recovery_duration_hrs = float(np.clip(self.recovery_duration_hrs, 0.5, 4.0))
        self.workload_intensity_pct = float(np.clip(self.workload_intensity_pct, 20.0, 100.0))
        self.exercise_duration_min = float(np.clip(self.exercise_duration_min, 0.0, 180.0))
        self.cabin_co2_mmhg = float(np.clip(self.cabin_co2_mmhg, 0.5, 6.0))


@dataclass
class SimulationComparisonResult:
    """Complete side-by-side output package comparing Scenario A and Scenario B."""
    generated_at_utc: str
    non_clinical_disclaimer: str
    scenario_a_params: Dict[str, Any]
    scenario_b_params: Dict[str, Any]
    trajectory_records: List[Dict[str, Any]]
    side_by_side_summary: Dict[str, Any]
    scenario_explanation: str
    equations_and_assumptions: Dict[str, Any]


# ==============================================================================
# 3. DIGITAL-TWIN RESEARCH SIMULATOR ENGINE
# ==============================================================================

class AstronautDigitalTwinSimulator:
    """
    Executes parallel rollouts for:
      - SCENARIO A — BASELINE CONDITIONS
      - SCENARIO B — MODIFIED CONDITIONS
    Clearly separating:
      - Sol 0 Anchor (`PUBLIC DATA` — calibrated to NASA OSDR OSD-575/569 & PhysioNet MMASH)
      - Sol 1..N Rollout (`SIMULATED DATA` — parametric ODE state-space regime). AUDIT FIX:
        Sols 1-7 were previously labeled MODEL PREDICTION although no ML model is called here.
    """

    def __init__(self) -> None:
        self.doc = DIGITAL_TWIN_MATHEMATICAL_DOCUMENTATION

    @staticmethod
    def get_default_scenarios(mission_duration_sols: int = 90) -> Tuple[ScenarioParameters, ScenarioParameters]:
        """Returns default Scenario A (Baseline Stressed) and Scenario B (Modified Countermeasure)."""
        scenario_a = ScenarioParameters(
            scenario_label="SCENARIO A — BASELINE CONDITIONS",
            mission_duration_sols=mission_duration_sols,
            gravity_g=0.0,
            sleep_duration_hrs=6.0,
            recovery_duration_hrs=1.0,
            workload_intensity_pct=80.0,
            exercise_duration_min=45.0,
            cabin_co2_mmhg=3.8,
        )
        scenario_b = ScenarioParameters(
            scenario_label="SCENARIO B — MODIFIED CONDITIONS",
            mission_duration_sols=mission_duration_sols,
            gravity_g=0.0,
            sleep_duration_hrs=8.0,
            recovery_duration_hrs=2.5,
            workload_intensity_pct=55.0,
            exercise_duration_min=120.0,
            cabin_co2_mmhg=2.0,
        )
        return scenario_a, scenario_b

    def _simulate_single_scenario(self, params: ScenarioParameters) -> List[Dict[str, Any]]:
        """Runs the discrete daily hybrid state-space equations for a single scenario."""
        params.validate_bounds()
        n_sols = params.mission_duration_sols
        workload_norm = params.workload_intensity_pct / 100.0
        g_norm = params.gravity_g

        # Initial Sol 0 State (Empirical Baseline Anchor from MMASH & OSDR L-3 Pre-flight)
        sleep_pressure_s = 22.0
        rmssd_ms = 48.2
        bmd_delta_pct = 0.0
        sans_proxy = 10.0

        records: List[Dict[str, Any]] = []

        for sol in range(0, n_sols + 1):
            if sol == 0:
                data_cls = DataClassification.PUBLIC_DATA.value
                origin_note = "MEASURED/PUBLIC BASELINE ANCHOR (OSDR L-3 & PhysioNet MMASH)"
            else:
                data_cls = DataClassification.SIMULATED_DATA.value
                origin_note = "PARAMETRIC ODE DIGITAL-TWIN SIMULATION (NASA HRP Priors)"

            # Compute composite readiness at current Sol
            sleep_readiness = 100.0 - sleep_pressure_s
            autonomic_readiness = float(np.clip((rmssd_ms / 55.0) * 100.0, 20.0, 100.0))
            bmd_readiness = float(np.clip(100.0 - abs(bmd_delta_pct) * 8.0, 20.0, 100.0))
            sans_readiness = 100.0 - sans_proxy
            composite_readiness = (
                0.25 * sleep_readiness
                + 0.25 * autonomic_readiness
                + 0.25 * bmd_readiness
                + 0.25 * sans_readiness
            )

            # Derived Heart Rate (bpm) inversely coupled to RMSSD and positively coupled to workload/fatigue
            hr_bpm = 64.0 + (48.2 - rmssd_ms) * 0.55 + (workload_norm - 0.5) * 14.0 + (sleep_pressure_s - 22.0) * 0.12

            records.append({
                "mission_sol": sol,
                "scenario_label": params.scenario_label,
                "data_classification": data_cls,
                "origin_note": origin_note,
                "sleep_pressure_index": round(float(sleep_pressure_s), 2),
                "sleep_readiness_pct": round(float(sleep_readiness), 2),
                "hrv_rmssd_ms": round(float(rmssd_ms), 2),
                "resting_hr_bpm": round(float(hr_bpm), 2),
                "bmd_delta_pct": round(float(bmd_delta_pct), 3),
                "sans_risk_index": round(float(sans_proxy), 2),
                "composite_readiness_pct": round(float(composite_readiness), 2),
            })

            # Step state equations forward from Sol t -> Sol t+1
            wake_hrs = 24.0 - params.sleep_duration_hrs
            co2_sleep_penalty = 1.25 * max(0.0, params.cabin_co2_mmhg - 2.5)
            d_sleep = (
                1.15 * (wake_hrs - 16.0)
                + 8.5 * (workload_norm - 0.55)
                - 2.65 * (params.sleep_duration_hrs - 7.5)
                - 1.40 * (params.recovery_duration_hrs - 1.5)
                + co2_sleep_penalty
            )
            sleep_pressure_s = float(np.clip(sleep_pressure_s + 0.22 * d_sleep, 6.0, 95.0))

            # Autonomic RMSSD update
            rmssd_target = (
                48.2
                - 14.0 * (workload_norm - 0.50)
                + 3.6 * (params.recovery_duration_hrs - 1.5)
                - 0.24 * (sleep_pressure_s - 22.0)
                + 0.04 * min(params.exercise_duration_min, 90.0)
            )
            rmssd_ms = float(np.clip(rmssd_ms + 0.28 * (rmssd_target - rmssd_ms), 14.0, 82.0))

            # Musculoskeletal BMD % update
            daily_ug_loss = -0.0433 * (1.0 - g_norm)
            daily_ex_mitigation = 0.0330 * (params.exercise_duration_min / 120.0)
            daily_catabolic_strain = -0.012 * max(0.0, workload_norm - 0.70) * max(0.0, (8.0 - params.sleep_duration_hrs) / 4.0)
            bmd_delta_pct = float(bmd_delta_pct + daily_ug_loss + daily_ex_mitigation + daily_catabolic_strain)

            # Neuro-Ocular SANS proxy update
            fluid_shift_term = 24.0 * (1.0 - math.exp(-(sol + 1) / 18.0)) * (1.0 - g_norm)
            co2_vasodilation_term = 8.4 * (max(0.0, params.cabin_co2_mmhg - 2.0) ** 1.25)
            recovery_drain_term = -4.5 * (params.recovery_duration_hrs / 3.0)
            fatigue_icp_term = 0.15 * max(0.0, sleep_pressure_s - 40.0)
            sans_proxy = float(
                np.clip(10.0 + fluid_shift_term + co2_vasodilation_term + recovery_drain_term + fatigue_icp_term, 5.0, 96.0)
            )

        return records

    def run_comparison(
        self,
        scenario_a: ScenarioParameters,
        scenario_b: ScenarioParameters,
    ) -> SimulationComparisonResult:
        """
        Simulates Scenario A (Baseline) and Scenario B (Modified) side-by-side, computes
        terminal and mean deltas, and generates an automated natural-language explanation.
        """
        # AUDIT FIX: both scenarios must share one horizon, otherwise "terminal" deltas compare
        # different Sols. Scenario A's (bounded) duration is authoritative.
        scenario_a.validate_bounds()
        if scenario_b.mission_duration_sols != scenario_a.mission_duration_sols:
            logger.warning(
                f"Scenario B duration {scenario_b.mission_duration_sols} != A {scenario_a.mission_duration_sols}; "
                "aligning B to A for a like-for-like comparison."
            )
            scenario_b.mission_duration_sols = scenario_a.mission_duration_sols
        traj_a = self._simulate_single_scenario(scenario_a)
        traj_b = self._simulate_single_scenario(scenario_b)

        end_a = traj_a[-1]
        end_b = traj_b[-1]

        deltas = {
            "composite_readiness_delta_pct": round(end_b["composite_readiness_pct"] - end_a["composite_readiness_pct"], 2),
            "hrv_rmssd_delta_ms": round(end_b["hrv_rmssd_ms"] - end_a["hrv_rmssd_ms"], 2),
            "bmd_preservation_delta_pct": round(end_b["bmd_delta_pct"] - end_a["bmd_delta_pct"], 3),
            "sans_risk_reduction_pct": round(end_b["sans_risk_index"] - end_a["sans_risk_index"], 2),
            "sleep_pressure_reduction_pct": round(end_b["sleep_pressure_index"] - end_a["sleep_pressure_index"], 2),
        }

        explanation = (
            f"Over a {scenario_a.mission_duration_sols}-Sol simulation, modifying conditions from "
            f"Scenario A (Sleep={scenario_a.sleep_duration_hrs}h, Recovery={scenario_a.recovery_duration_hrs}h, "
            f"Workload={scenario_a.workload_intensity_pct}%, Exercise={scenario_a.exercise_duration_min}m, "
            f"Cabin pCO2={scenario_a.cabin_co2_mmhg} mmHg) to Scenario B "
            f"(Sleep={scenario_b.sleep_duration_hrs}h, Recovery={scenario_b.recovery_duration_hrs}h, "
            f"Workload={scenario_b.workload_intensity_pct}%, Exercise={scenario_b.exercise_duration_min}m, "
            f"Cabin pCO2={scenario_b.cabin_co2_mmhg} mmHg) shifts end-of-mission Composite Readiness from "
            f"{end_a['composite_readiness_pct']:.1f}% to {end_b['composite_readiness_pct']:.1f}% "
            f"({deltas['composite_readiness_delta_pct']:+.1f}%). Specifically, HRV RMSSD shifts by "
            f"{deltas['hrv_rmssd_delta_ms']:+.1f} ms, femoral BMD loss changes from "
            f"{end_a['bmd_delta_pct']:.2f}% to {end_b['bmd_delta_pct']:.2f}% "
            f"({deltas['bmd_preservation_delta_pct']:+.2f}%), and the SANS neuro-ocular proxy shifts from "
            f"{end_a['sans_risk_index']:.1f}% to {end_b['sans_risk_index']:.1f}% "
            f"({deltas['sans_risk_reduction_pct']:+.1f}%). Note: Sol 0 reflects literature-anchored PUBLIC DATA "
            f"baseline values; Sols 1–{scenario_a.mission_duration_sols} are parametric SIMULATED OUTPUTS."
        )

        combined_records: List[Dict[str, Any]] = []
        for row_a, row_b in zip(traj_a, traj_b):
            combined_records.append({
                "mission_sol": row_a["mission_sol"],
                "data_classification": row_a["data_classification"],
                "origin_note": row_a["origin_note"],
                "scenario_a_readiness_pct": row_a["composite_readiness_pct"],
                "scenario_b_readiness_pct": row_b["composite_readiness_pct"],
                "scenario_a_rmssd_ms": row_a["hrv_rmssd_ms"],
                "scenario_b_rmssd_ms": row_b["hrv_rmssd_ms"],
                "scenario_a_bmd_pct": row_a["bmd_delta_pct"],
                "scenario_b_bmd_pct": row_b["bmd_delta_pct"],
                "scenario_a_sans_index": row_a["sans_risk_index"],
                "scenario_b_sans_index": row_b["sans_risk_index"],
                "scenario_a_sleep_pressure": row_a["sleep_pressure_index"],
                "scenario_b_sleep_pressure": row_b["sleep_pressure_index"],
            })

        return SimulationComparisonResult(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            non_clinical_disclaimer=self.doc["scientific_disclaimer"],
            scenario_a_params=asdict(scenario_a),
            scenario_b_params=asdict(scenario_b),
            trajectory_records=combined_records,
            side_by_side_summary={
                "terminal_sol": scenario_a.mission_duration_sols,
                "scenario_a_terminal": end_a,
                "scenario_b_terminal": end_b,
                "net_effect_b_minus_a": deltas,
            },
            scenario_explanation=explanation,
            equations_and_assumptions=self.doc,
        )

    def export_simulation_results(
        self,
        result: SimulationComparisonResult,
        output_dir: str = "data/processed",
    ) -> Dict[str, str]:
        """Exports the side-by-side Scenario A vs Scenario B results to CSV and JSON."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        csv_path = out_path / "astravive_digital_twin_scenarios_ab.csv"
        json_path = out_path / "astravive_digital_twin_scenarios_ab.json"

        pd.DataFrame(result.trajectory_records).to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(f"Exported Digital Twin Scenario A/B comparison to {csv_path} and {json_path}")
        return {"csv_path": str(csv_path), "json_path": str(json_path)}


if __name__ == "__main__":
    print("=" * 92)
    print("ASTRAVIVE — INTERACTIVE DIGITAL-TWIN RESEARCH SIMULATOR (SCENARIO A vs. SCENARIO B)")
    print("=" * 92)
    sim = AstronautDigitalTwinSimulator()
    scen_a, scen_b = sim.get_default_scenarios(mission_duration_sols=90)
    comp = sim.run_comparison(scen_a, scen_b)
    paths = sim.export_simulation_results(comp)
    print("\n[DISCLAIMER]:", comp.non_clinical_disclaimer)
    print("\n[SCENARIO EXPLANATION]:\n", comp.scenario_explanation)
    print("\n[EXPORTED FILES]:", json.dumps(paths, indent=2))