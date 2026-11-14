"""
ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN
Module: Production Streamlit & Cloud Deployment Entry Point (app.py)
Target: NASA Space Apps Challenge 2026

USAGE:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from astravive_system_integrator import AstraviveSystemIntegrator

st.set_page_config(
    page_title="ASTRAVIVE — Mission-Health Digital Twin",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Remove default Streamlit padding so the NASA Mission Control console fills the viewport
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            max-width: 100% !important;
        }
        header[data-testid="stHeader"] {
            background: rgba(3, 7, 18, 0.85);
        }
        body {
            background-color: #030712;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing ASTRAVIVE 10-Stage Biomedical & Digital-Twin Engine...")
def initialize_backend_bundle() -> dict:
    """Runs the 10-stage Python pipeline on startup and caches the validated state bundle."""
    integrator = AstraviveSystemIntegrator(
        output_dir="data/integrated_bundle",
        sqlite_db_path="data/astravive.db",
    )
    bundle = integrator.execute_integrated_workflow(mission_duration_sols=90)
    return {
        "status": bundle.overall_system_status,
        "timestamp_utc": bundle.integration_timestamp_utc,
        "components_count": len(bundle.component_health_checks),
        "data_mode": bundle.data_mode,
        "bundle_path": bundle.exported_files.get("unified_bundle_json", ""),
    }


def main() -> None:
    backend_meta = initialize_backend_bundle()

    with st.sidebar:
        st.markdown("### ASTRAVIVE Python Backend")
        status = backend_meta["status"]
        (st.success if status == "OPERATIONAL" else st.warning)(f"Backend status: **{status}**")
        if backend_meta.get("data_mode") == "SYNTHETIC_DEMO":
            st.info("No local dataset found: the backend ran on clearly labeled SIMULATED demo data.")
        st.caption(f"Initialized UTC: `{backend_meta['timestamp_utc']}`")
        if st.button("Re-run 10-Stage Python Pipeline & ML Cross-Validation"):
            st.cache_resource.clear()
            st.rerun()

        bundle_file = Path(backend_meta["bundle_path"])
        if bundle_file.exists():
            st.download_button(
                label="Download Unified Backend Bundle (.json)",
                data=bundle_file.read_text(encoding="utf-8"),
                file_name="astravive_unified_system_bundle.json",
                mime="application/json",
            )

    html_file = Path("astravive_app.html")
    if html_file.exists():
        html_content = html_file.read_text(encoding="utf-8")
        # AUDIT FIX: iframe height was hard-coded; make it configurable (default fits the tallest tab).
        try:
            height = max(600, min(6000, int(os.environ.get("ASTRAVIVE_IFRAME_HEIGHT", "2200"))))
        except ValueError:
            height = 2200
        components.html(html_content, height=height, scrolling=True)
    else:
        st.error(
            "CRITICAL: `astravive_app.html` not found in the root directory. "
            "Ensure `astravive_app.html` is placed in the same folder as `app.py`."
        )


if __name__ == "__main__":
    main()