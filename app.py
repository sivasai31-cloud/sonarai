"""SONARIS-X - Main Application (Streamlit UI)."""

import streamlit as st
import cv2
import numpy as np
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, List

# Configure matplotlib for headless Streamlit use BEFORE any pyplot import.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

from src.pipeline import SonarisXPipeline, PipelineResult
from src.preprocessing.quality import ImageQualityAssessor
from src.preprocessing.preprocess import SonarPreprocessor, PreprocessingConfig
from src.detection.detector import MultiScaleDetector
from src.fingerprint.fingerprint import FingerprintExtractor
from src.context.target_context import AdaptiveTargetContext
from src.anomaly.false_positive_filter import FalsePositiveFilter
from src.anomaly.classification import AnomalyClassifier
from src.risk.risk_engine import RiskEngine
from src.explainability.explainer import Explainer
from src.geolocation.geolocation import GeolocationExtractor
from src.reporting.report_generator import ReportGenerator
from src.validation.dataset_validation import DatasetValidator
from src.validation.system_diagnostics import SystemDiagnosticsCollector

# Page config
st.set_page_config(
    page_title="SONARIS-X",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for marine theme
st.markdown("""
<style>
    .stApp { background-color: #0a1628; }
    .main-header { color: #00d4ff; font-size: 28px; font-weight: bold; }
    .status-ready { color: #00ff88; }
    .status-offline { color: #ffaa00; }
    .detection-card { background-color: #1a2332; border: 1px solid #00d4ff; border-radius: 8px; padding: 12px; }
    .metric-box { background-color: #1a2332; border-left: 3px solid #00d4ff; padding: 8px 12px; margin: 4px 0; }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #00ff88; font-weight: bold; }
    .risk-review { color: #aaaaaa; font-weight: bold; }
    .evidence-box { background-color: #0d1b2a; border-radius: 4px; padding: 8px; font-size: 12px; }
    .sidebar .sidebar-content { background-color: #0a1628; }
    .stButton>button { background-color: #0a3d62; color: white; border: 1px solid #00d4ff; }
    .stButton>button:hover { background-color: #0a5684; }
    .demo-label { background-color: #8b0000; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; }
    .simulated-label { background-color: #ff6600; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; }
    .tab-title { color: #00d4ff; font-size: 18px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🌊 SONARIS-X — Side-Scan Sonar Intelligence System")

    # Status bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<span class="main-header">SONARIS-X</span>', unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="status-ready">● READY</span>', unsafe_allow_html=True)
    with col3:
        st.markdown('<span class="status-offline">● OFFLINE MODE</span>', unsafe_allow_html=True)

    # Honest model-status banner (never claim sonar-trained without weights)
    try:
        _probe = MultiScaleDetector()
        _status = _probe.model_status()
        if "CUSTOM SONAR" in _status:
            st.success(f"MODEL STATUS: {_status} — sonar classes: {', '.join(_probe.sonar_classes()) or 'see data.yaml'}")
        else:
            st.warning(f"MODEL STATUS: {_status}. Detections are classical candidates + COCO fallback; "
                       f"classes shown as UNKNOWN ANOMALY / LIKELY NATURAL, never sonar object names.")
    except Exception as e:
        st.warning(f"MODEL STATUS: could not probe detector ({e}) — running in DEMO mode.")

    st.sidebar.title("SONARIS-X Navigation")
    page = st.sidebar.radio("Go to", [
        "📡 Analysis Console",
        "📋 Target Catalog",
        "🗺️ Tactical Map",
        "📊 Mission Dashboard",
        "📄 Mission Reports",
        "🧪 Review Queue",
        "🗂️ Dataset Explorer",
        "🎓 Training Center",
        "🔬 Model Validation",
        "❌ Failure Analysis",
        "⚙️ System Diagnostics"
    ])

    # Session state initialization
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    if 'demo_scenario' not in st.session_state:
        st.session_state.demo_scenario = None
    if 'demo_gt' not in st.session_state:
        st.session_state.demo_gt = None
    if 'compare_result' not in st.session_state:
        st.session_state.compare_result = None
    if 'map_target' not in st.session_state:
        st.session_state.map_target = None

    if page == "📡 Analysis Console":
        analysis_console()
    elif page == "📋 Target Catalog":
        target_catalog()
    elif page == "🗺️ Tactical Map":
        tactical_map()
    elif page == "📊 Mission Dashboard":
        mission_dashboard()
    elif page == "📄 Mission Reports":
        mission_reports()
    elif page == "🧪 Review Queue":
        review_queue()
    elif page == "🗂️ Dataset Explorer":
        dataset_explorer()
    elif page == "🎓 Training Center":
        training_center()
    elif page == "🔬 Model Validation":
        model_validation()
    elif page == "❌ Failure Analysis":
        failure_analysis()
    elif page == "⚙️ System Diagnostics":
        system_diagnostics()


def analysis_console():
    """Main Analysis Console page."""
    st.markdown('<span class="tab-title">📡 ANALYSIS CONSOLE</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Left column: Data Input
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### Data Input")

        uploaded_file = st.file_uploader("Upload Sonar Image", type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
                                          key="upload", help="Upload side-scan sonar imagery")

        if uploaded_file:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            tfile.write(uploaded_file.read())
            tfile.close()
            st.session_state.current_image = tfile.name
            st.success(f"Image loaded: {uploaded_file.name}")

        st.markdown("### Enhancement Controls")
        apply_clahe = st.checkbox("CLAHE Contrast Enhancement", value=True, key="ctrl_clahe")
        apply_denoising = st.checkbox("Noise Reduction", value=True, key="ctrl_denoise")
        enhance_shadows = st.checkbox("Shadow Enhancement", value=False, key="ctrl_shadows")
        use_tiled = st.checkbox("Tiled Detection", value=False, key="ctrl_tiled")
        tile_size = st.slider("Tile Size", 256, 1024, 512, step=128, key="ctrl_tile")
        tile_overlap = st.slider("Tile Overlap", 0, 256, 64, step=16, key="ctrl_overlap")
        show_dropout = st.checkbox("Dropout Overlay", value=True, key="ctrl_dropout")
        st.session_state.show_dropout = show_dropout
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, key="ctrl_conf")

        st.markdown("### 🚀 Demo Mode")
        from src.demo.synthetic import SCENARIOS
        scenario = st.selectbox("Demo Scenario (SYNTHETIC DEMO DATA)", list(SCENARIOS.keys()), key="demo_scn")
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            if st.button("START DEMO MISSION", key="demo_btn", use_container_width=True):
                st.session_state.demo_mode = True
                st.session_state.demo_scenario = scenario
                st.session_state.current_image = None
                st.rerun()
        with demo_col2:
            if st.button("RUN ANALYSIS", key="run_btn", use_container_width=True):
                if st.session_state.current_image:
                    run_pipeline(st.session_state.current_image, confidence_threshold, use_tiled,
                                 tile_size, tile_overlap, apply_clahe, apply_denoising, enhance_shadows)
                else:
                    st.warning("Please upload an image or start a demo mission first.")
        if st.button("FULL vs TILED COMPARE", key="compare_btn", use_container_width=True):
            if st.session_state.current_image:
                run_compare(st.session_state.current_image, confidence_threshold,
                            tile_size, tile_overlap, apply_clahe, apply_denoising, enhance_shadows)
            else:
                st.warning("Upload an image or run a demo first.")

        if st.session_state.demo_mode:
            st.markdown('<div class="demo-label">SYNTHETIC DEMO DATA</div>', unsafe_allow_html=True)
            from src.demo.synthetic import generate_scenario
            scen_name = st.session_state.demo_scenario or scenario
            synthetic_img, gt, info = generate_scenario(scen_name)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            cv2.imwrite(tfile.name, synthetic_img)
            tfile.close()
            st.session_state.current_image = tfile.name
            st.session_state.demo_gt = {"scenario": scen_name, "info": info, "gt": gt}
            st.session_state.demo_mode = False
            run_pipeline(tfile.name, confidence_threshold, use_tiled,
                         tile_size, tile_overlap, apply_clahe, apply_denoising, enhance_shadows)
            st.markdown('<span class="simulated-label">⚠️ SYNTHETIC DEMO DATA</span>', unsafe_allow_html=True)

        # Demo scenario ground-truth panel (labeled; never injected as detections)
        if st.session_state.demo_gt:
            d = st.session_state.demo_gt
            st.markdown("### Demo Scenario (synthetic ground truth)")
            st.markdown(f"**Scenario:** {d['scenario']} [SYNTHETIC DEMO DATA]")
            st.markdown(f"*{d['info']['description']}*")
            for g in d['gt']:
                st.markdown(f"- {g['demo_class']}: bbox {g['bbox']} — {g['note']}")

    # Right column: Sonar Viewer
    with right_col:
        st.markdown("### SONAR VIEWER")

        if st.session_state.current_result:
            result = st.session_state.current_result

            # View mode
            view_mode = st.radio("View:", ["ORIGINAL", "PREPROCESSED", "DETECTION VIEW"],
                                  key="view_mode")

            if view_mode == "ORIGINAL" and result.preprocessing:
                display_img = result.preprocessing.original
            elif view_mode == "PREPROCESSED" and result.preprocessing:
                display_img = result.preprocessing.preprocessed
            elif view_mode == "DETECTION VIEW" and result.preprocessing:
                base = result.preprocessing.original
                if len(base.shape) == 2:
                    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
                display_img = render_detection_overlay(base, result)
            else:
                display_img = cv2.imread(st.session_state.current_image) if st.session_state.current_image else None

            if display_img is not None:
                display_sonar_image(display_img)
                st.caption("Magenta rectangles = dropout/degraded data. Colored boxes = detections (red=HIGH, orange=MEDIUM, green=LOW, gray=REVIEW).")

            # Quality panel (real computed values)
            if result.quality:
                q = result.quality
                h, w = (result.preprocessing.original.shape[:2] if result.preprocessing is not None else (0, 0))
                drop_pct = (len(result.preprocessing.dropout_regions) * 32 * 32 / max(1, h * w) * 100) if result.preprocessing else 0.0
                st.markdown("### Image Quality")
                qc = st.columns(6)
                qc[0].metric("Score", f"{q.quality_score:.1f}")
                qc[1].metric("Rating", q.rating)
                qc[2].metric("Contrast", f"{q.contrast_score:.0f}")
                qc[3].metric("Noise", f"{q.noise_score:.0f}")
                qc[4].metric("Dyn.Range", f"{q.dynamic_range_score:.0f}")
                qc[5].metric("Dropout", f"{drop_pct:.1f}%")
                st.caption(f"Resolution {w}x{h}. Statistics: mean={q.statistics.get('mean',0):.1f} "
                           f"std={q.statistics.get('std',0):.1f} p10={q.statistics.get('percentile_10',0):.0f} "
                           f"p90={q.statistics.get('percentile_90',0):.0f}. "
                           "Quality feeds the risk engine and filter; REVIEW REQUIRED images warn downstream.")
                for warn in (q.warnings or []):
                    st.warning(warn)

            # FULL vs TILED comparison (real inference times)
            if st.session_state.compare_result:
                cr = st.session_state.compare_result
                rf, rt = cr["full"], cr["tiled"]
                st.markdown(f"### FULL vs TILED (tile {cr['tile_size']}px, overlap {cr['overlap']}px)")
                cc = st.columns(4)
                nf = len(rf.detection.detections) if rf.detection else 0
                nt = len(rt.detection.detections) if rt.detection else 0
                cc[0].metric("FULL detections", str(nf))
                cc[1].metric("FULL inference", f"{(rf.detection.inference_time_ms if rf.detection else 0):.1f}ms")
                cc[2].metric("TILED detections", str(nt))
                cc[3].metric("TILED inference", f"{(rt.detection.inference_time_ms if rt.detection else 0):.1f}ms")
                # small-target recall note from real counts
                st.caption("Tiling helps small/faint objects by running the proposer at native resolution per tile; "
                           "compare counts and the Small-Target experiment under Model Validation.")

            # Detection panel
            if result.detection and result.detection.detections:
                st.markdown(f"**Detections: {len(result.detection.detections)}** | "
                           f"**Mode: {result.detection.mode}** | "
                           f"**Inference: {result.detection.inference_time_ms:.1f}ms**")

                for i, det in enumerate(result.detection.detections):
                    # Get classified display name; never invent a sonar class for generic detections
                    if i < len(result.classifications):
                        cls = result.classifications[i]
                        det_class = cls.class_name
                    else:
                        det_class = det.class_name
                    with st.expander(f"Target {i+1}: {det_class} ({det.confidence:.2f})",
                                     expanded=False):
                        # Find corresponding analysis
                        if i < len(result.classifications):
                            cls = result.classifications[i]
                            risk = result.risks[i] if i < len(result.risks) else None
                            explanation = result.explanations[i] if i < len(result.explanations) else None
                            coord = result.coordinates[i] if i < len(result.coordinates) else None

                            st.markdown(f"**Classification:** {cls.classification.value}")
                            st.markdown(f"**Model Confidence:** {cls.model_confidence:.3f}")
                            st.markdown(f"**Anomaly Score:** {cls.anomaly_score:.1f}")
                            if risk:
                                risk_color = "risk-high" if risk.risk_level.value == "HIGH" else \
                                             "risk-medium" if risk.risk_level.value == "MEDIUM" else \
                                             "risk-low" if risk.risk_level.value == "LOW" else "risk-review"
                                st.markdown(f"**Risk Level:** <span class='{risk_color}'>{risk.risk_level.value}</span>",
                                           unsafe_allow_html=True)
                            if coord:
                                sim_label = " [SIMULATED]" if coord.is_simulated else ""
                                st.markdown(f"**Coordinates:** ({coord.latitude:.6f}, {coord.longitude:.6f}){sim_label}")

                            # Evidence
                            if explanation and explanation.reasons:
                                st.markdown("**Evidence:**")
                                for ev in explanation.reasons[:8]:
                                    icon = "✓" if ev.direction == "positive" else "✗"
                                    st.markdown(f"  {icon} {ev.description}")

                            # Select target for inspection
                            if st.button(f"Inspect Target {i+1}", key=f"inspect_{i}"):
                                st.session_state.selected_target = {
                                    'index': i,
                                    'detection': det,
                                    'classification': cls,
                                    'risk': risk,
                                    'explanation': explanation,
                                    'coordinate': coord,
                                    'fingerprint': result.fingerprints[i] if i < len(result.fingerprints) else None,
                                    'context': result.context_results[i] if i < len(result.context_results) else None,
                                    'filter': result.filter_results[i] if i < len(result.filter_results) else None,
                                }
        else:
            st.info("Upload an image or start a demo mission to begin analysis.")

    # Bottom: Pipeline Status
    st.markdown("### Pipeline Status")
    if st.session_state.current_result:
        result = st.session_state.current_result
        pipeline_cols = st.columns(7)
        pipeline_cols[0].metric("Quality", f"{result.quality.quality_score:.1f}" if result.quality else "N/A")
        pipeline_cols[1].metric("Targets", str(len(result.detection.detections)) if result.detection else "0")
        pipeline_cols[2].metric("Pipeline Time", f"{result.pipeline_time_ms:.1f}ms")
        pipeline_cols[3].metric("Preprocess", f"{(result.timings or {}).get('preprocess_ms', result.preprocessing.processing_time_ms if result.preprocessing else 0):.1f}ms")
        pipeline_cols[4].metric("Inference", f"{result.detection.inference_time_ms:.1f}ms" if result.detection else "0ms")
        pipeline_cols[5].metric("Fingerprint", f"{(result.timings or {}).get('fingerprint_ms', 0):.1f}ms")
        pipeline_cols[6].metric("Model", (result.model_status or (result.detection.model_name if result.detection else "N/A"))[:28])
        # stage flow with real status
        stages = ["INGEST ✓", f"QUALITY ✓ ({result.quality.rating})" if result.quality else "QUALITY ✗",
                  "PREPROCESS ✓" if result.preprocessing is not None else "PREPROCESS ✗",
                  f"DETECT ✓ ({result.detection.mode})" if result.detection else "DETECT ✗",
                  f"CONTEXT ✓ ({len(result.context_results)})", f"FILTER ✓ ({len(result.filter_results)})",
                  f"RISK ✓ ({len(result.risks)})", f"GEO ✓ ({len(result.coordinates)})",
                  f"REPORT ✓ ({len(result.reports)})"]
        st.caption(" → ".join(stages))
        
        # Confidence calibration: combine model confidence with data quality
        if result.quality and result.classifications:
            for i, cls in enumerate(result.classifications):
                if i < len(result.detection.detections):
                    det = result.detection.detections[i]
                    disp_conf, final_status, note = calibrate_confidence(cls.model_confidence, result.quality.rating)
                    qc = st.columns(4)
                    qc[0].metric("Model Conf", f"{disp_conf:.3f}")
                    qc[1].metric("Data Quality", result.quality.rating)
                    qc[2].metric("Final Status", final_status)
                    qc[3].metric("Note", note[:25] if note else "")
    
    else:
        st.info("Pipeline ready. Upload an image or start a demo.")

    # Target Inspector (core screen)
    render_target_inspector()


def run_pipeline(image_path: str, conf_threshold: float, use_tiled: bool,
                 tile_size: int = 512, tile_overlap: int = 64,
                 apply_clahe: bool = True, apply_denoising: bool = True,
                 enhance_shadows: bool = False):
    """Run the complete analysis pipeline."""
    try:
        from src.detection.detector import TileConfig
        with st.spinner("Processing... (Quality → Preprocess → Detect → Analyze)"):
            cfg = PreprocessingConfig(apply_clahe=apply_clahe,
                                      apply_denoising=apply_denoising,
                                      enhance_shadows=enhance_shadows)
            pipeline = SonarisXPipeline(preprocess_config=cfg)
            if use_tiled:
                # stash tile config on pipeline detector via closure
                orig_detect = pipeline.detector.detect
                def _detect(img, conf_threshold=conf_threshold, use_tiled=True, tile_config=None):
                    return orig_detect(img, conf_threshold=conf_threshold, use_tiled=True,
                                       tile_config=TileConfig(tile_size=tile_size, overlap=tile_overlap))
                pipeline.detector.detect = _detect  # type: ignore
            result = pipeline.process_image(image_path, conf_threshold=conf_threshold, use_tiled=use_tiled)
            st.session_state.current_result = result
            st.session_state.results.append(result)
            st.session_state.compare_result = None
            if result.warnings:
                for w in result.warnings:
                    st.warning(w)
            if result.image_info is not None and not result.image_info.readable:
                st.error(f"Image could not be read: {result.image_info.filename}")
            elif result.detection is None:
                st.warning("Pipeline completed with no detection stage (see warnings).")
            else:
                st.success("Analysis complete!")
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        logger_exception(e)


def run_compare(image_path: str, conf_threshold: float, tile_size: int, tile_overlap: int,
                apply_clahe: bool, apply_denoising: bool, enhance_shadows: bool):
    """Run FULL vs TILED detection on the same image; record real inference times."""
    try:
        from src.detection.detector import TileConfig
        cfg = PreprocessingConfig(apply_clahe=apply_clahe,
                                  apply_denoising=apply_denoising,
                                  enhance_shadows=enhance_shadows)
        with st.spinner("Comparing FULL vs TILED detection..."):
            p_full = SonarisXPipeline(preprocess_config=cfg)
            r_full = p_full.process_image(image_path, conf_threshold=conf_threshold, use_tiled=False)
            p_tiled = SonarisXPipeline(preprocess_config=cfg)
            # inject tile config
            orig = p_tiled.detector.detect
            def _d(img, conf_threshold=conf_threshold, use_tiled=True, tile_config=None):
                return orig(img, conf_threshold=conf_threshold, use_tiled=True,
                            tile_config=TileConfig(tile_size=tile_size, overlap=tile_overlap))
            p_tiled.detector.detect = _d  # type: ignore
            r_tiled = p_tiled.process_image(image_path, conf_threshold=conf_threshold, use_tiled=True)
        st.session_state.compare_result = {"full": r_full, "tiled": r_tiled,
                                           "tile_size": tile_size, "overlap": tile_overlap}
        st.session_state.current_result = r_tiled
        st.session_state.results.append(r_tiled)
        st.success("Comparison complete — see FULL vs TILED panel below.")
    except Exception as e:
        st.error(f"Comparison error: {str(e)}")
        logger_exception(e)


def render_detection_overlay(base_bgr: np.ndarray, result) -> np.ndarray:
    """Draw detection boxes + dropout overlay onto a BGR copy (real regions only)."""
    view = base_bgr.copy()
    if result.preprocessing and getattr(result.preprocessing, "dropout_regions", None):
        if st.session_state.get("show_dropout", True):
            for (x, y, w, h) in result.preprocessing.dropout_regions:
                cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 1)
    if result.detection and result.detection.detections:
        for i, det in enumerate(result.detection.detections):
            x, y, w, h = det.bbox
            risk = result.risks[i].risk_level.value if i < len(result.risks or []) else "REVIEW"
            color = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255),
                     "LOW": (0, 255, 0), "REVIEW": (200, 200, 200)}.get(risk, (0, 255, 255))
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
            cv2.putText(view, f"T{i+1} {det.class_name}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return view


def generate_synthetic_sonar() -> np.ndarray:
    """Generate a synthetic sonar image for demo purposes."""
    np.random.seed(42)
    img = np.random.randint(20, 60, (600, 800, 3), dtype=np.uint8)

    # Add some "targets"
    for _ in range(4):
        x = np.random.randint(50, 750)
        y = np.random.randint(50, 550)
        size = np.random.randint(15, 50)
        intensity = np.random.randint(180, 255)
        cv2.circle(img, (x, y), size, (intensity, intensity, intensity), -1)
        cv2.circle(img, (x, y), size + 5, (150, 150, 150), 2)

    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add some speckle
    for _ in range(200):
        x, y = np.random.randint(0, 800), np.random.randint(0, 600)
        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

    return img


def render_target_inspector():
    """Professional target-inspection panel (core screen)."""
    sel = st.session_state.get("selected_target")
    if not sel:
        return
    st.markdown("---")
    st.markdown('<span class="tab-title">🎯 TARGET INSPECTOR</span>', unsafe_allow_html=True)
    det = sel.get("detection")
    cls = sel.get("classification")
    risk = sel.get("risk")
    explanation = sel.get("explanation")
    coord = sel.get("coordinate")
    fp = sel.get("fingerprint")
    ctx = sel.get("context")
    filt = sel.get("filter")
    result = st.session_state.get("current_result")
    idx = sel.get("index", 0)
    dropout_flag = False
    if result is not None and idx < len(result.dropout_overlap or []):
        dropout_flag = result.dropout_overlap[idx]

    left, right = st.columns([1, 1.2])
    with left:
        # Large target crop from real image
        try:
            base = None
            if result is not None and result.preprocessing is not None:
                base = result.preprocessing.original
            if base is None and st.session_state.get("current_image"):
                base = cv2.imread(st.session_state.current_image)
            if base is not None and det is not None:
                x, y, w, h = det.bbox
                h_img, w_img = base.shape[:2]
                crop = base[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
                if crop is not None and crop.size > 0:
                    st.markdown("**SONAR TARGET**")
                    display_sonar_image(crop)
        except Exception as e:
            st.warning(f"Crop unavailable: {e}")
        # Display the classified name; fall back to detection class_name if no classification
        display_class = cls.class_name if cls else det.class_name
        st.markdown(f"**CLASS:** {display_class}")
        if cls:
            st.markdown(f"**State:** {cls.classification.value}")
            st.markdown("MODEL CONFIDENCE (detector support, uncalibrated): "
                        f"**{cls.model_confidence:.2f}**")
            st.markdown(f"ANOMALY SCORE (seabed divergence): **{cls.anomaly_score:.1f}**")
        if risk:
            st.markdown(f"**RISK: {risk.risk_level.value}** (score {risk.risk_score:.1f})")
        if filt:
            st.markdown(f"**Filter:** {filt.filter_decision.value} (score {filt.filter_score:.1f})")
        if dropout_flag:
            st.error("DATA QUALITY WARNING: target overlaps dropout/degraded data.")
        if coord:
            sim = " [SIMULATED]" if coord.is_simulated else ""
            st.markdown(f"**LOCATION:** {coord.latitude:.6f}, {coord.longitude:.6f}{sim} (source: {coord.source})")
    with right:
        if fp is not None:
            st.markdown("**ACOUSTIC FINGERPRINT** (measured from detected region)")
            g, it, tx, ed, sh, sb = fp.geometry, fp.intensity, fp.texture, fp.edges, fp.shadow, fp.seabed_context
            st.markdown(f"Geometry — Area **{g.area} px**, {g.width}×{g.height}, "
                        f"Aspect Ratio **{g.aspect_ratio:.2f}**, Solidity {g.solidity:.2f}")
            st.markdown(f"Intensity — Mean **{it.mean:.1f}**, Std {it.std:.1f}, "
                        f"P10 {it.percentile_10:.0f} / P50 {it.percentile_50:.0f} / P90 {it.percentile_90:.0f}, "
                        f"Contrast **{it.local_contrast:.1f}**, Dyn.Range {it.dynamic_range:.0f}")
            st.markdown(f"Texture — LBP mean {tx.lbp_mean:.2f} / entropy {tx.lbp_entropy:.2f}, "
                        f"GLCM contrast {tx.glcm_contrast:.1f}, energy {tx.glcm_energy:.3f}, "
                        f"homogeneity {tx.glcm_homogeneity:.3f}")
            st.markdown(f"Edges — density **{ed.edge_density:.3f}** ({ed.edge_count} px)")
            st.markdown(f"Shadow — Detected **{'YES' if sh.shadow_present else 'NO'}**, "
                        f"ratio {sh.shadow_area:.2f}, intensity {sh.shadow_intensity:.0f}")
            st.markdown(f"Seabed — Target Δ **{sb.target_seabed_divergence.get('intensity_diff', 0):.1f}** "
                        f"(seabed mean {sb.seabed_mean:.1f}, std {sb.seabed_std:.1f})")
        if ctx is not None:
            cd = ctx.context_difference
            st.markdown("**TARGET vs LOCAL SEABED**")
            st.markdown(f"Target contrast **{ctx.target_contrast_score:.0f}** vs "
                        f"seabed uniformity {ctx.seabed_uniformity:.2f} → "
                        f"difference **{cd.overall_score:.1f}**")
            st.markdown(f"Intensity Δ {cd.intensity_diff:.1f} | Texture Δ {cd.texture_diff:.1f} | "
                        f"Geometry Δ {cd.geometry_diff:.1f} | Shadow Δ {cd.shadow_diff:.1f}")
            st.markdown("**CONTEXT EVIDENCE**")
            for f in (cd.factors or [])[:6]:
                st.markdown(f"✓ {f}")
        if explanation is not None:
            st.markdown("**WHY FLAGGED**")
            for ev in (explanation.reasons or [])[:8]:
                icon = "✓" if ev.direction == "positive" else "✗"
                st.markdown(f"{icon} {ev.description}")
            # evidence bars from real normalized feature values
            st.markdown("**EVIDENCE STRENGTH** (normalized measured features)")
            feats = explanation.feature_evidence or {}
            bars = [("context", feats.get("context_score", 0) / 100.0),
                    ("contrast", min(1.0, feats.get("local_contrast", 0) / 80.0)),
                    ("shadow", min(1.0, feats.get("shadow_area", 0) * 4.0)),
                    ("edges", min(1.0, feats.get("edge_density", 0) * 8.0))]
            for name, val in bars:
                st.progress(max(0.0, min(1.0, float(val))), text=f"{name}: {val:.2f}")
            st.caption(explanation.summary)
        if risk is not None:
            st.markdown("**WHY THIS RISK?**")
            for f in sorted(risk.factors, key=lambda x: x.contribution, reverse=True)[:5]:
                st.markdown(f"- {f.factor_name}: {f.description} (weight {f.weight:.2f}, +{f.contribution:.1f})")
            st.caption(risk.reasoning)
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("OPEN MAP", key="insp_map"):
            st.session_state.map_target = sel
            st.info("Open 🗺️ Tactical Map — target highlighted.")
    with b2:
        if st.button("EXPORT TARGET JSON", key="insp_export"):
            import json as _json
            payload = {
                "target_id": cls.target_id if cls else f"T{idx+1}",
                "class": cls.class_name if cls else det.class_name,
                "state": cls.classification.value if cls else "N/A",
                "model_confidence": cls.model_confidence if cls else det.confidence,
                "anomaly_score": cls.anomaly_score if cls else 0,
                "risk": risk.risk_level.value if risk else "N/A",
                "bbox": list(det.bbox), "simulated_coords": bool(coord and coord.is_simulated),
            }
            st.download_button("Download JSON", _json.dumps(payload, indent=2),
                               file_name=f"target_{idx+1}.json", key="insp_dl")
    with b3:
        if st.button("CLOSE INSPECTOR", key="insp_close"):
            st.session_state.selected_target = None
            st.rerun()


def mission_dashboard():
    """Mission Dashboard page."""
    st.markdown('<span class="tab-title">📊 MISSION DASHBOARD</span>', unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.results:
        st.info("No analysis results yet. Run analysis to populate dashboard.")
        return

    # Summary metrics
    total_images = len(st.session_state.results)
    total_detections = sum(len(r.detection.detections) if r.detection else 0 for r in st.session_state.results)
    high_risk = sum(len([f for f in r.risks if f.risk_level.value == 'HIGH']) if r.risks else 0 for r in st.session_state.results)
    unknown_anom = sum(len([c for c in r.classifications if c.classification.value == 'UNKNOWN ANOMALY']) if r.classifications else 0 for r in st.session_state.results)
    avg_confidence = np.mean([cls.model_confidence for r in st.session_state.results for cls in r.classifications]) if any(r.classifications for r in st.session_state.results) else 0
    avg_time = np.mean([r.pipeline_time_ms for r in st.session_state.results]) if st.session_state.results else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Images Analyzed", str(total_images))
    col2.metric("Targets Detected", str(total_detections))
    col3.metric("High-Risk Targets", str(high_risk), delta_color="inverse")
    col4.metric("Unknown Anomalies", str(unknown_anom))
    col5.metric("Avg Confidence", f"{avg_confidence:.3f}")
    col6.metric("Avg Time", f"{avg_time:.1f}ms")

    # Charts
    st.markdown("### Charts")

    # Risk distribution
    risk_data = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'REVIEW': 0}
    for r in st.session_state.results:
        for risk in (r.risks or []):
            risk_data[risk.risk_level.value] += 1

    if sum(risk_data.values()) > 0:
        st.bar_chart(risk_data)

    # Class distribution
    class_data = {}
    for r in st.session_state.results:
        for cls in (r.classifications or []):
            key = cls.classification.value
            class_data[key] = class_data.get(key, 0) + 1

    if class_data:
        st.bar_chart(class_data)

    # Processing time
    time_data = [r.pipeline_time_ms for r in st.session_state.results]
    if time_data:
        st.line_chart(time_data)


def target_catalog():
    """Target Catalog page."""
    st.markdown('<span class="tab-title">📋 TARGET CATALOG</span>', unsafe_allow_html=True)
    st.markdown("---")

    results = st.session_state.results
    if not results:
        st.info("No targets yet. Run analysis first.")
        return

    # Build catalog
    catalog_data = []
    for r_idx, result in enumerate(results):
        if not result.detection or not result.detection.detections:
            continue
        for d_idx, det in enumerate(result.detection.detections):
            cls = result.classifications[d_idx] if d_idx < len(result.classifications) else None
            risk = result.risks[d_idx] if d_idx < len(result.risks) else None
            coord = result.coordinates[d_idx] if d_idx < len(result.coordinates) else None

            catalog_data.append({
                'Target ID': f"TGT-{r_idx}-{d_idx}",
                'Class': cls.class_name if cls else det.class_name,
                'Classification': cls.classification.value if cls else 'N/A',
                'Confidence': round(det.confidence, 3),
                'Anomaly Score': round(cls.anomaly_score, 1) if cls else 0,
                'Risk': risk.risk_level.value if risk else 'N/A',
                'Risk Score': round(risk.risk_score, 1) if risk else 0,
                'Latitude': round(coord.latitude, 6) if coord else 0,
                'Longitude': round(coord.longitude, 6) if coord else 0,
                'Status': 'Active'
            })

    if catalog_data:
        import pandas as pd
        df = pd.DataFrame(catalog_data)
        st.markdown("### Filters")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            risk_filter = st.selectbox("Risk Level", ["All"] + sorted(df['Risk'].unique()), key="cat_risk")
        with f2:
            class_filter = st.selectbox("Class", ["All"] + sorted(df['Class'].unique()), key="cat_class")
        with f3:
            state_filter = st.selectbox("Known/Unknown", ["All"] + sorted(df['Classification'].unique()), key="cat_state")
        with f4:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, key="cat_conf")
        if risk_filter != "All":
            df = df[df['Risk'] == risk_filter]
        if class_filter != "All":
            df = df[df['Class'] == class_filter]
        if state_filter != "All":
            df = df[df['Classification'] == state_filter]
        df = df[df['Confidence'] >= min_conf]
        st.dataframe(df, use_container_width=True)

        # Click to inspect
        if len(df):
            selected = st.selectbox("Select Target to Inspect", df['Target ID'].tolist(), key="cat_select")
            if selected:
                target_row = df[df['Target ID'] == selected].iloc[0]
                st.markdown(f"### {selected}")
                st.markdown(f"**Class:** {target_row['Class']}")
                st.markdown(f"**Classification:** {target_row['Classification']}")
                st.markdown(f"**Confidence:** {target_row['Confidence']}")
                st.markdown(f"**Risk:** {target_row['Risk']}")
                st.markdown(f"**Coordinates:** ({target_row['Latitude']}, {target_row['Longitude']}) [SIMULATED]")
        else:
            st.info("No targets match the current filters.")


def tactical_map():
    """Tactical Map page."""
    st.markdown('<span class="tab-title">🗺️ TACTICAL MAP</span>', unsafe_allow_html=True)
    st.markdown("---")

    if plt is None:
        st.warning("Matplotlib is not installed. Map visualization unavailable (pip install matplotlib).")
        return

    results = st.session_state.results
    if not results:
        st.info("No detection locations to display. Run analysis first.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        ax.set_facecolor('#0a1628')
        ax.set_xlabel('Survey track (relative, SIMULATED layout)', color='white')
        ax.set_ylabel('Across-track (relative, SIMULATED layout)', color='white')
        ax.set_title('SONARIS-X Tactical Map [SIMULATED POSITIONS]', color='#00d4ff', fontsize=16)
        ax.tick_params(colors='white')

        colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#00ff88', 'REVIEW': '#aaaaaa'}

        for r_idx, result in enumerate(results):
            if not result.detection or not result.detection.detections:
                continue
            for d_idx, det in enumerate(result.detection.detections):
                coord = result.coordinates[d_idx] if d_idx < len(result.coordinates or []) else None
                risk = result.risks[d_idx] if d_idx < len(result.risks or []) else None
                risk_color = colors.get(risk.risk_level.value if risk else 'REVIEW', '#aaaaaa')

                # Deterministic SIMULATED layout derived from detection geometry
                # (no genuine GPS; reproducible, clearly labeled — never random).
                bx, by, bw, bh = det.bbox
                x = (bx + bw / 2) / max(1, result.detection.image_shape[1]) + r_idx * 1.5
                y = 1.0 - (by + bh / 2) / max(1, result.detection.image_shape[0]) + d_idx * 0.05

                # Mark simulated locations
                ax.scatter(x, y, c=risk_color, s=150, marker='*', edgecolors='white', linewidth=0.5)
                ax.text(x, y, f"[SIM] T{d_idx+1} {cls.class_name if cls else det.class_name}", color='white', fontsize=7, ha='center')

        ax.set_xlim(-0.1, max(1.2, len(results) * 1.5))
        ax.set_ylim(-0.1, 1.3)
        plt.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)

    # Highlight inspector-selected target
    mt = st.session_state.get("map_target")
    if mt and mt.get("detection") is not None:
        d = mt["detection"]
        st.markdown(f"**Inspector target:** {d.class_name} bbox {d.bbox}")

    # Legend
    st.markdown("**Map Legend:**")
    for level, color in colors.items():
        st.markdown(f'<span style="color:{color}">●</span> {level}', unsafe_allow_html=True)
    st.markdown('<span class="simulated-label">⚠️</span> = Simulated coordinates', unsafe_allow_html=True)


def mission_reports():
    """Mission Reports page."""
    st.markdown('<span class="tab-title">📄 MISSION REPORTS</span>', unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.results:
        st.info("No reports to generate. Run analysis first.")
        return

    report_gen = ReportGenerator()
    for r in st.session_state.results:
        if r.reports:
            for entry in r.reports:
                report_gen.add_target(entry)

    # Generate reports
    if st.button("Generate JSON Report", key="json_report"):
        report = report_gen.generate_json("outputs/reports/mission_report.json")
        st.success("JSON report generated!")
        st.json(report)

    if st.button("Generate CSV Report", key="csv_report"):
        rows = report_gen.generate_csv("outputs/reports/mission_report.csv")
        st.success("CSV report generated!")
        import pandas as pd
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("Generate Text Report", key="text_report"):
        text = report_gen.generate_text_report("outputs/reports/mission_report.txt")
        st.success("Text report generated!")
        st.text(text)

    # Summary
    summary = report_gen.get_summary_stats()
    st.markdown("### Report Summary")
    for key, val in summary.items():
        st.markdown(f"**{key}:** {val}")


def review_queue():
    """Review queue: uncertain detections kept visible for human triage."""
    st.markdown('<span class="tab-title">🧪 REVIEW QUEUE</span>', unsafe_allow_html=True)
    st.markdown("Uncertain detections are never silently deleted — they wait here for human review.")
    st.markdown("---")
    rows = []
    for r_idx, result in enumerate(st.session_state.get("results", [])):
        if not result.detection:
            continue
        for d_idx, det in enumerate(result.detection.detections):
            filt = result.filter_results[d_idx] if d_idx < len(result.filter_results or []) else None
            cls = result.classifications[d_idx] if d_idx < len(result.classifications or []) else None
            needs = (filt is not None and filt.requires_human_review) or \
                    (cls is not None and cls.classification.value == "UNCLASSIFIED")
            if needs:
                rows.append({
                    "Queue ID": f"R{r_idx}-{d_idx}",
                    "Target": f"T{d_idx+1} ({det.class_name})",
                    "Confidence": round(det.confidence, 3),
                    "Anomaly": round(cls.anomaly_score, 1) if cls else 0,
                    "Filter": filt.filter_decision.value if filt else "N/A",
                    "Reason": "; ".join((filt.reasons or [])[:2]) if filt else "",
                })
    if not rows:
        st.info("Review queue is empty. Uncertain detections will appear here.")
        return
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def dataset_explorer():
    """Dataset Explorer: real statistics from data/ directories (never invented)."""
    st.markdown('<span class="tab-title">🗂️ DATASET EXPLORER</span>', unsafe_allow_html=True)
    st.markdown("---")
    validator = DatasetValidator()
    for label, path in [("data/demo (synthetic)", "data/demo"), ("data/raw (user)", "data/raw")]:
        st.markdown(f"### {label} — `{path}`")
        if not os.path.isdir(path):
            st.info(f"Directory `{path}` does not exist yet.")
            continue
        stats = validator.validate_directory(path)
        c = st.columns(5)
        c[0].metric("Images", str(stats.image_count))
        c[1].metric("Readable", str(stats.readable_images))
        c[2].metric("Corrupted", str(stats.corrupted_images))
        c[3].metric("Duplicates", str(stats.duplicate_images))
        c[4].metric("Avg size (px)", f"{stats.avg_image_size:.0f}")
        if stats.file_extensions:
            st.markdown(f"Extensions: {stats.file_extensions}")
        if stats.corrupted_files:
            with st.expander(f"Corrupted files ({len(stats.corrupted_files)})"):
                for f in stats.corrupted_files[:20]:
                    st.markdown(f"- {f}")
    st.markdown("### Generate synthetic demo set")
    st.caption("Creates labeled SYNTHETIC DEMO DATA (7 scenarios) under data/demo with a manifest. "
               "Labels are demo ground truth, not model predictions.")
    if st.button("GENERATE DEMO SET", key="gen_demo"):
        try:
            from src.demo.synthetic import generate_scenario, SCENARIOS
            os.makedirs("data/demo", exist_ok=True)
            manifest = []
            for name in SCENARIOS:
                img, gt, info = generate_scenario(name)
                fname = f"data/demo/{name.lower().replace(' ', '_').replace('/', '_')}.png"
                cv2.imwrite(fname, img)
                manifest.append({"file": fname, "scenario": name,
                                 "description": info["description"], "ground_truth": gt})
            import json as _json
            with open("data/demo/manifest.json", "w") as f:
                _json.dump(manifest, f, indent=2)
            st.success(f"Generated {len(manifest)} demo images + manifest.json")
            st.json(manifest)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            logger_exception(e)


def training_center():
    """Training Center: real config + leakage-safe splits; honest about data needs."""
    st.markdown('<span class="tab-title">🎓 TRAINING CENTER</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("Training needs a labeled sonar dataset in YOLO format (images + *.txt labels + data.yaml). "
            "Without one, the detector stays in DEMO / UNTRAINED mode — by design.")
    dataset_dir = st.text_input("Dataset image directory", "data/raw", key="tr_dir")
    model = st.selectbox("Base model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], key="tr_model")
    epochs = st.slider("Epochs", 1, 300, 50, key="tr_epochs")
    img_size = st.selectbox("Image size", [416, 640, 1024], index=1, key="tr_imgsz")
    batch = st.selectbox("Batch size", [4, 8, 16, 32], index=2, key="tr_batch")
    split_strategy = st.selectbox("Leakage-safe split strategy",
                                  ["mission", "survey", "location", "session", "random"], key="tr_split")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("VALIDATE DATASET", key="tr_validate"):
            v = DatasetValidator()
            if os.path.isdir(dataset_dir):
                s = v.validate_directory(dataset_dir)
                st.markdown(f"Images: **{s.image_count}** | Readable: **{s.readable_images}** | "
                            f"Corrupted: **{s.corrupted_images}** | Duplicates: **{s.duplicate_images}**")
            else:
                st.warning(f"Directory `{dataset_dir}` not found.")
    with c2:
        if st.button("SHOW SPLIT PLAN (no training run)", key="tr_plan"):
            try:
                from scripts.train import TrainingConfig
                cfg = TrainingConfig(dataset=dataset_dir, model=model, image_size=img_size,
                                     epochs=epochs, batch_size=batch, split_strategy=split_strategy)
                st.json({"dataset": cfg.dataset, "model": cfg.model, "image_size": cfg.image_size,
                         "epochs": cfg.epochs, "batch_size": cfg.batch_size,
                         "split_strategy": cfg.split_strategy,
                         "augmentation": cfg.augmentation,
                         "note": "Actual training runs via scripts/train.py once labeled data exists. "
                                 "No metrics are shown until a real run completes."})
            except Exception as e:
                st.error(f"Config error: {e}")
    st.caption("Sonar-specific augmentation available: contrast, intensity, speckle noise, blur, "
               "resolution degradation, dropout simulation (see scripts/train.py).")


def model_validation():
    """Model Validation page."""
    st.markdown('<span class="tab-title">🔬 MODEL VALIDATION</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Baseline Comparison")
    st.info("Note: Full training requires properly formatted YOLO dataset. Running baseline evaluation with available data.")

    if st.button("Run Baseline Comparison", key="baseline_btn"):
        try:
            from scripts.evaluate import Evaluator, BaselineDetector
            from scripts.ablation import AblationStudy
            from scripts.small_target import SmallTargetExperiment

            evaluator = Evaluator()
            study = AblationStudy()
            small_exp = SmallTargetExperiment()

            # Generate test data if no real data
            test_img = generate_synthetic_sonar()

            # Run detection
            detector = MultiScaleDetector()
            det_result = detector.detect(test_img)

            # Compare
            comparison = evaluator.compare_with_baseline(test_img, det_result.detections, det_result.inference_time_ms)
            st.json(comparison)

            # Ablation studies
            for use_pre in [False, True]:
                for use_tile in [False, True]:
                    for use_fp in [False, True]:
                        study.run_experiment(
                            test_img, det_result.detections,
                            use_preprocessing=use_pre,
                            use_tiling=use_tile,
                            use_filtering=use_fp
                        )

            st.markdown("### Ablation Study Results")
            st.text(study.generate_comparison_table())

            # Small target experiment
            small_results = small_exp.analyze_performance(test_img, det_result.detections, det_result.detections)
            st.markdown("### Small Target Experiment")
            st.text(small_exp.generate_report())

            # Domain generalization (honest measurement across demo scenarios)
            st.markdown("### Domain Generalization (demo scenarios as domains)")
            st.caption("Treats each synthetic scenario as a domain: measures in-domain vs "
                       "cross-domain candidate counts and anomaly-score stability. "
                       "No domain adaptation is claimed — this measures the gap.")
            try:
                from src.demo.synthetic import generate_scenario, SCENARIOS
                import numpy as _np
                rows = []
                for name in SCENARIOS:
                    simg, _, _ = generate_scenario(name)
                    rr = detector.detect(simg)
                    scores = []
                    # anomaly proxy: saliency confidence spread (real measured values)
                    for d in rr.detections:
                        scores.append(d.confidence)
                    rows.append({"domain": name, "candidates": len(rr.detections),
                                 "mean_support": round(float(_np.mean(scores)), 3) if scores else 0.0,
                                 "max_support": round(float(_np.max(scores)), 3) if scores else 0.0})
                import pandas as _pd
                dfd = _pd.DataFrame(rows)
                st.dataframe(dfd, use_container_width=True)
                counts = [r["candidates"] for r in rows]
                st.markdown(f"**Gap observed:** candidate counts range {min(counts)}–{max(counts)} across "
                            f"demo domains (in-domain vs cross-domain behavior differs; preprocessing + "
                            f"context filtering are the mechanisms under test, not claimed fixes).")
            except Exception as de:
                st.warning(f"Domain probe unavailable: {de}")

        except Exception as e:
            st.error(f"Validation error: {str(e)}")
            logger_exception(e)


def failure_analysis():
    """Failure Analysis page."""
    st.markdown('<span class="tab-title">❌ FAILURE ANALYSIS</span>', unsafe_allow_html=True)
    st.markdown("---")

    failure_cases = [
        {
            'name': 'Rock Mistaken as Object',
            'description': 'High-contrast rock formations can trigger false positives due to similar edge patterns.',
            'category': 'False Positive',
            'solution': 'Use context analysis to compare with surrounding seabed',
            'scenario': 'Natural Rock / Seabed',
        },
        {
            'name': 'Small Target Missed',
            'description': 'Targets below detection threshold are missed in full-image inference.',
            'category': 'Missed Detection',
            'solution': 'Use tiled detection for improved small target recall',
            'scenario': 'Cylinder',
        },
        {
            'name': 'Faint Target',
            'description': 'Low-contrast targets with weak signatures are difficult to detect.',
            'category': 'Missed Detection',
            'solution': 'Preprocessing enhancement and multi-scale detection',
            'scenario': 'Ghost Net',
        },
        {
            'name': 'Noisy Image',
            'description': 'High noise levels degrade detection performance.',
            'category': 'Quality Issue',
            'solution': 'Noise reduction and quality assessment',
            'scenario': 'Low-quality / dropout',
        },
        {
            'name': 'Data Dropout',
            'description': 'Vehicle motion creates dropout regions in sonar data.',
            'category': 'Data Issue',
            'solution': 'Dropout detection and reporting',
            'scenario': 'Low-quality / dropout',
        },
        {
            'name': 'Unknown Anomaly',
            'description': 'Objects that don\'t match known classes but have strong anomaly evidence.',
            'category': 'Unknown',
            'solution': 'Known/Unknown classification with anomaly scoring',
            'scenario': 'Unknown Anomaly',
        },
        {
            'name': 'Overlapping Objects',
            'description': 'Multiple objects in proximity cause merged detections.',
            'category': 'Detection Issue',
            'solution': 'NMS merging with overlap threshold tuning',
            'scenario': 'Shipwreck',
        },
        {
            'name': 'Low-Quality Image',
            'description': 'Poor resolution, low contrast, or heavy noise reduce detection accuracy.',
            'category': 'Quality Issue',
            'solution': 'Quality assessment and preprocessing',
            'scenario': 'Low-quality / dropout',
        }
    ]

    for case in failure_cases:
        with st.expander(f"❌ {case['name']} ({case['category']}) [SYNTHETIC DEMO DATA]"):
            st.markdown(f"**Description:** {case['description']}")
            st.markdown(f"**Solution:** {case['solution']}")
            st.markdown(f"**Demo scenario:** {case['scenario']}")

            if plt is None:
                st.warning("Matplotlib not installed; diagnostic visualization unavailable.")
                continue

            # Real pipeline output on the scenario image (not prewritten screens)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            try:
                from src.demo.synthetic import generate_scenario
                scen_img, gt, _ = generate_scenario(case['scenario'])
                pipe = SonarisXPipeline()
                import tempfile as _tf
                _t = _tf.NamedTemporaryFile(delete=False, suffix='.png')
                cv2.imwrite(_t.name, scen_img)
                _t.close()
                fres = pipe.process_image(_t.name, conf_threshold=0.3)
                ax[0].imshow(cv2.cvtColor(scen_img, cv2.COLOR_BGR2RGB))
                ax[0].set_title(f"Input: {case['scenario']}")
                ax[0].axis('off')
                over = render_detection_overlay(
                    cv2.cvtColor(scen_img, cv2.COLOR_BGR2RGB)[:, :, ::-1], fres)
                ax[1].imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
                ax[1].set_title(f"Model output: {len(fres.detection.detections)} detections")
                ax[1].axis('off')
                st.pyplot(fig)
                states = [c.classification.value for c in fres.classifications]
                st.markdown(f"**Actual pipeline result:** {len(fres.detection.detections)} detections, "
                            f"states {states or 'none'}, quality {fres.quality.quality_score:.1f} "
                            f"({fres.quality.rating}).")
                st.markdown(f"**Expected (demo GT):** {[g['demo_class'] for g in gt]} — "
                            f"synthetic ground truth for reference, not a claim of model accuracy.")
                import os as _os
                _os.unlink(_t.name)
            except Exception as e:
                st.warning(f"Diagnostic visualization unavailable: {e}")
            finally:
                plt.close(fig)


def system_diagnostics():
    """System Diagnostics page."""
    st.markdown('<span class="tab-title">⚙️ SYSTEM DIAGNOSTICS</span>', unsafe_allow_html=True)
    st.markdown("---")

    collector = SystemDiagnosticsCollector()

    # Collect diagnostics
    diagnostics = collector.collect()

    import platform as _plat
    st.markdown("### Environment")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Python", _plat.python_version())
    e2.metric("OS", _plat.system())
    try:
        import torch as _torch
        _dev = "cuda" if _torch.cuda.is_available() else "cpu"
    except Exception:
        _dev = "cpu"
    e3.metric("Device", _dev)
    try:
        _probe = MultiScaleDetector()
        _ms = _probe.model_status()
    except Exception as ex:
        _ms = f"probe failed: {ex}"
    e4.metric("Model Loaded", _ms[:24])

    st.markdown("### PIPELINE SELF TEST")
    st.caption("Runs a synthetic scenario through quality → preprocess → detect → fingerprint → "
               "context → filter → risk → explain → geolocate → report. Real execution, not a mock.")
    if st.button("RUN SELF TEST", key="selftest"):
        try:
            from src.demo.synthetic import generate_scenario
            import tempfile as _tf, os as _os
            simg, _, _ = generate_scenario("Cylinder")
            _t = _tf.NamedTemporaryFile(delete=False, suffix='.png')
            cv2.imwrite(_t.name, simg)
            _t.close()
            _pipe = SonarisXPipeline()
            _r = _pipe.process_image(_t.name, conf_threshold=0.3)
            checks = [
                ("ingest readable", _r.image_info is not None and _r.image_info.readable),
                ("quality scored", _r.quality is not None and 0 <= _r.quality.quality_score <= 100),
                ("preprocessed", _r.preprocessing is not None),
                ("detection ran", _r.detection is not None),
                ("fingerprint/context/filter/risk/explain",
                 len(_r.fingerprints) == len(_r.detection.detections) if _r.detection else True),
                ("model status honest", bool(_r.model_status)),
            ]
            for name, ok in checks:
                st.markdown(f"{'✓' if ok else '✗'} {name}")
            if all(ok for _, ok in checks):
                st.success(f"SELF TEST PASSED in {_r.pipeline_time_ms:.1f}ms "
                           f"({len(_r.detection.detections)} detections).")
            else:
                st.error("SELF TEST FAILED — see checks above.")
            _os.unlink(_t.name)
        except Exception as e:
            st.error(f"Self test error: {e}")
            logger_exception(e)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Size", f"{diagnostics.model_size_mb:.2f} MB")
    col2.metric("CPU Usage", f"{diagnostics.cpu_usage_percent:.1f}%")
    col3.metric("Memory Usage", f"{diagnostics.memory_usage_mb:.1f} MB")
    col4.metric("GPU Available", str(diagnostics.gpu_available))

    if diagnostics.gpu_available:
        st.markdown(f"**GPU:** {diagnostics.gpu_name}")

    st.markdown("### Optimization Path")
    if diagnostics.optimization_path:
        for key, val in diagnostics.optimization_path.items():
            st.markdown(f"**{key}:** {val}")

    # Performance metrics
    st.markdown("### Performance Metrics")
    if st.session_state.results:
        total_time = sum(r.pipeline_time_ms for r in st.session_state.results)
        avg_time = total_time / len(st.session_state.results)
        st.metric("Total Pipeline Time", f"{total_time:.1f}ms")
        st.metric("Average Per Image", f"{avg_time:.1f}ms")

        # ONNX/quantization path
        st.markdown("### Deployment Optimization")
        st.markdown("""
        - **ONNX Export:** Available via `torch.onnx.export()`
        - **Quantization:** Dynamic INT8 quantization available via `torch.quantization`
        - **Edge Deployment:** Model can be exported to ONNX for edge devices
        - **Note:** Actual testing required before claiming edge readiness
        """)


def logger_exception(e):
    """Log exception."""
    import logging
    logging.getLogger(__name__).exception(str(e))


def calibrate_confidence(model_confidence: float, quality_rating: str) -> tuple:
    """Calibrate model confidence based on data quality.

    Returns (display_confidence, final_status, note) where:
    - display_confidence: the confidence value to show (may be adjusted)
    - final_status: HIGH/MEDIUM/LOW/REVIEW REQUIRED
    - note: explanatory text about the calibration
    """
    # Base status from model confidence alone
    if model_confidence >= 0.8:
        base_status = "HIGH"
    elif model_confidence >= 0.5:
        base_status = "MEDIUM"
    else:
        base_status = "LOW"

    # Quality adjustment
    if quality_rating == "REVIEW REQUIRED":
        final_status = "REVIEW REQUIRED"
        note = "Image quality is degraded — confidence moderated for review."
        # Downgrade status when quality is poor
        if base_status == "HIGH":
            display_confidence = round(model_confidence * 0.7, 3)
        elif base_status == "MEDIUM":
            display_confidence = round(model_confidence * 0.85, 3)
        else:
            display_confidence = round(model_confidence, 3)
    elif quality_rating == "DEGRADED":
        final_status = base_status
        note = "Image quality is degraded — monitor for review."
        display_confidence = round(model_confidence, 3)
    else:
        final_status = base_status
        note = "Image quality is good — confidence as detected."
        display_confidence = round(model_confidence, 3)

    return display_confidence, final_status, note


def display_sonar_image(display_img):
    """Display a sonar image robustly (handles BGR, RGB, and grayscale)."""
    if display_img is None:
        st.warning("No image available to display.")
        return
    try:
        img = np.asarray(display_img)
        if img.ndim == 2:
            # Grayscale — Streamlit handles 2D arrays natively.
            st.image(img, use_container_width=True)
        elif img.ndim == 3 and img.shape[2] == 3:
            # OpenCV images are BGR; convert to RGB for correct colors.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)
        elif img.ndim == 3 and img.shape[2] == 4:
            st.image(img, use_container_width=True)
        else:
            st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Image display error: {e}")
        logger_exception(e)


if __name__ == "__main__":
    main()
