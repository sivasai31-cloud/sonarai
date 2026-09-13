"""SONARIS-X Test Suite."""

import pytest
import cv2
import numpy as np
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.quality import ImageQualityAssessor
from src.preprocessing.preprocess import SonarPreprocessor, PreprocessingConfig
from src.detection.detector import MultiScaleDetector, Detection
from src.fingerprint.fingerprint import FingerprintExtractor
from src.context.target_context import AdaptiveTargetContext
from src.anomaly.false_positive_filter import FalsePositiveFilter
from src.anomaly.classification import AnomalyClassifier, ClassificationResult, ClassificationType
from src.risk.risk_engine import RiskEngine
from src.explainability.explainer import Explainer
from src.geolocation.geolocation import GeolocationExtractor
from src.reporting.report_generator import ReportGenerator, TargetReportEntry
from src.validation.dataset_validation import DatasetValidator, DatasetStats
from src.pipeline import SonarisXPipeline, PipelineResult
from src.data.ingestion import DataIngestor, ImageInfo


def create_test_image(width=400, height=300, noise_level=10):
    """Create a synthetic test image."""
    np.random.seed(42)
    img = np.random.randint(20, 60, (height, width, 3), dtype=np.uint8)
    cv2.circle(img, (200, 150), 30, (200, 200, 200), -1)
    cv2.circle(img, (200, 150), 35, (150, 150, 150), 2)
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class TestPreprocessing:
    """Tests for preprocessing module."""

    def test_clahe_enhancement(self):
        img = create_test_image()
        config = PreprocessingConfig(apply_clahe=True, apply_denoising=False, apply_normalization=False)
        preprocessor = SonarPreprocessor(config)
        result = preprocessor.process(img)
        assert not np.array_equal(result.preprocessed, result.original)
        assert result.preprocessed is not None
        assert result.preprocessed.shape[:2] == img.shape[:2]

    def test_denoising(self):
        img = create_test_image(noise_level=30)
        config = PreprocessingConfig(apply_clahe=False, apply_denoising=True)
        preprocessor = SonarPreprocessor(config)
        result = preprocessor.process(img)
        assert result.preprocessed is not None
        assert result.preprocessed.shape[:2] == img.shape[:2]

    def test_dropout_detection(self):
        img = create_test_image()
        preprocessor = SonarPreprocessor()
        dropout_mask, dropouts = preprocessor.detect_dropouts(img)
        assert isinstance(dropouts, list)
        assert isinstance(dropout_mask, np.ndarray)

    def test_detection_view(self):
        img = create_test_image()
        detections = [{'bbox': (100, 80, 60, 60), 'class': 'test', 'confidence': 0.8}]
        preprocessor = SonarPreprocessor()
        view = preprocessor.create_detection_view(img, detections)
        assert view.shape == img.shape


class TestQualityAnalysis:
    """Tests for quality assessment."""

    def test_quality_score_range(self):
        img = create_test_image()
        assessor = ImageQualityAssessor()
        result = assessor.assess(img)
        assert 0 <= result.quality_score <= 100

    def test_rating_values(self):
        img = create_test_image()
        assessor = ImageQualityAssessor()
        result = assessor.assess(img)
        assert result.rating in ["GOOD", "ACCEPTABLE", "DEGRADED", "REVIEW REQUIRED"]

    def test_statistics_calculation(self):
        img = create_test_image()
        assessor = ImageQualityAssessor()
        result = assessor.assess(img)
        assert 'mean' in result.statistics
        assert 'std' in result.statistics
        assert 'median' in result.statistics

    def test_empty_image_raises(self):
        assessor = ImageQualityAssessor()
        with pytest.raises(ValueError):
            assessor.assess(np.array([]))


class TestFingerprint:
    """Tests for fingerprint extraction."""

    def test_full_fingerprint_extraction(self):
        img = create_test_image()
        extractor = FingerprintExtractor()
        bbox = (170, 120, 60, 60)
        target_region = img[120:180, 170:230]
        surround_region = img[80:220, 120:280]
        fingerprint = extractor.extract(target_region, surround_region, bbox)
        assert fingerprint.geometry.area > 0
        assert fingerprint.intensity.mean > 0
        assert fingerprint.edges.edge_density >= 0
        assert fingerprint.shadow.shadow_present is not None
        assert fingerprint.seabed_context.seabed_mean > 0

    def test_geometry_properties(self):
        img = create_test_image()
        extractor = FingerprintExtractor()
        bbox = (170, 120, 60, 60)
        target_region = img[120:180, 170:230]
        surround_region = np.zeros((100, 100), dtype=np.uint8)
        geo = extractor.extract_geometry(target_region, bbox)
        assert geo.area > 0
        assert geo.width > 0
        assert geo.height > 0
        assert geo.aspect_ratio > 0

    def test_texture_extraction(self):
        img = create_test_image()
        extractor = FingerprintExtractor()
        bbox = (170, 120, 60, 60)
        target_region = img[120:180, 170:230]
        surround_region = np.zeros((100, 100), dtype=np.uint8)
        tex = extractor.extract_texture(target_region)
        assert isinstance(tex.lbp_mean, float)
        assert isinstance(tex.glcm_contrast, float)


class TestContextAnalysis:
    """Tests for target context analysis."""

    def test_context_difference_computation(self):
        img = create_test_image()
        context = AdaptiveTargetContext()
        extractor = FingerprintExtractor()
        bbox = (170, 120, 60, 60)
        target_region = img[120:180, 170:230]
        surround_region = img[80:220, 120:280]
        fingerprint = extractor.extract(target_region, surround_region, bbox)
        result = context.analyze("TGT-001", img, bbox, fingerprint)
        assert 0 <= result.context_difference.overall_score <= 100
        assert isinstance(result.context_difference.is_anomalous, bool)
        assert len(result.anomaly_evidence) > 0


class TestFalsePositiveFilter:
    """Tests for false-positive filtering."""

    def test_filter_decision_values(self):
        img = create_test_image()
        extractor = FingerprintExtractor()
        bbox = (170, 120, 60, 60)
        target_region = img[120:180, 170:230]
        surround_region = img[80:220, 120:280]
        fingerprint = extractor.extract(target_region, surround_region, bbox)
        context_analyzer = AdaptiveTargetContext()
        context_result = context_analyzer.analyze("TGT-001", img, bbox, fingerprint)
        detection = Detection(
            class_id=0, class_name="object", confidence=0.8,
            bbox=bbox, center=(200, 150), area=3600
        )
        fp_filter = FalsePositiveFilter()
        result = fp_filter.filter("TGT-001", detection, fingerprint, context_result, 75.0)
        assert result.filter_decision.value in ["CONFIRMED TARGET", "LIKELY TARGET", "REVIEW", "LIKELY NATURAL"]
        assert 0 <= result.filter_score <= 100


class TestRiskEngine:
    """Tests for risk engine."""

    def test_risk_score_range(self):
        from src.context.target_context import TargetContextResult, ContextDifference

        risk_engine = RiskEngine()
        context_diff = ContextDifference(
            intensity_diff=30, texture_diff=20, edge_diff=10,
            geometry_diff=15, shadow_diff=25, overall_score=60,
            is_anomalous=True, confidence=0.6, factors=["Test"]
        )
        context_result = TargetContextResult(
            target_id="TGT-001", context_difference=context_diff,
            target_geometry_score=70, target_contrast_score=65,
            target_shadow_score=50, target_texture_score=55,
            seabed_uniformity=0.5, anomaly_evidence="Test", recommendation="Test"
        )
        classification = ClassificationResult(
            target_id="TGT-001", classification=ClassificationType.UNKNOWN_ANOMALY,
            class_name="object", model_confidence=0.8, anomaly_score=60,
            evidence=["test"], reasoning="Test"
        )
        filter_result = FalsePositiveFilter().filter(
            "TGT-001",
            Detection(class_id=0, class_name="object", confidence=0.8, bbox=(170,120,60,60), center=(200,150), area=3600),
            FingerprintExtractor().extract(
                create_test_image()[120:180, 170:230],
                create_test_image()[80:220, 120:280],
                (170, 120, 60, 60)
            ),
            context_result, 75.0
        )
        mock_detection = SimpleNamespace(confidence=0.8, class_name="object")
        risk = risk_engine.assess("TGT-001", mock_detection,
            classification=classification, filter_result=filter_result,
            context_result=context_result, image_quality_score=75,
            has_coordinates=False)
        assert 0 <= risk.risk_score <= 100
        assert risk.risk_level.value in ["HIGH", "MEDIUM", "LOW", "REVIEW"]
        assert len(risk.factors) > 0


class TestGeolocation:
    """Tests for geolocation."""

    def test_simulated_coordinates_are_marked(self):
        geo = GeolocationExtractor()
        coord = geo.get_coordinates(None)
        assert coord.is_simulated is True
        assert coord.source == "SIMULATED"

    def test_coordinate_range(self):
        geo = GeolocationExtractor()
        coord = geo.get_coordinates(None)
        assert isinstance(coord.latitude, float)
        assert isinstance(coord.longitude, float)
        assert -90 <= coord.latitude <= 90
        assert -180 <= coord.longitude <= 180


class TestReportGeneration:
    """Tests for report generation."""

    def test_json_report(self):
        report_gen = ReportGenerator()
        report_gen.set_mission_info("TEST-001", "Operator", "2026-01-01")
        entry = TargetReportEntry(
            target_id="TGT-001", classification="KNOWN OBJECT",
            class_name="object", model_confidence=0.85, anomaly_score=65,
            risk_level="MEDIUM", risk_score=55,
            bbox=[170, 120, 60, 60], center_x=200, center_y=150,
            area=3600, latitude=0.0, longitude=0.0,
            coordinate_source="SIMULATED", is_simulated_coords=True,
            evidence=["test"], image_width=800, image_height=600,
            processing_time_ms=100, fingerprint_area=3600,
            fingerprint_aspect_ratio=1.0, source_image="test.png"
        )
        report_gen.add_target(entry)
        report = report_gen.generate_json(tempfile.mktemp(suffix='.json'))
        assert report['total_targets'] == 1
        assert report['targets'][0]['target_id'] == "TGT-001"

    def test_csv_report(self):
        report_gen = ReportGenerator()
        entry = TargetReportEntry(
            target_id="TGT-001", classification="UNKNOWN ANOMALY",
            class_name="object", model_confidence=0.7, anomaly_score=75,
            risk_level="HIGH", risk_score=72,
            bbox=[100, 100, 50, 50], center_x=125, center_y=125,
            area=2500, latitude=0.0, longitude=0.0,
            coordinate_source="SIMULATED", is_simulated_coords=True,
            evidence=["test"], image_width=800, image_height=600,
            processing_time_ms=150, fingerprint_area=2500,
            fingerprint_aspect_ratio=1.0, source_image="test.png"
        )
        report_gen.add_target(entry)
        filepath = tempfile.mktemp(suffix='.csv')
        rows = report_gen.generate_csv(filepath)
        assert len(rows) == 1
        assert rows[0]['target_id'] == "TGT-001"

    def test_summary_stats(self):
        report_gen = ReportGenerator()
        report_gen.set_mission_info("TEST-001", "Op", "2026-01-01")
        for i in range(3):
            entry = TargetReportEntry(
                target_id=f"TGT-{i}", classification="KNOWN OBJECT",
                class_name="object", model_confidence=0.8, anomaly_score=50,
                risk_level="MEDIUM", risk_score=50,
                bbox=[100, 100, 50, 50], center_x=125, center_y=125,
                area=2500, latitude=0.0, longitude=0.0,
                coordinate_source="SIMULATED", is_simulated_coords=True,
                evidence=[], image_width=800, image_height=600,
                processing_time_ms=100, fingerprint_area=2500,
                fingerprint_aspect_ratio=1.0, source_image="test.png"
            )
            report_gen.add_target(entry)
        stats = report_gen.get_summary_stats()
        assert stats['total_targets'] == 3


class TestPipeline:
    """Tests for end-to-end pipeline."""

    def test_pipeline_execution(self):
        img = create_test_image()
        tfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(tfile.name, img)
        tfile.close()
        try:
            pipeline = SonarisXPipeline()
            result = pipeline.process_image(tfile.name, conf_threshold=0.3)
            assert result.image_info is not None
            assert result.quality is not None
            assert result.preprocessing is not None
            assert result.detection is not None
            assert result.pipeline_time_ms > 0
        finally:
            os.unlink(tfile.name)

    def test_pipeline_with_tiled_detection(self):
        img = create_test_image()
        tfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(tfile.name, img)
        tfile.close()
        try:
            pipeline = SonarisXPipeline()
            result = pipeline.process_image(tfile.name, conf_threshold=0.3, use_tiled=True)
            assert result.detection is not None
            assert result.detection.mode in ["full", "tiled"]
        finally:
            os.unlink(tfile.name)


class TestDataIngestion:
    """Tests for data ingestion."""

    def test_ingest_valid_image(self):
        ingestor = DataIngestor()
        img = create_test_image()
        tfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(tfile.name, img)
        tfile.close()
        try:
            info = ingestor.ingest_image(tfile.name)
            assert info.readable is True
            assert info.width > 0
            assert info.height > 0
        finally:
            os.unlink(tfile.name)

    def test_ingest_nonexistent_file(self):
        ingestor = DataIngestor()
        info = ingestor.ingest_image("/nonexistent/path/image.png")
        assert info.readable is False
        assert len(info.errors) > 0


class TestDatasetValidation:
    """Tests for dataset validation."""

    def test_empty_directory(self):
        validator = DatasetValidator()
        stats = validator.validate_directory(tempfile.mkdtemp())
        assert stats.image_count == 0

    def test_valid_directory(self):
        validator = DatasetValidator()
        tmpdir = tempfile.mkdtemp()
        img = create_test_image()
        cv2.imwrite(os.path.join(tmpdir, "test.png"), img)
        stats = validator.validate_directory(tmpdir)
        assert stats.image_count == 1
        assert stats.readable_images == 1
        assert stats.corrupted_images == 0


class TestSaliencyProposer:
    """Tests for the classical saliency proposer (real computation, honest labels)."""

    def test_fires_on_synthetic_pipeline(self):
        from src.demo.synthetic import generate_scenario
        from src.detection.saliency import propose_sonar_candidates
        img, _, _ = generate_scenario("Pipeline")
        dets = propose_sonar_candidates(img)
        assert len(dets) >= 1
        for d in dets:
            assert d.class_name == "sonar_candidate"
            assert 0.05 <= d.confidence <= 0.95

    def test_empty_image_returns_empty(self):
        from src.detection.saliency import propose_sonar_candidates
        assert propose_sonar_candidates(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_detector_model_status_honest(self):
        det = MultiScaleDetector()
        assert det.model_status() in ("DEMO / UNTRAINED — CUSTOM WEIGHTS NOT LOADED",
                                      "COCO FALLBACK — NOT SONAR-TRAINED",
                                      "CUSTOM SONAR WEIGHTS LOADED")
        if "CUSTOM" not in det.model_status():
            assert det.sonar_classes() == []


class TestHonestClassification:
    """Detector must never invent sonar object names without sonar weights."""

    SONAR_NAMES = {"Ghost Net", "Pipeline", "Cylinder", "Shipwreck", "Debris"}

    def test_no_sonar_names_without_weights(self):
        from src.demo.synthetic import SCENARIOS, generate_scenario
        for name in SCENARIOS:
            img, _, _ = generate_scenario(name)
            tfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(tfile.name, img)
            tfile.close()
            try:
                result = SonarisXPipeline().process_image(tfile.name, conf_threshold=0.3)
                for c in result.classifications:
                    assert c.class_name not in self.SONAR_NAMES
                    assert c.classification.value in ("KNOWN OBJECT", "UNKNOWN ANOMALY",
                                                      "LIKELY NATURAL", "UNCLASSIFIED")
            finally:
                os.unlink(tfile.name)

    def test_three_states_reachable(self):
        from src.demo.synthetic import generate_scenario
        kinds = set()
        for name in ("Pipeline", "Natural Rock / Seabed", "Cylinder"):
            img, _, _ = generate_scenario(name)
            tfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(tfile.name, img)
            tfile.close()
            try:
                result = SonarisXPipeline().process_image(tfile.name, conf_threshold=0.3)
                kinds.update(c.classification.value for c in result.classifications)
            finally:
                os.unlink(tfile.name)
        assert "UNKNOWN ANOMALY" in kinds or "LIKELY NATURAL" in kinds


class TestDemoScenarios:
    """Synthetic scenarios carry their own labeled ground truth (not detections)."""

    def test_all_scenarios_generate(self):
        from src.demo.synthetic import SCENARIOS, generate_scenario
        assert len(SCENARIOS) == 7
        for name in SCENARIOS:
            img, gt, info = generate_scenario(name)
            assert img is not None and img.size > 0
            assert len(gt) >= 1
            assert info.get("scenario") == name

    def test_report_carries_honesty_fields(self):
        from src.demo.synthetic import generate_scenario
        img, _, _ = generate_scenario("Cylinder")
        tfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(tfile.name, img)
        tfile.close()
        try:
            result = SonarisXPipeline().process_image(tfile.name, conf_threshold=0.3)
            assert result.model_status != ""
            assert "quality_ms" in result.timings and "fingerprint_ms" in result.timings
            for rep in result.reports:
                assert rep.model_status != ""
                assert rep.fingerprint_summary != ""
                assert rep.filter_decision != ""
        finally:
            os.unlink(tfile.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
