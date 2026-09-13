"""SONARIS-X Main Pipeline Module.

Orchestrates the complete processing pipeline:
RAW SONAR → DATA INGESTION → QUALITY ASSESSMENT → SONAR PREPROCESSING
→ MULTI-SCALE DETECTION → TARGET FINGERPRINT → LOCAL SEABED CONTEXT
→ FALSE-POSITIVE FILTER → KNOWN/UNKNOWN → CONFIDENCE → RISK
→ EXPLAINABILITY → GEOLOCATION → TARGET CATALOG → REPORT
"""

import time
import json
import uuid
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.data.ingestion import DataIngestor, ImageInfo
from src.preprocessing.quality import ImageQualityAssessor
from src.preprocessing.preprocess import SonarPreprocessor, PreprocessingConfig
from src.detection.detector import MultiScaleDetector, Detection, DetectionResult
from src.fingerprint.fingerprint import FingerprintExtractor
from src.context.target_context import AdaptiveTargetContext, TargetContextResult
from src.anomaly.false_positive_filter import FalsePositiveFilter
from src.anomaly.classification import AnomalyClassifier, ClassificationResult
from src.risk.risk_engine import RiskEngine, RiskAssessment
from src.explainability.explainer import Explainer
from src.geolocation.geolocation import GeolocationExtractor, Coordinates
from src.reporting.report_generator import ReportGenerator, TargetReportEntry

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    image_info: Optional[ImageInfo] = None
    quality: Optional[Any] = None
    preprocessing: Optional[Any] = None
    detection: Optional[DetectionResult] = None
    fingerprints: List[Any] = field(default_factory=list)
    context_results: List[Any] = field(default_factory=list)
    filter_results: List[Any] = field(default_factory=list)
    classifications: List[Any] = field(default_factory=list)
    risks: List[Any] = field(default_factory=list)
    explanations: List[Any] = field(default_factory=list)
    coordinates: List[Coordinates] = field(default_factory=list)
    reports: List[TargetReportEntry] = field(default_factory=list)
    pipeline_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    dropout_overlap: List[bool] = field(default_factory=list)
    model_status: str = ""


class SonarisXPipeline:
    """Main pipeline orchestrator for SONARIS-X."""

    def __init__(self, model_path: Optional[str] = None,
                 preprocess_config: Optional[PreprocessingConfig] = None):
        self.detector = MultiScaleDetector(model_path)
        self.ingestor = DataIngestor()
        self.quality_assessor = ImageQualityAssessor()
        self.preprocessor = SonarPreprocessor(preprocess_config)
        self.fingerprint_extractor = FingerprintExtractor()
        self.context_analyzer = AdaptiveTargetContext()
        self.false_positive_filter = FalsePositiveFilter()
        self.classifier = AnomalyClassifier()
        self.risk_engine = RiskEngine()
        self.explainer = Explainer()
        self.geolocation = GeolocationExtractor()
        self.report_generator = ReportGenerator()

        self._results: List[PipelineResult] = []
        self._target_counter = 0
        # Honesty wiring: only advertise sonar classes the loaded weights
        # can actually recognize (empty for COCO fallback / demo).
        try:
            self.classifier.set_sonar_classes(self.detector.sonar_classes())
        except Exception:
            pass

    @staticmethod
    def _bbox_overlaps_dropout(bbox, dropout_regions) -> bool:
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        for dx, dy, dw, dh in dropout_regions or []:
            if x < dx + dw and x2 > dx and y < dy + dh and y2 > dy:
                return True
        return False

    def process_image(self, image_path: str,
                      conf_threshold: float = 0.5,
                      use_tiled: bool = False,
                      image_quality_score: Optional[float] = None) -> PipelineResult:
        """Process a single image through the complete pipeline."""
        pipeline_start = time.time()
        warnings = []

        # Step 1: Data Ingestion
        image_info = self.ingestor.ingest_image(image_path)
        if not image_info.readable:
            warnings.append(f"Image {image_info.filename} could not be read")
            return PipelineResult(
                image_info=image_info, quality=None, preprocessing=None, detection=None,
                pipeline_time_ms=0, warnings=warnings,
                fingerprints=[], context_results=[], filter_results=[],
                classifications=[], risks=[], explanations=[],
                coordinates=[], reports=[]
            )

        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            warnings.append(f"Image {image_info.filename} could not be decoded by OpenCV")
            return PipelineResult(
                image_info=image_info, quality=None, preprocessing=None, detection=None,
                pipeline_time_ms=(time.time() - pipeline_start) * 1000, warnings=warnings,
                fingerprints=[], context_results=[], filter_results=[],
                classifications=[], risks=[], explanations=[],
                coordinates=[], reports=[]
            )

        # Step 2: Quality Assessment
        t0 = time.time()
        quality = self.quality_assessor.assess(img)
        timings: Dict[str, float] = {}
        timings["quality_ms"] = (time.time() - t0) * 1000
        if image_quality_score is None:
            image_quality_score = quality.quality_score

        if quality.rating == "REVIEW REQUIRED":
            warnings.append(f"Image quality is REVIEW REQUIRED: {quality.quality_score:.1f}")

        # Step 3: Preprocessing
        t0 = time.time()
        preprocessed = self.preprocessor.process(img)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        # Step 4: Multi-Scale Detection
        detection = self.detector.detect(
            preprocessed.preprocessed,
            conf_threshold=conf_threshold,
            use_tiled=use_tiled
        )
        timings["inference_ms"] = detection.inference_time_ms
        for wmsg in (detection.warnings or []):
            warnings.append(wmsg)

        # Step 5-12: Process each detection
        fingerprints = []
        context_results = []
        filter_results = []
        classifications = []
        risks = []
        explanations = []
        coordinates = []
        reports = []
        dropout_overlap: List[bool] = []
        timings.setdefault("fingerprint_ms", 0.0)
        timings.setdefault("context_ms", 0.0)
        timings.setdefault("filter_risk_explain_ms", 0.0)

        for i, det in enumerate(detection.detections):
            target_id = f"TGT-{uuid.uuid4().hex[:8].upper()}"

            # Step 5: Fingerprint
            # Get target region
            x, y, w, h = det.bbox
            h_img, w_img = img.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)

            if x2 <= x1 or y2 <= y1:
                warnings.append(f"Target {target_id} has invalid bbox, skipping")
                continue

            target_region = img[y1:y2, x1:x2]
            surround_region = self.context_analyzer.extract_surrounding_region(img, det.bbox)

            t0 = time.time()
            fingerprint = self.fingerprint_extractor.extract(target_region, surround_region, det.bbox)
            timings["fingerprint_ms"] += (time.time() - t0) * 1000
            fingerprints.append(fingerprint)

            # Step 6: Context Analysis
            t0 = time.time()
            context_result = self.context_analyzer.analyze(target_id, img, det.bbox, fingerprint)
            timings["context_ms"] += (time.time() - t0) * 1000
            context_results.append(context_result)

            # Step 7: False Positive Filter
            t0 = time.time()
            filter_result = self.false_positive_filter.filter(
                target_id, det, fingerprint, context_result, image_quality_score
            )
            filter_results.append(filter_result)

            # Step 8: Known/Unknown Classification
            classification = self.classifier.classify(
                target_id, det, fingerprint, context_result
            )
            classifications.append(classification)

            # Step 9-10: Risk Assessment
            has_coords = False
            risk = self.risk_engine.assess(
                target_id, det, classification, filter_result, context_result,
                image_quality_score, has_coords
            )
            risks.append(risk)

            # Step 11: Explainability
            explanation = self.explainer.explain(
                target_id, det, fingerprint, context_result, classification, risk
            )
            explanations.append(explanation)
            timings["filter_risk_explain_ms"] += (time.time() - t0) * 1000

            # DATA QUALITY WARNING: detection overlapping dropout regions
            overlaps_dropout = self._bbox_overlaps_dropout(
                det.bbox, preprocessed.dropout_regions
            )
            dropout_overlap.append(overlaps_dropout)
            if overlaps_dropout:
                warnings.append(
                    f"DATA QUALITY WARNING: {target_id} overlaps dropout/degraded data."
                )

            # Step 12: Geolocation
            coord = self.geolocation.get_coordinates(det)
            coordinates.append(coord)

            # Create report entry
            fpsum = (
                f"area={fingerprint.geometry.area},ar={fingerprint.geometry.aspect_ratio:.2f},"
                f"mean={fingerprint.intensity.mean:.1f},std={fingerprint.intensity.std:.1f},"
                f"contrast={fingerprint.intensity.local_contrast:.1f},"
                f"lbp={fingerprint.texture.lbp_mean:.2f},glcm={fingerprint.texture.glcm_contrast:.1f},"
                f"edge={fingerprint.edges.edge_density:.3f},"
                f"shadow={fingerprint.shadow.shadow_present}:{fingerprint.shadow.shadow_area:.2f}"
            )
            report_entry = TargetReportEntry(
                target_id=target_id,
                classification=classification.classification.value,
                class_name=classification.class_name,
                model_confidence=classification.model_confidence,
                anomaly_score=classification.anomaly_score,
                risk_level=risk.risk_level.value,
                risk_score=risk.risk_score,
                bbox=list(det.bbox),
                center_x=det.center[0],
                center_y=det.center[1],
                area=fingerprint.geometry.area,
                latitude=coord.latitude,
                longitude=coord.longitude,
                coordinate_source=coord.source,
                is_simulated_coords=coord.is_simulated,
                evidence=classification.evidence,
                image_width=img.shape[1],
                image_height=img.shape[0],
                processing_time_ms=0,
                fingerprint_area=fingerprint.geometry.area,
                fingerprint_aspect_ratio=fingerprint.geometry.aspect_ratio,
                source_image=image_path,
                model_status=self.detector.model_status(),
                quality_score=image_quality_score,
                quality_rating=quality.rating,
                overlaps_dropout=overlaps_dropout,
                fingerprint_summary=fpsum,
                filter_decision=filter_result.filter_decision.value,
            )
            reports.append(report_entry)

            self._target_counter += 1

        pipeline_time = (time.time() - pipeline_start) * 1000

        result = PipelineResult(
            image_info=image_info,
            quality=quality,
            preprocessing=preprocessed,
            detection=detection,
            fingerprints=fingerprints,
            context_results=context_results,
            filter_results=filter_results,
            classifications=classifications,
            risks=risks,
            explanations=explanations,
            coordinates=coordinates,
            reports=reports,
            pipeline_time_ms=pipeline_time,
            warnings=warnings,
            timings={k: round(v, 2) for k, v in timings.items()},
            dropout_overlap=dropout_overlap,
            model_status=self.detector.model_status(),
        )

        self._results.append(result)
        for report_entry in reports:
            self.report_generator.add_target(report_entry)

        logger.info(f"Pipeline complete: {len(detection.detections)} detections in {pipeline_time:.1f}ms")
        return result

    def process_batch(self, image_paths: List[str], **kwargs) -> List[PipelineResult]:
        """Process multiple images."""
        results = []
        for path in image_paths:
            result = self.process_image(path, **kwargs)
            results.append(result)
        return results

    def get_summary(self) -> Dict:
        """Get pipeline summary statistics."""
        total_detections = sum(len(r.detection.detections) if r.detection and r.detection.detections else 0 for r in self._results)
        total_time = sum(r.pipeline_time_ms for r in self._results)
        high_risk = sum(len([f for f in (r.risks or []) if f.risk_level.value == 'HIGH']) for r in self._results)
        unknown = sum(len([c for c in (r.classifications or []) if c.classification.value == 'UNKNOWN ANOMALY']) for r in self._results)

        return {
            'images_processed': len(self._results),
            'total_detections': total_detections,
            'high_risk_targets': high_risk,
            'unknown_anomalies': unknown,
            'total_pipeline_time_ms': total_time,
            'average_time_per_image_ms': round(total_time / max(1, len(self._results)), 1),
            'report_summary': self.report_generator.get_summary_stats()
        }
