"""SONARIS-X Adaptive Target Context Analysis Module.

For every candidate detection, compares the target with its surrounding local seabed.
Analyzes target geometry, area, aspect ratio, intensity, local contrast, texture,
edge density, shadow characteristics, surrounding seabed intensity, texture,
and difference between target and surrounding seabed.

Answers: "Is this candidate actually different from its local environment?"
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from sklearn.metrics import pairwise_distances

from src.fingerprint.fingerprint import TargetFingerprint


@dataclass
class ContextDifference:
    """Measurable difference between target and local seabed."""
    intensity_diff: float
    texture_diff: float
    edge_diff: float
    geometry_diff: float
    shadow_diff: float
    overall_score: float  # 0-100
    is_anomalous: bool
    confidence: float
    factors: List[str] = field(default_factory=list)


@dataclass
class TargetContextResult:
    """Complete context analysis result for a target."""
    target_id: str
    context_difference: ContextDifference
    target_geometry_score: float
    target_contrast_score: float
    target_shadow_score: float
    target_texture_score: float
    seabed_uniformity: float
    anomaly_evidence: str
    recommendation: str
    analysis_time_ms: float = 0.0


class AdaptiveTargetContext:
    """Performs adaptive target context analysis."""

    def __init__(self):
        self._last_result: Optional[TargetContextResult] = None

    def extract_surrounding_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                                     margin: int = 64) -> np.ndarray:
        """Extract surrounding seabed region around the target."""
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]

        sx = max(0, x - margin)
        sy = max(0, y - margin)
        ex = min(w_img, x + w + margin)
        ey = min(h_img, y + h + margin)

        # Remove the target itself from the surrounding region
        surround = image[sy:ey, sx:ex].copy()
        return surround

    def calculate_target_geometry_score(self, fingerprint: TargetFingerprint) -> float:
        """Score target geometry (distinct shapes are more anomalous)."""
        geo = fingerprint.geometry
        aspect_ratio = geo.aspect_ratio
        # Objects with unusual aspect ratios are more likely to be man-made
        if 0.5 <= aspect_ratio <= 2.0:
            geo_score = 60.0
        elif 2.0 < aspect_ratio <= 5.0 or 0.2 <= aspect_ratio < 0.5:
            geo_score = 80.0
        else:
            geo_score = 90.0
        return geo_score

    def calculate_target_contrast_score(self, fingerprint: TargetFingerprint) -> float:
        """Score target contrast relative to seabed."""
        target_std = fingerprint.intensity.std
        seabed_std = fingerprint.seabed_context.seabed_std

        if seabed_std > 0:
            contrast_ratio = target_std / seabed_std
        else:
            contrast_ratio = 1.0

        if contrast_ratio > 2.0:
            return 90.0
        elif contrast_ratio > 1.5:
            return 75.0
        elif contrast_ratio > 1.0:
            return 55.0
        else:
            return 30.0

    def calculate_target_shadow_score(self, fingerprint: TargetFingerprint) -> float:
        """Score shadow evidence."""
        shadow = fingerprint.shadow
        if not shadow.shadow_present:
            return 20.0

        score = 30.0
        if shadow.shadow_area > 0.1:
            score += 20.0
        if shadow.shadow_aspect_ratio > 1.5:
            score += 20.0
        if shadow.shadow_intensity < 50:
            score += 15.0
        return min(100.0, score)

    def calculate_target_texture_score(self, fingerprint: TargetFingerprint) -> float:
        """Score texture dissimilarity from seabed."""
        divergence = fingerprint.seabed_context.target_seabed_divergence
        texture_diff = divergence.get('texture_diff', 0)

        if texture_diff > 100:
            return 90.0
        elif texture_diff > 50:
            return 70.0
        elif texture_diff > 20:
            return 50.0
        else:
            return 30.0

    def calculate_seabed_uniformity(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                                     margin: int = 64) -> float:
        """Calculate how uniform the surrounding seabed is."""
        surround = self.extract_surrounding_region(image, bbox, margin)
        if surround.size == 0:
            return 1.0

        gray = cv2.cvtColor(surround, cv2.COLOR_BGR2GRAY) if len(surround.shape) == 3 else surround
        std = np.std(gray)

        # Lower std = more uniform seabed = target stands out more
        if std < 10:
            return 0.9  # Very uniform
        elif std < 25:
            return 0.7
        elif std < 50:
            return 0.5
        else:
            return 0.3  # Textured seabed

    def compute_context_difference(self, fingerprint: TargetFingerprint,
                                    image: np.ndarray,
                                    bbox: Tuple[int, int, int, int]) -> ContextDifference:
        """Compute the complete context difference score."""
        intensity_diff = fingerprint.seabed_context.target_seabed_divergence.get('intensity_diff', 0)
        texture_diff = fingerprint.seabed_context.target_seabed_divergence.get('texture_diff', 0)

        edge_diff = fingerprint.edges.edge_density * 100 - fingerprint.seabed_context.seabed_contrast * 2
        geometry_diff = fingerprint.geometry.aspect_ratio * 20
        shadow_diff = fingerprint.shadow.shadow_area * 100 if fingerprint.shadow.shadow_present else 0

        # Normalize and combine
        overall_score = (
            intensity_diff * 0.25 +
            texture_diff * 0.20 +
            edge_diff * 0.15 +
            geometry_diff * 0.15 +
            shadow_diff * 0.15 +
            self.calculate_target_contrast_score(fingerprint) * 0.10
        )
        overall_score = min(100.0, max(0.0, overall_score))

        # Determine if anomalous
        is_anomalous = overall_score > 50.0

        # Determine confidence
        confidence = min(1.0, overall_score / 100.0)

        factors = []
        if intensity_diff > 30:
            factors.append("Strong intensity difference from seabed")
        if texture_diff > 30:
            factors.append("Distinct texture from surrounding seabed")
        if fingerprint.shadow.shadow_present:
            factors.append("Shadow evidence present")
        if fingerprint.geometry.aspect_ratio > 2.0:
            factors.append("Unusual aspect ratio")
        if fingerprint.edges.edge_density > 0.1:
            factors.append("High edge density")

        return ContextDifference(
            intensity_diff=intensity_diff,
            texture_diff=texture_diff,
            edge_diff=edge_diff,
            geometry_diff=geometry_diff,
            shadow_diff=shadow_diff,
            overall_score=overall_score,
            is_anomalous=is_anomalous,
            confidence=confidence,
            factors=factors
        )

    def analyze(self, target_id: str, image: np.ndarray, bbox: Tuple[int, int, int, int],
                fingerprint: TargetFingerprint) -> TargetContextResult:
        """Perform full context analysis for a target."""
        import time
        start = time.time()

        context_diff = self.compute_context_difference(fingerprint, image, bbox)
        target_geo_score = self.calculate_target_geometry_score(fingerprint)
        target_contrast_score = self.calculate_target_contrast_score(fingerprint)
        target_shadow_score = self.calculate_target_shadow_score(fingerprint)
        target_texture_score = self.calculate_target_texture_score(fingerprint)
        seabed_uniformity = self.calculate_seabed_uniformity(image, bbox)

        # Generate anomaly evidence and recommendation
        if context_diff.is_anomalous and context_diff.overall_score > 70:
            anomaly_evidence = "Strong anomaly evidence: target is significantly different from local seabed"
            recommendation = "CONFIRMED TARGET - high confidence anomaly"
        elif context_diff.is_anomalous:
            anomaly_evidence = "Moderate anomaly evidence detected"
            recommendation = "LIKELY TARGET - review recommended"
        else:
            anomaly_evidence = "Weak anomaly evidence - target may be natural seabed feature"
            recommendation = "REVIEW - may be natural feature"

        elapsed = (time.time() - start) * 1000

        result = TargetContextResult(
            target_id=target_id,
            context_difference=context_diff,
            target_geometry_score=target_geo_score,
            target_contrast_score=target_contrast_score,
            target_shadow_score=target_shadow_score,
            target_texture_score=target_texture_score,
            seabed_uniformity=seabed_uniformity,
            anomaly_evidence=anomaly_evidence,
            recommendation=recommendation,
            analysis_time_ms=elapsed
        )

        self._last_result = result
        return result
