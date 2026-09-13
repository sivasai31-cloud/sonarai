"""SONARIS-X False Positive Filter Module.

Builds a real filtering/verification layer using:
- minimum size checks
- target/seabed similarity
- local contrast
- geometry
- shadow evidence
- detection confidence
- image quality

Outputs: CONFIRMED TARGET, LIKELY TARGET, REVIEW, LIKELY NATURAL
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.context.target_context import TargetContextResult, ContextDifference


class FilterDecision(Enum):
    CONFIRMED_TARGET = "CONFIRMED TARGET"
    LIKELY_TARGET = "LIKELY TARGET"
    REVIEW = "REVIEW"
    LIKELY_NATURAL = "LIKELY NATURAL"


@dataclass
class FilterResult:
    """Result of false-positive filtering."""
    target_id: str
    original_class: str
    original_confidence: float
    filter_decision: FilterDecision
    filter_score: float  # 0-100, higher = more likely target
    reasons: List[str] = field(default_factory=list)
    deductions: List[str] = field(default_factory=list)
    image_quality_score: float = 0.0
    size_score: float = 0.0
    contrast_score: float = 0.0
    geometry_score: float = 0.0
    shadow_score: float = 0.0
    context_score: float = 0.0
    requires_human_review: bool = False


class FalsePositiveFilter:
    """Filters false positives using multiple evidence sources."""

    def __init__(self):
        self._min_width: int = 10
        self._min_height: int = 10
        self._min_area: int = 50
        self._min_confidence: float = 0.3

    def check_size(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, str]:
        """Check if target meets minimum size requirements."""
        x, y, w, h = bbox
        area = w * h

        if w < self._min_width or h < self._min_height:
            return 0.0, f"Target too small: {w}x{h}"
        if area < self._min_area:
            return 0.0, f"Target area too small: {area}"

        # Score based on size
        size_score = min(100.0, area / 100.0)
        return size_score, "Size OK"

    def check_contrast(self, local_contrast: float, image_quality: float) -> Tuple[float, str]:
        """Check local contrast evidence."""
        if local_contrast > 50:
            return 90.0, "Strong local contrast"
        elif local_contrast > 30:
            return 70.0, "Moderate local contrast"
        elif local_contrast > 15:
            return 40.0, "Weak local contrast"
        else:
            return 20.0, "Very weak contrast"

    def check_geometry(self, aspect_ratio: float, solidity: float) -> Tuple[float, str]:
        """Check geometric properties."""
        score = 50.0
        reasons = []

        # Unusual aspect ratios suggest man-made objects
        if aspect_ratio > 3.0 or (aspect_ratio < 0.33 and aspect_ratio > 0):
            score += 25.0
            reasons.append("Unusual aspect ratio")
        elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
            score += 15.0
            reasons.append("Somewhat unusual aspect ratio")
        else:
            reasons.append("Normal aspect ratio")

        # Solidity - man-made objects tend to have different solidity
        if solidity > 0.8:
            score += 15.0
            reasons.append("High solidity")
        elif solidity < 0.5:
            score += 5.0
            reasons.append("Low solidity (possibly irregular)")

        return min(100.0, score), "; ".join(reasons)

    def check_shadow(self, shadow_present: bool, shadow_area: float,
                      shadow_intensity: float) -> Tuple[float, str]:
        """Check shadow evidence."""
        if not shadow_present:
            return 10.0, "No shadow evidence"

        score = 30.0
        if shadow_area > 0.15:
            score += 30.0
        elif shadow_area > 0.05:
            score += 15.0
        if shadow_intensity < 50:
            score += 20.0

        return min(100.0, score), "Shadow evidence present"

    def check_context(self, context_diff: ContextDifference) -> Tuple[float, str]:
        """Check context difference score."""
        score = context_diff.overall_score
        if score > 70:
            return score, "Strong context difference from seabed"
        elif score > 50:
            return score, "Moderate context difference"
        elif score > 30:
            return score, "Weak context difference"
        else:
            return score, "Target similar to seabed"

    def check_image_quality(self, quality_score: float) -> float:
        """Score based on image quality."""
        if quality_score >= 80:
            return 100.0
        elif quality_score >= 60:
            return 75.0
        elif quality_score >= 40:
            return 50.0
        else:
            return 25.0

    def filter(self, target_id: str, detection, fingerprint, context_result: TargetContextResult,
               image_quality_score: float) -> FilterResult:
        """Run the complete false-positive filter."""
        from src.fingerprint.fingerprint import TargetFingerprint, GeometryFingerprint, IntensityFingerprint, ShadowFingerprint

        geo = fingerprint.geometry
        intensity = fingerprint.intensity
        shadow = fingerprint.shadow
        context_diff = context_result.context_difference

        # Run all checks
        size_score, size_reason = self.check_size(detection.bbox)
        contrast_score, contrast_reason = self.check_contrast(intensity.local_contrast, image_quality_score)
        geometry_score, geometry_reason = self.check_geometry(geo.aspect_ratio, geo.solidity)
        shadow_score, shadow_reason = self.check_shadow(shadow.shadow_present, shadow.shadow_area, shadow.shadow_intensity)
        context_score, context_reason = self.check_context(context_diff)

        # Weighted filter score
        filter_score = (
            size_score * 0.15 +
            contrast_score * 0.20 +
            geometry_score * 0.20 +
            shadow_score * 0.15 +
            context_score * 0.20 +
            detection.confidence * 100.0 * 0.10
        )
        filter_score = min(100.0, max(0.0, filter_score))

        # Determine decision
        reasons = []
        deductions = []

        if detection.confidence < self._min_confidence:
            deductions.append(f"Low detection confidence: {detection.confidence:.2f}")
        if size_score < 30:
            deductions.append(size_reason)

        # Make decision
        if filter_score >= 75 and context_diff.is_anomalous:
            decision = FilterDecision.CONFIRMED_TARGET
            reasons.append("High filter score with strong anomaly evidence")
        elif filter_score >= 60:
            decision = FilterDecision.LIKELY_TARGET
            reasons.append("Likely target but needs verification")
        elif filter_score >= 40:
            decision = FilterDecision.REVIEW
            reasons.append("Uncertain - requires human review")
        else:
            decision = FilterDecision.LIKELY_NATURAL
            reasons.append("Low confidence target - likely natural feature")

        # Add context reasons
        if context_diff.factors:
            reasons.extend(context_diff.factors)

        requires_review = decision in (FilterDecision.REVIEW,)

        return FilterResult(
            target_id=target_id,
            original_class=detection.class_name,
            original_confidence=detection.confidence,
            filter_decision=decision,
            filter_score=filter_score,
            reasons=reasons,
            deductions=deductions,
            image_quality_score=image_quality_score,
            size_score=size_score,
            contrast_score=contrast_score,
            geometry_score=geometry_score,
            shadow_score=shadow_score,
            context_score=context_score,
            requires_human_review=requires_review
        )
