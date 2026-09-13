"""SONARIS-X Risk Engine Module.

Creates an actual risk-scoring engine based on:
- model confidence
- anomaly score
- object class
- object size
- evidence strength
- image quality
- location availability

Outputs: HIGH, MEDIUM, LOW, REVIEW
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.context.target_context import TargetContextResult
from src.anomaly.false_positive_filter import FilterResult
from src.anomaly.classification import ClassificationResult


class RiskLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    REVIEW = "REVIEW"


@dataclass
class RiskFactor:
    """A single factor contributing to risk."""
    factor_name: str
    factor_value: float
    weight: float
    contribution: float
    description: str


@dataclass
class RiskAssessment:
    """Complete risk assessment for a target."""
    target_id: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    factors: List[RiskFactor] = field(default_factory=list)
    reasoning: str = ""
    confidence_interval: str = ""
    assessment_time_ms: float = 0.0


class RiskEngine:
    """Calculates evidence-based risk scores."""

    def __init__(self):
        self._weights = {
            'model_confidence': 0.20,
            'anomaly_score': 0.25,
            'object_size': 0.10,
            'evidence_strength': 0.20,
            'image_quality': 0.10,
            'location_availability': 0.05,
            'false_positive_risk': 0.10
        }

    def assess(self, target_id: str, detection, classification: ClassificationResult,
               filter_result: FilterResult, context_result: TargetContextResult,
               image_quality_score: float, has_coordinates: bool = False) -> RiskAssessment:
        """Calculate complete risk assessment."""
        import time
        start = time.time()

        factors: List[RiskFactor] = []
        total_weighted_score = 0.0

        # 1. Model Confidence factor
        conf_value = detection.confidence
        conf_contribution = conf_value * self._weights['model_confidence'] * 100
        total_weighted_score += conf_contribution
        factors.append(RiskFactor(
            factor_name="Model Confidence",
            factor_value=conf_value,
            weight=self._weights['model_confidence'],
            contribution=conf_contribution,
            description=f"Detection confidence: {conf_value:.2f}"
        ))

        # 2. Anomaly Score factor
        anomaly_value = classification.anomaly_score / 100.0
        anomaly_contribution = anomaly_value * self._weights['anomaly_score'] * 100
        total_weighted_score += anomaly_contribution
        factors.append(RiskFactor(
            factor_name="Anomaly Score",
            factor_value=classification.anomaly_score,
            weight=self._weights['anomaly_score'],
            contribution=anomaly_contribution,
            description=f"Anomaly evidence score: {classification.anomaly_score:.1f}"
        ))

        # 3. Object Size factor
        size_value = classification.context_result.context_difference.geometry_diff / 100.0 if classification.context_result else 0.5
        size_contribution = size_value * self._weights['object_size'] * 100
        total_weighted_score += size_contribution
        factors.append(RiskFactor(
            factor_name="Object Size",
            factor_value=size_value,
            weight=self._weights['object_size'],
            contribution=size_contribution,
            description=f"Target relative size: {size_value:.2f}"
        ))

        # 4. Evidence Strength factor
        evidence_value = filter_result.filter_score / 100.0
        evidence_contribution = evidence_value * self._weights['evidence_strength'] * 100
        total_weighted_score += evidence_contribution
        factors.append(RiskFactor(
            factor_name="Evidence Strength",
            factor_value=filter_result.filter_score,
            weight=self._weights['evidence_strength'],
            contribution=evidence_contribution,
            description=f"Filter score: {filter_result.filter_score:.1f}"
        ))

        # 5. Image Quality factor
        quality_value = image_quality_score / 100.0
        quality_contribution = quality_value * self._weights['image_quality'] * 100
        total_weighted_score += quality_contribution
        factors.append(RiskFactor(
            factor_name="Image Quality",
            factor_value=image_quality_score,
            weight=self._weights['image_quality'],
            contribution=quality_contribution,
            description=f"Image quality score: {image_quality_score:.1f}"
        ))

        # 6. Location Availability factor
        loc_value = 1.0 if has_coordinates else 0.0
        loc_contribution = loc_value * self._weights['location_availability'] * 100
        total_weighted_score += loc_contribution
        factors.append(RiskFactor(
            factor_name="Location Availability",
            factor_value=loc_value,
            weight=self._weights['location_availability'],
            contribution=loc_contribution,
            description="GPS coordinates available" if has_coordinates else "No GPS coordinates"
        ))

        # 7. False Positive Risk factor (inverse)
        fp_value = 1.0 - (filter_result.filter_score / 100.0)
        fp_contribution = fp_value * self._weights['false_positive_risk'] * 100
        total_weighted_score += fp_contribution
        factors.append(RiskFactor(
            factor_name="False Positive Risk",
            factor_value=fp_value,
            weight=self._weights['false_positive_risk'],
            contribution=fp_contribution,
            description=f"False positive risk: {fp_value:.2f}"
        ))

        # Clamp total score
        risk_score = min(100.0, max(0.0, total_weighted_score))

        # Determine risk level
        if risk_score >= 70:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 50:
            risk_level = RiskLevel.MEDIUM
        elif risk_score >= 30:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.REVIEW

        # Generate reasoning
        reasoning_parts = []
        if risk_level == RiskLevel.HIGH:
            reasoning_parts.append("HIGH RISK: Significant anomaly detected with strong evidence")
        elif risk_level == RiskLevel.MEDIUM:
            reasoning_parts.append("MEDIUM RISK: Notable anomaly requiring attention")
        elif risk_level == RiskLevel.LOW:
            reasoning_parts.append("LOW RISK: Minor anomaly or low-confidence detection")
        else:
            reasoning_parts.append("REVIEW NEEDED: Insufficient evidence for confident assessment")

        # Add top contributing factors
        sorted_factors = sorted(factors, key=lambda f: f.contribution, reverse=True)
        top_factors = sorted_factors[:3]
        reasoning_parts.append(
            f"Top factors: {', '.join([f'{f.factor_name}: {f.contribution:.1f}' for f in top_factors])}"
        )

        # Confidence interval estimate
        conf_low = max(0.0, risk_score - 10.0)
        conf_high = min(100.0, risk_score + 10.0)
        confidence_interval = f"[{conf_low:.1f}, {conf_high:.1f}]"

        elapsed = (time.time() - start) * 1000

        assessment = RiskAssessment(
            target_id=target_id,
            risk_level=risk_level,
            risk_score=risk_score,
            factors=factors,
            reasoning=" ".join(reasoning_parts),
            confidence_interval=confidence_interval,
            assessment_time_ms=elapsed
        )

        return assessment
