"""SONARIS-X Explainability Module.

Provides WHY was this flagged explanations for every detection.
Uses measurable feature evidence instead of fake explainability.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.fingerprint.fingerprint import TargetFingerprint
from src.context.target_context import TargetContextResult
from src.anomaly.classification import ClassificationResult
from src.risk.risk_engine import RiskAssessment


@dataclass
class ExplanationEvidence:
    """A single piece of explainability evidence."""
    category: str  # e.g., "Contrast", "Geometry", "Shadow"
    description: str
    value: float
    direction: str  # "positive" (supports anomaly) or "negative" (against)


@dataclass
class Explanation:
    """Complete explainability result for a detection."""
    target_id: str
    was_flagged: bool
    reasons: List[ExplanationEvidence] = field(default_factory=list)
    summary: str = ""
    feature_evidence: Dict[str, float] = field(default_factory=dict)
    confidence_in_explanation: float = 0.0


class Explainer:
    """Generates measurable explainability for detections."""

    def __init__(self):
        self._last_explanation: Optional[Explanation] = None

    def explain(self, target_id: str, detection, fingerprint: TargetFingerprint,
                context_result: TargetContextResult, classification: ClassificationResult,
                risk_assessment: RiskAssessment) -> Explanation:
        """Generate complete explainability for a detection."""
        reasons: List[ExplanationEvidence] = []
        feature_evidence: Dict[str, float] = {}

        # Geometry evidence
        aspect_ratio = fingerprint.geometry.aspect_ratio
        if aspect_ratio > 2.0:
            reasons.append(ExplanationEvidence(
                category="Geometry",
                description=f"Unusual aspect ratio {aspect_ratio:.2f} suggests man-made object",
                value=aspect_ratio,
                direction="positive"
            ))
            feature_evidence['aspect_ratio'] = aspect_ratio
        elif aspect_ratio < 0.5:
            reasons.append(ExplanationEvidence(
                category="Geometry",
                description=f"Very flat aspect ratio {aspect_ratio:.2f}",
                value=aspect_ratio,
                direction="positive"
            ))
            feature_evidence['aspect_ratio'] = aspect_ratio
        else:
            reasons.append(ExplanationEvidence(
                category="Geometry",
                description=f"Normal aspect ratio {aspect_ratio:.2f}",
                value=aspect_ratio,
                direction="negative"
            ))
            feature_evidence['aspect_ratio'] = aspect_ratio

        # Contrast evidence
        local_contrast = fingerprint.intensity.local_contrast
        if local_contrast > 40:
            reasons.append(ExplanationEvidence(
                category="Contrast",
                description=f"Strong local contrast ({local_contrast:.1f}) indicates target stands out",
                value=local_contrast,
                direction="positive"
            ))
            feature_evidence['local_contrast'] = local_contrast
        else:
            reasons.append(ExplanationEvidence(
                category="Contrast",
                description=f"Weak local contrast ({local_contrast:.1f})",
                value=local_contrast,
                direction="negative"
            ))
            feature_evidence['local_contrast'] = local_contrast

        # Shadow evidence
        if fingerprint.shadow.shadow_present:
            shadow_area = fingerprint.shadow.shadow_area
            reasons.append(ExplanationEvidence(
                category="Shadow",
                description=f"Shadow detected with area {shadow_area:.3f}",
                value=shadow_area,
                direction="positive"
            ))
            feature_evidence['shadow_area'] = shadow_area
        else:
            reasons.append(ExplanationEvidence(
                category="Shadow",
                description="No shadow evidence detected",
                value=0.0,
                direction="negative"
            ))
            feature_evidence['shadow_area'] = 0.0

        # Texture evidence
        texture_diff = context_result.context_difference.texture_diff
        if texture_diff > 30:
            reasons.append(ExplanationEvidence(
                category="Texture",
                description=f"Distinct texture from seabed (diff={texture_diff:.1f})",
                value=texture_diff,
                direction="positive"
            ))
            feature_evidence['texture_diff'] = texture_diff
        else:
            reasons.append(ExplanationEvidence(
                category="Texture",
                description=f"Similar texture to seabed (diff={texture_diff:.1f})",
                value=texture_diff,
                direction="negative"
            ))
            feature_evidence['texture_diff'] = texture_diff

        # Intensity evidence
        intensity_diff = context_result.context_difference.intensity_diff
        if intensity_diff > 20:
            reasons.append(ExplanationEvidence(
                category="Intensity",
                description=f"Strong intensity difference from seabed ({intensity_diff:.1f})",
                value=intensity_diff,
                direction="positive"
            ))
            feature_evidence['intensity_diff'] = intensity_diff
        else:
            reasons.append(ExplanationEvidence(
                category="Intensity",
                description=f"Weak intensity difference ({intensity_diff:.1f})",
                value=intensity_diff,
                direction="negative"
            ))
            feature_evidence['intensity_diff'] = intensity_diff

        # Edge density evidence
        edge_density = fingerprint.edges.edge_density
        if edge_density > 0.08:
            reasons.append(ExplanationEvidence(
                category="Edges",
                description=f"High edge density ({edge_density:.3f}) suggests structured object",
                value=edge_density,
                direction="positive"
            ))
            feature_evidence['edge_density'] = edge_density
        else:
            reasons.append(ExplanationEvidence(
                category="Edges",
                description=f"Low edge density ({edge_density:.3f})",
                value=edge_density,
                direction="negative"
            ))
            feature_evidence['edge_density'] = edge_density

        # Context evidence
        context_score = context_result.context_difference.overall_score
        if context_score > 50:
            reasons.append(ExplanationEvidence(
                category="Context",
                description=f"Target differs from local seabed (score={context_score:.1f})",
                value=context_score,
                direction="positive"
            ))
            feature_evidence['context_score'] = context_score

        # Classification evidence
        if classification.classification.value == "UNKNOWN ANOMALY":
            reasons.append(ExplanationEvidence(
                category="Classification",
                description="Target does not match any known object class",
                value=classification.anomaly_score,
                direction="positive"
            ))

        # Calculate explanation confidence
        positive_reasons = sum(1 for r in reasons if r.direction == "positive")
        explanation_confidence = positive_reasons / max(1, len(reasons))

        # Generate summary
        was_flagged = classification.classification.value in ("UNKNOWN ANOMALY",) or risk_assessment.risk_level.value in ("HIGH", "MEDIUM")
        summary = self._generate_summary(was_flagged, reasons, feature_evidence)

        explanation = Explanation(
            target_id=target_id,
            was_flagged=was_flagged,
            reasons=reasons,
            summary=summary,
            feature_evidence=feature_evidence,
            confidence_in_explanation=explanation_confidence
        )

        self._last_explanation = explanation
        return explanation

    def _generate_summary(self, was_flagged: bool, reasons: List[ExplanationEvidence],
                             features: Dict[str, float]) -> str:
        """Generate a human-readable summary of the explanation."""
        positive = [r for r in reasons if r.direction == "positive"]
        negative = [r for r in reasons if r.direction == "negative"]

        if was_flagged:
            parts = ["This target was flagged because:"]
            if positive:
                parts.append("• " + "; ".join([r.description for r in positive[:4]]))
            if negative:
                parts.append("However: " + "; ".join([r.description for r in negative[:2]]))
        else:
            parts = ["This target was not flagged because:"]
            if negative:
                parts.append("• " + "; ".join([r.description for r in negative[:4]]))

        return " ".join(parts)
