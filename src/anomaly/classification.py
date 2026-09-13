"""SONARIS-X Anomaly Classification Module.

Three useful states:
  KNOWN OBJECT   — only when a sonar-trained model actually recognizes one
                   of its trained sonar classes with sufficient confidence.
  UNKNOWN ANOMALY — strong anomaly evidence but no confident known-class match.
  LIKELY NATURAL  — weak anomaly evidence; resembles surrounding seabed.

Honesty rule: COCO-fallback and classical-saliency candidates
(class "sonar_candidate" / generic "object") can NEVER become KNOWN OBJECT
with a sonar object name. Without custom sonar weights, outputs are limited
to UNKNOWN ANOMALY / LIKELY NATURAL / REVIEW.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.context.target_context import TargetContextResult


class ClassificationType(Enum):
    KNOWN_OBJECT = "KNOWN OBJECT"
    UNKNOWN_ANOMALY = "UNKNOWN ANOMALY"
    LIKELY_NATURAL = "LIKELY NATURAL"
    UNCLASSIFIED = "UNCLASSIFIED"  # legacy alias of REVIEW; kept for compat


@dataclass
class ClassificationResult:
    """Result of known/unknown classification."""
    target_id: str
    classification: ClassificationType
    class_name: str
    model_confidence: float
    anomaly_score: float  # 0-100
    evidence: List[str] = field(default_factory=list)
    context_result: Optional[TargetContextResult] = None
    reasoning: str = ""


# Classes a generic/COCO detector may emit that must never be presented
# as marine-sonar intelligence.
NON_SONAR_CLASSES = {
    "object", "sonar_candidate", "person", "bicycle", "car", "boat",
    "bird", "cat", "dog", "chair", "couch", "tv", "laptop", "cell phone",
}


class AnomalyClassifier:
    """Classifies targets as known objects, unknown anomalies, or natural."""

    def __init__(self, known_classes: Optional[List[str]] = None):
        # Sonar classes recognizable ONLY when custom sonar weights are loaded
        # and explicitly supply these names. Empty by default = honest mode.
        self.known_classes = known_classes or []
        self._anomaly_threshold: float = 60.0
        self._natural_threshold: float = 35.0
        self._confidence_threshold: float = 0.7

    def set_sonar_classes(self, classes: List[str]):
        """Register sonar classes from loaded custom weights."""
        self.known_classes = list(classes)

    def classify(self, target_id: str, detection, fingerprint,
                 context_result: TargetContextResult) -> ClassificationResult:
        """Classify a target as known, unknown-anomaly, or likely-natural."""
        evidence = []
        # MODEL CONFIDENCE: how strongly the detector supports its label.
        # NOT a calibrated probability (no calibration implemented).
        model_confidence = float(detection.confidence)
        # ANOMALY SCORE: how different the region is from local seabed.
        anomaly_score = float(context_result.context_difference.overall_score)

        det_class = str(getattr(detection, "class_name", "sonar_candidate"))
        is_generic = det_class.lower() in NON_SONAR_CLASSES
        is_known_sonar = det_class in self.known_classes and not is_generic

        if is_known_sonar and model_confidence >= self._confidence_threshold:
            classification = ClassificationType.KNOWN_OBJECT
            reasoning = (
                f"Target matched sonar-trained class '{det_class}' with "
                f"detector support {model_confidence:.2f} (uncalibrated) and "
                f"anomaly score {anomaly_score:.1f}."
            )
            evidence.append(f"Sonar-trained class match: {det_class}")
            evidence.append(f"Detector support (uncalibrated): {model_confidence:.2f}")
        elif anomaly_score >= self._anomaly_threshold:
            classification = ClassificationType.UNKNOWN_ANOMALY
            reasoning = (
                f"No confident known-class match for '{det_class}', but anomaly "
                f"score {anomaly_score:.1f} exceeds threshold {self._anomaly_threshold}. "
                f"Treated as unidentified anomaly — NOT forced into a known class."
            )
            evidence.append(f"No known-class match: {det_class}")
            evidence.append(f"Strong anomaly score: {anomaly_score:.1f}")
        elif anomaly_score <= self._natural_threshold:
            classification = ClassificationType.LIKELY_NATURAL
            reasoning = (
                f"Weak anomaly evidence ({anomaly_score:.1f}). Region resembles "
                f"surrounding seabed; likely a natural formation."
            )
            evidence.append(f"Weak anomaly score: {anomaly_score:.1f}")
            evidence.append("Resembles local seabed")
        else:
            classification = ClassificationType.UNCLASSIFIED
            reasoning = (
                f"Ambiguous evidence: detector support {model_confidence:.2f}, "
                f"anomaly score {anomaly_score:.1f}. Requires human review."
            )
            evidence.append(f"Ambiguous anomaly score: {anomaly_score:.1f}")

        if context_result.context_difference.factors:
            evidence.extend(context_result.context_difference.factors)

        evidence.append(f"Geometry: aspect_ratio={fingerprint.geometry.aspect_ratio:.2f}")
        evidence.append(f"Intensity: mean={fingerprint.intensity.mean:.1f}, std={fingerprint.intensity.std:.1f}")
        evidence.append(f"Shadow: present={fingerprint.shadow.shadow_present}")
        evidence.append(f"Context score: {context_result.context_difference.overall_score:.1f}")

        # Display name: never invent a sonar object name for generic candidates
        display_name = det_class if is_known_sonar else (
            "Unidentified anomaly" if classification == ClassificationType.UNKNOWN_ANOMALY
            else ("Rock-like seabed formation" if classification == ClassificationType.LIKELY_NATURAL
                  else det_class)
        )

        return ClassificationResult(
            target_id=target_id,
            classification=classification,
            class_name=display_name,
            model_confidence=model_confidence,
            anomaly_score=anomaly_score,
            evidence=evidence,
            context_result=context_result,
            reasoning=reasoning
        )
