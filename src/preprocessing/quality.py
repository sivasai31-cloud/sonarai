"""SONARIS-X Image Quality Assessment Module.

Calculates actual image quality metrics including:
- resolution
- contrast
- noise level
- dynamic range
- dropout regions
- image statistics

Outputs quality score and rating: GOOD / ACCEPTABLE / DEGRADED / REVIEW REQUIRED
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from scipy import stats as scipy_stats


@dataclass
class QualityAssessment:
    """Result of image quality assessment."""
    quality_score: float  # 0-100
    rating: str  # GOOD, ACCEPTABLE, DEGRADED, REVIEW REQUIRED
    resolution_score: float
    contrast_score: float
    noise_score: float
    dynamic_range_score: float
    dropout_score: float
    statistics: Dict[str, float] = field(default_factory=dict)
    dropout_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ImageQualityAssessor:
    """Calculates actual image quality metrics from sonar imagery."""

    def __init__(self):
        self._last_assessment: Optional[QualityAssessment] = None

    def calculate_resolution_score(self, img: np.ndarray) -> float:
        """Calculate resolution score based on image dimensions and edge sharpness."""
        h, w = img.shape[:2]
        # Base resolution score from dimensions
        total_pixels = h * w
        if total_pixels == 0:
            return 0.0

        # Resolution component
        if total_pixels > 2000000:
            res_score = 100.0
        elif total_pixels > 1000000:
            res_score = 80.0
        elif total_pixels > 500000:
            res_score = 60.0
        elif total_pixels > 200000:
            res_score = 40.0
        else:
            res_score = 20.0

        # Sharpness via Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 500:
            sharpness_bonus = 20.0
        elif laplacian_var > 200:
            sharpness_bonus = 10.0
        elif laplacian_var > 50:
            sharpness_bonus = 5.0
        else:
            sharpness_bonus = 0.0

        return min(100.0, res_score + sharpness_bonus)

    def calculate_contrast_score(self, img: np.ndarray) -> float:
        """Calculate contrast score using standard deviation and histogram analysis."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Standard deviation as contrast measure
        std = np.std(gray)
        if std > 80:
            contrast_score = 100.0
        elif std > 50:
            contrast_score = 80.0
        elif std > 30:
            contrast_score = 60.0
        elif std > 15:
            contrast_score = 40.0
        else:
            contrast_score = 20.0

        # Also check histogram spread
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist[hist > 0]  # Only bins with pixels
        if len(hist) > 0:
            # How many bins are used
            bin_coverage = len(hist) / 256.0
            contrast_score = min(100.0, contrast_score * 0.5 + bin_coverage * 100.0 * 0.5)

        return contrast_score

    def calculate_noise_score(self, img: np.ndarray) -> float:
        """Calculate noise level (higher score = less noise = better quality)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Estimate noise using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_est = np.var(laplacian)

        # Noise is inversely related to Laplacian variance
        if noise_est > 500:
            noise_score = 100.0
        elif noise_est > 200:
            noise_score = 80.0
        elif noise_est > 100:
            noise_score = 60.0
        elif noise_est > 50:
            noise_score = 40.0
        else:
            noise_score = 20.0

        return noise_score

    def calculate_dynamic_range_score(self, img: np.ndarray) -> float:
        """Calculate dynamic range based on histogram analysis."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        total = hist.sum()
        if total == 0:
            return 0.0

        hist = hist / total  # Normalize

        # Find non-zero bins
        non_zero = np.where(hist > 0.001)[0]
        if len(non_zero) == 0:
            return 0.0

        range_span = non_zero[-1] - non_zero[0]
        dr_score = (range_span / 255.0) * 100.0

        # Check for clipping
        dark_clip = hist[:10].sum()
        bright_clip = hist[-10:].sum()
        if dark_clip > 0.1 or bright_clip > 0.1:
            dr_score *= 0.7  # Penalty for clipping

        return dr_score

    def detect_dropout_regions(self, img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """Detect dropout regions (uniform areas indicating data loss)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Use adaptive threshold to find uniform regions
        h, w = gray.shape
        block_size = 32
        dropouts = []
        dropout_count = 0

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                std = np.std(block)
                if std < 2.0:  # Very uniform region = potential dropout
                    dropouts.append((x, y, block_size, block_size))
                    dropout_count += 1

        dropout_ratio = dropout_count / max(1, (h * w) / (block_size * block_size))
        return dropouts, dropout_ratio

    def calculate_statistics(self, img: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive image statistics."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        stats = {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'median': float(np.median(gray)),
            'min': float(np.min(gray)),
            'max': float(np.max(gray)),
            'percentile_10': float(np.percentile(gray, 10)),
            'percentile_25': float(np.percentile(gray, 25)),
            'percentile_75': float(np.percentile(gray, 75)),
            'percentile_90': float(np.percentile(gray, 90)),
            'skewness': float(scipy_stats.skew(gray.flatten())),
            'kurtosis': float(scipy_stats.kurtosis(gray.flatten())),
        }
        return stats

    def assess(self, img: np.ndarray) -> QualityAssessment:
        """Perform full quality assessment on an image."""
        if img is None or img.size == 0:
            raise ValueError("Cannot assess null or empty image")

        # Convert to BGR if grayscale for consistency
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        resolution_score = self.calculate_resolution_score(img)
        contrast_score = self.calculate_contrast_score(img)
        noise_score = self.calculate_noise_score(img)
        dynamic_range_score = self.calculate_dynamic_range_score(img)
        dropout_regions, dropout_ratio = self.detect_dropout_regions(img)

        # Weighted quality score
        quality_score = (
            resolution_score * 0.15 +
            contrast_score * 0.25 +
            noise_score * 0.25 +
            dynamic_range_score * 0.20 +
            (1.0 - min(dropout_ratio, 0.5)) * 100.0 * 0.15
        )

        quality_score = max(0.0, min(100.0, quality_score))

        # Determine rating
        if quality_score >= 80:
            rating = "GOOD"
        elif quality_score >= 60:
            rating = "ACCEPTABLE"
        elif quality_score >= 40:
            rating = "DEGRADED"
        else:
            rating = "REVIEW REQUIRED"

        stats = self.calculate_statistics(img)

        warnings = []
        if dropout_ratio > 0.1:
            warnings.append(f"Dropout regions detected: {dropout_ratio:.1%} of image")
        if contrast_score < 40:
            warnings.append("Low contrast detected")
        if noise_score < 40:
            warnings.append("High noise level detected")

        assessment = QualityAssessment(
            quality_score=quality_score,
            rating=rating,
            resolution_score=resolution_score,
            contrast_score=contrast_score,
            noise_score=noise_score,
            dynamic_range_score=dynamic_range_score,
            dropout_score=(1.0 - min(dropout_ratio, 0.5)) * 100.0,
            statistics=stats,
            dropout_regions=dropout_regions,
            warnings=warnings
        )

        self._last_assessment = assessment
        return assessment
