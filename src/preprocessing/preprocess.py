"""SONARIS-X Sonar Preprocessing Module.

Implements a preprocessing pipeline containing:
1. Grayscale normalization
2. CLAHE contrast enhancement
3. Noise/speckle reduction
4. Dropout detection
5. Optional enhancement controls

All operations actually affect the image.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    apply_denoising: bool = True
    denoise_strength: float = 10.0
    apply_normalization: bool = True
    apply_bilateral_filter: bool = False
    bilateral_diameter: int = 9
    bilateral_sigma: float = 75.0
    dropout_detection_threshold: float = 2.0
    enhance_shadows: bool = False


@dataclass
class PreprocessingResult:
    """Result of preprocessing operations."""
    original: np.ndarray
    preprocessed: np.ndarray
    detection_view: np.ndarray
    processing_log: List[str] = field(default_factory=list)
    dropout_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    enhancement_applied: Dict[str, bool] = field(default_factory=dict)
    processing_time_ms: float = 0.0


class SonarPreprocessor:
    """Preprocessing pipeline for side-scan sonar imagery."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._last_result: Optional[PreprocessingResult] = None
        self._log: List[str] = []

    def _add_log(self, msg: str):
        """Internal logging helper."""
        self._log.append(msg)

    def grayscale_normalization(self, img: np.ndarray) -> np.ndarray:
        """Normalize grayscale values using histogram equalization."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid
            )
            normalized = clahe.apply(gray)
            self._add_log(f"Applied CLAHE (clip={self.config.clahe_clip_limit}, grid={self.config.clahe_tile_grid})")
        else:
            # Simple histogram stretching
            p2, p98 = np.percentile(gray, (2, 98))
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self._add_log("Applied histogram stretching")

        return normalized

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply noise/speckle reduction."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if self.config.apply_denoising:
            # Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoising(
                gray, None,
                h=self.config.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
            self._add_log(f"Applied Non-Local Means denoising (strength={self.config.denoise_strength})")
        else:
            denoised = gray.copy()

        # Optional bilateral filter
        if self.config.apply_bilateral_filter:
            denoised = cv2.bilateralFilter(
                denoised,
                self.config.bilateral_diameter,
                self.config.bilateral_sigma,
                self.config.bilateral_sigma
            )
            self._add_log(f"Applied bilateral filter (d={self.config.bilateral_diameter})")

        return denoised

    def detect_dropouts(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Detect dropout regions in the image."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape
        block_size = 32
        dropouts = []
        dropout_mask = np.zeros_like(gray)

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                std = np.std(block)
                if std < self.config.dropout_detection_threshold:
                    dropouts.append((x, y, block_size, block_size))
                    dropout_mask[y:y+block_size, x:x+block_size] = 255

        self._add_log(f"Detected {len(dropouts)} dropout regions")
        return dropout_mask, dropouts

    def enhance_shadows(self, img: np.ndarray) -> np.ndarray:
        """Enhance shadow regions in sonar imagery."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Adaptive threshold for shadow enhancement
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Blend with original
        result = cv2.addWeighted(gray, 0.7, enhanced, 0.3, 0)
        self._add_log("Applied shadow enhancement")
        return result

    def create_detection_view(self, img: np.ndarray, detections: list = None) -> np.ndarray:
        """Create a visualization view with detections overlaid."""
        if len(img.shape) == 3:
            view = img.copy()
        else:
            view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if detections:
            for det in detections:
                if 'bbox' in det:
                    x, y, w, h = det['bbox']
                    cv2.rectangle(view, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = det.get('class', 'UNKNOWN')
                    conf = det.get('confidence', 0.0)
                    cv2.putText(view, f"{label} {conf:.2f}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self._add_log(f"Created detection view with {len(detections or [])} detections")
        return view

    def process(self, img: np.ndarray, detections: list = None) -> PreprocessingResult:
        """Run the full preprocessing pipeline."""
        import time
        start_time = time.time()

        self._log = []
        self._add_log("Starting preprocessing pipeline")

        # Step 1: Grayscale normalization
        normalized = self.grayscale_normalization(img)
        self._add_log("Grayscale normalization complete")

        # Step 2: Denoising
        denoised = self.denoise(normalized)
        self._add_log("Denoising complete")

        # Step 3: Dropout detection
        dropout_mask, dropouts = self.detect_dropouts(denoised)
        self._add_log("Dropout detection complete")

        # Step 4: Optional shadow enhancement
        preprocessed = denoised.copy()
        if self.config.enhance_shadows:
            preprocessed = self.enhance_shadows(denoised)

        # Step 5: Create detection view
        detection_view = self.create_detection_view(preprocessed, detections)

        elapsed = (time.time() - start_time) * 1000

        result = PreprocessingResult(
            original=img.copy(),
            preprocessed=preprocessed,
            detection_view=detection_view,
            dropout_regions=dropouts,
            processing_log=self._log.copy(),
            enhancement_applied={
                'clahe': self.config.apply_clahe,
                'denoising': self.config.apply_denoising,
                'bilateral': self.config.apply_bilateral_filter,
                'shadows': self.config.enhance_shadows,
            },
            processing_time_ms=elapsed
        )

        self._last_result = result
        return result
