"""SONARIS-X Acoustic/Image Fingerprint Module.

For every candidate detection, calculates a measurable fingerprint including:
- GEOMETRY: area, width, height, aspect ratio
- INTENSITY: mean, std, percentiles, local contrast
- TEXTURE: LBP, GLCM-derived statistics
- EDGES: edge density
- SHADOW: shadow presence, characteristics
- SEABED CONTEXT: local seabed statistics, target/seabed divergence
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


@dataclass
class GeometryFingerprint:
    """Geometric properties of a target."""
    area: int
    width: int
    height: int
    aspect_ratio: float
    perimeter: int
    solidity: float = 0.0
    extent: float = 0.0


@dataclass
class IntensityFingerprint:
    """Intensity statistics of a target."""
    mean: float
    std: float
    median: float
    min: float
    max: float
    percentile_10: float
    percentile_50: float
    percentile_90: float
    local_contrast: float
    dynamic_range: float


@dataclass
class TextureFingerprint:
    """Texture statistics using LBP and GLCM."""
    lbp_mean: float
    lbp_std: float
    lbp_entropy: float
    glcm_contrast: float
    glcm_correlation: float
    glcm_energy: float
    glcm_homogeneity: float
    glcm_entropy: float


@dataclass
class EdgeFingerprint:
    """Edge characteristics."""
    edge_density: float
    edge_count: int
    edge_direction_variance: float
    edge_strongest_points: int


@dataclass
class ShadowFingerprint:
    """Shadow characteristics."""
    shadow_present: bool
    shadow_area: float
    shadow_intensity: float
    shadow_aspect_ratio: float
    shadow_distance_from_target: float


@dataclass
class SeabedContextFingerprint:
    """Local seabed statistics and divergence."""
    seabed_mean: float
    seabed_std: float
    seabed_contrast: float
    seabed_texture: Dict[str, float]
    target_seabed_divergence: Dict[str, float]


@dataclass
class TargetFingerprint:
    """Complete fingerprint for a target detection."""
    geometry: GeometryFingerprint
    intensity: IntensityFingerprint
    texture: TextureFingerprint
    edges: EdgeFingerprint
    shadow: ShadowFingerprint
    seabed_context: SeabedContextFingerprint
    fingerprint_id: str = ""
    extraction_time_ms: float = 0.0


class FingerprintExtractor:
    """Extracts measurable fingerprints for target detections."""

    def __init__(self):
        self._last_fingerprint: Optional[TargetFingerprint] = None

    def extract_geometry(self, target_region: np.ndarray, bbox: Tuple[int, int, int, int]) -> GeometryFingerprint:
        """Extract geometric properties."""
        h, w = target_region.shape[:2]
        area = h * w
        aspect_ratio = max(w, h) / max(min(w, h), 1)

        # Find contours for perimeter and solidity
        gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY) if len(target_region.shape) == 3 else target_region
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perimeter = 0
        solidity = 0.0
        extent = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest, True)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = cv2.contourArea(largest) / hull_area
            x, y, bw, bh = cv2.boundingRect(largest)
            rect_area = bw * bh
            if rect_area > 0:
                extent = cv2.contourArea(largest) / rect_area

        return GeometryFingerprint(
            area=area, width=w, height=h,
            aspect_ratio=aspect_ratio, perimeter=int(perimeter),
            solidity=solidity, extent=extent
        )

    def extract_intensity(self, target_region: np.ndarray) -> IntensityFingerprint:
        """Extract intensity statistics."""
        gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY) if len(target_region.shape) == 3 else target_region

        mean = float(np.mean(gray))
        std = float(np.std(gray))
        median = float(np.median(gray))
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        p10 = float(np.percentile(gray, 10))
        p50 = float(np.percentile(gray, 50))
        p90 = float(np.percentile(gray, 90))

        # Local contrast (std of local regions)
        local_contrast = std

        # Dynamic range
        dynamic_range = max_val - min_val

        return IntensityFingerprint(
            mean=mean, std=std, median=median,
            min=min_val, max=max_val,
            percentile_10=p10, percentile_50=p50, percentile_90=p90,
            local_contrast=local_contrast, dynamic_range=dynamic_range
        )

    def extract_texture(self, target_region: np.ndarray) -> TextureFingerprint:
        """Extract texture statistics using LBP and GLCM."""
        gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY) if len(target_region.shape) == 3 else target_region

        # LBP
        lbp_radius = 1
        lbp_points = 8 * lbp_radius
        lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, lbp_points + 3), range=(0, lbp_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        lbp_mean = float(np.mean(lbp))
        lbp_std = float(np.std(lbp))
        lbp_entropy = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)))

        # GLCM
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        glcm_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
        glcm_correlation = float(graycoprops(glcm, 'correlation')[0, 0])
        glcm_energy = float(graycoprops(glcm, 'energy')[0, 0])
        glcm_homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
        glcm_entropy = float(-np.sum(glcm * np.log2(glcm + 1e-10)))

        return TextureFingerprint(
            lbp_mean=lbp_mean, lbp_std=lbp_std, lbp_entropy=lbp_entropy,
            glcm_contrast=glcm_contrast, glcm_correlation=glcm_correlation,
            glcm_energy=glcm_energy, glcm_homogeneity=glcm_homogeneity,
            glcm_entropy=glcm_entropy
        )

    def extract_edges(self, target_region: np.ndarray) -> EdgeFingerprint:
        """Extract edge characteristics."""
        gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY) if len(target_region.shape) == 3 else target_region

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        edge_count = int(np.sum(edges > 0))

        # Edge direction analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        direction = np.arctan2(sobely, sobelx)
        direction_variance = float(np.var(direction))

        return EdgeFingerprint(
            edge_density=edge_density,
            edge_count=edge_count,
            edge_direction_variance=direction_variance,
            edge_strongest_points=int(np.sum(edges > 127))
        )

    def extract_shadow(self, target_region: np.ndarray) -> ShadowFingerprint:
        """Extract shadow characteristics."""
        gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY) if len(target_region.shape) == 3 else target_region

        # Shadow detection: dark regions below/behind the target
        threshold = np.percentile(gray, 20)
        shadow_mask = gray < threshold
        shadow_present = shadow_mask.sum() > gray.size * 0.01

        if shadow_present:
            shadow_area = float(shadow_mask.sum() / gray.size)
            shadow_intensity = float(np.mean(gray[shadow_mask]))
            ys, xs = np.where(shadow_mask)
            if len(xs) > 1 and len(ys) > 1:
                shadow_w = xs.max() - xs.min()
                shadow_h = ys.max() - ys.min()
                shadow_aspect = max(shadow_w, shadow_h) / max(min(shadow_w, shadow_h), 1)
                # Distance from target center
                target_center = (target_region.shape[1] / 2, target_region.shape[0] / 2)
                shadow_center = (xs.mean(), ys.mean())
                shadow_dist = np.sqrt((shadow_center[0] - target_center[0])**2 +
                                      (shadow_center[1] - target_center[1])**2)
            else:
                shadow_aspect = 1.0
                shadow_dist = 0.0
        else:
            shadow_area = 0.0
            shadow_intensity = 0.0
            shadow_aspect = 0.0
            shadow_dist = 0.0

        return ShadowFingerprint(
            shadow_present=shadow_present,
            shadow_area=shadow_area,
            shadow_intensity=shadow_intensity,
            shadow_aspect_ratio=shadow_aspect,
            shadow_distance_from_target=shadow_dist
        )

    def extract_seabed_context(self, target_region: np.ndarray,
                                surround_region: np.ndarray) -> SeabedContextFingerprint:
        """Extract local seabed statistics and target/seabed divergence."""
        target_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY) if len(target_region.shape) == 3 else target_region
        seabed_gray = cv2.cvtColor(surround_region, cv2.COLOR_BGR2GRAY) if len(surround_region.shape) == 3 else surround_region

        # Seabed statistics
        seabed_mean = float(np.mean(seabed_gray))
        seabed_std = float(np.std(seabed_gray))
        seabed_contrast = seabed_std

        # Seabed texture (GLCM)
        try:
            seabed_glcm = graycomatrix(seabed_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            seabed_texture = {
                'contrast': float(graycoprops(seabed_glcm, 'contrast')[0, 0]),
                'correlation': float(graycoprops(seabed_glcm, 'correlation')[0, 0]),
                'energy': float(graycoprops(seabed_glcm, 'energy')[0, 0]),
                'homogeneity': float(graycoprops(seabed_glcm, 'homogeneity')[0, 0])
            }
        except Exception:
            seabed_texture = {'contrast': 0, 'correlation': 0, 'energy': 0, 'homogeneity': 0}

        # Target-seabed divergence
        divergence = {
            'intensity_diff': abs(float(np.mean(target_gray)) - seabed_mean),
            'contrast_diff': abs(float(np.std(target_gray)) - seabed_std),
            'texture_diff': abs(
                float(graycoprops(
                    graycomatrix(target_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True),
                    'contrast'
                )[0, 0]) - seabed_texture['contrast']
            ) if seabed_texture['contrast'] > 0 else 0.0
        }

        return SeabedContextFingerprint(
            seabed_mean=seabed_mean, seabed_std=seabed_std, seabed_contrast=seabed_contrast,
            seabed_texture=seabed_texture, target_seabed_divergence=divergence
        )

    def extract(self, target_region: np.ndarray, surround_region: np.ndarray,
                bbox: Tuple[int, int, int, int]) -> TargetFingerprint:
        """Extract complete fingerprint for a target."""
        import time
        start = time.time()

        geometry = self.extract_geometry(target_region, bbox)
        intensity = self.extract_intensity(target_region)
        texture = self.extract_texture(target_region)
        edges = self.extract_edges(target_region)
        shadow = self.extract_shadow(target_region)
        seabed = self.extract_seabed_context(target_region, surround_region)

        elapsed = (time.time() - start) * 1000

        fingerprint = TargetFingerprint(
            geometry=geometry,
            intensity=intensity,
            texture=texture,
            edges=edges,
            shadow=shadow,
            seabed_context=seabed,
            extraction_time_ms=elapsed
        )

        self._last_fingerprint = fingerprint
        return fingerprint
