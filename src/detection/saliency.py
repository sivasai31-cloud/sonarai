"""SONARIS-X Sonar Saliency Proposer (classical, DEMO/UNTRAINED).

Honest classical candidate generator for side-scan sonar imagery when no
sonar-trained weights are loaded. It does NOT know object classes — every
candidate is emitted as class "sonar_candidate" with a computed saliency
score (NOT a calibrated probability, NOT a model confidence).

The score is computed from measured evidence:
  saliency = w1 * normalized_local_contrast + w2 * normalized_edge_density
             + w3 * normalized_size_prior
All inputs are measured from the image. No hardcoded detections.
"""

import cv2
import numpy as np
from typing import List

from src.detection.detector import Detection


def _components(mask: np.ndarray, min_area: int, max_area: int):
    """Connected components within [min_area, max_area]."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            out.append((
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
                area,
            ))
    return out


def propose_sonar_candidates(
    img_bgr: np.ndarray,
    min_area: int = 120,
    max_area_fraction: float = 0.25,
) -> List[Detection]:
    """Propose sonar-salient candidate regions using classical CV.

    Steps (all real computation):
    1. CLAHE normalization
    2. Bright-outlier threshold at an adaptive image percentile
       (targets are bright outliers; percentile raised until components
       separate instead of merging into background)
    3. Morphological open (speckle removal) + light close (fragment join)
    4. Connected-component extraction + area filtering
    5. Per-candidate saliency score from measured contrast, edges, size

    Returns List[Detection] with class_name="sonar_candidate".
    """
    if img_bgr is None or img_bgr.size == 0:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
    h, w = gray.shape
    img_area = h * w
    max_area = int(img_area * max_area_fraction)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    # Adaptive percentile: start at p93, raise until components separate
    comps = []
    used_pct = 93.0
    for pct in (93.0, 95.0, 97.0, 98.5):
        thr = float(np.percentile(norm, pct))
        if thr >= 250:
            continue
        _, binary = cv2.threshold(norm, thr, 255, cv2.THRESH_BINARY)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        comps = _components(cleaned, min_area, max_area)
        used_pct = pct
        if comps:
            break
    if not comps:
        return []

    edges = cv2.Canny(norm, 50, 150)
    edge_global = float(np.mean(edges > 0))
    global_std = float(np.std(norm)) + 1e-6

    detections: List[Detection] = []
    for (x, y, bw, bh, area) in comps:
        if bw < 8 or bh < 8:
            continue
        roi = norm[y:y + bh, x:x + bw]
        roi_std = float(np.std(roi))
        contrast_term = min(1.0, roi_std / (global_std * 2.0))
        # Brightness lift of roi vs image median (targets are bright)
        lift = max(0.0, float(np.median(roi)) - float(np.median(norm))) / 255.0
        edge_roi = edges[y:y + bh, x:x + bw]
        edge_density = float(np.mean(edge_roi > 0)) if edge_roi.size else 0.0
        edge_term = min(1.0, edge_density / max(edge_global * 3.0, 1e-6))
        size_frac = area / img_area
        if size_frac < 0.0005:
            size_term = 0.2
        elif size_frac < 0.02:
            size_term = 1.0
        elif size_frac < 0.08:
            size_term = 0.7
        else:
            size_term = 0.4
        saliency = round(
            0.35 * contrast_term + 0.25 * min(1.0, lift * 4.0)
            + 0.25 * edge_term + 0.15 * size_term, 3
        )
        saliency = max(0.05, min(0.95, saliency))
        detections.append(Detection(
            class_id=-1,
            class_name="sonar_candidate",
            confidence=saliency,
            bbox=(int(x), int(y), int(bw), int(bh)),
            center=(float(x + bw / 2), float(y + bh / 2)),
            area=int(bw * bh),
            model_source="SALIENCY/CLASSICAL-DEMO",
        ))
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections[:20]
