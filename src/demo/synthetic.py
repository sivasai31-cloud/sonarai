"""SONARIS-X Synthetic Demo Scenarios (SYNTHETIC DEMO DATA).

Deterministic sonar-like image generators, one per scenario. Each returns
(image_bgr, ground_truth_boxes, scenario_info) where ground_truth_boxes are
the painted target rectangles used ONLY to label the demo scenario panel —
never injected as detections. Detections always come from the real pipeline
(saliency proposer + YOLO if available).

Scenarios: Ghost Net, Pipeline, Cylinder, Shipwreck, Natural Rock / Seabed,
Unknown Anomaly, Low-quality / dropout.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


def _seabed(shape=(600, 800), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(24, 58, (*shape, 3)).astype(np.uint8)
    # sand-ripple texture: horizontal sinusoidal bands
    yy = np.arange(shape[0])[:, None]
    bands = (8 * np.sin(yy / 9.0)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + bands[:, :, None], 0, 255).astype(np.uint8)
    noise = rng.normal(0, 7, (*shape, 3)).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _add_shadow(img: np.ndarray, x: int, y: int, w: int, h: int, dx: int = 10, dy: int = 14):
    sx, sy = x + dx, y + dy
    cv2.rectangle(img, (sx, sy), (sx + w, sy + h), (8, 8, 8), -1)


def _speckle(img: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(y)), 1, (255, 255, 255), -1)


def ghost_net(seed: int = 11) -> Tuple[np.ndarray, List[Dict], Dict]:
    img = _seabed(seed=seed)
    x, y, w, h = 220, 240, 320, 90
    cv2.rectangle(img, (x, y), (x + w, y + h), (190, 190, 190), -1)
    rng = np.random.default_rng(seed)
    for i in range(0, w, 12):  # net mesh
        cv2.line(img, (x + i, y), (x + i, y + h), (120, 120, 120), 1)
    for j in range(0, h, 12):
        cv2.line(img, (x, y + j), (x + w, y + j), (120, 120, 120), 1)
    _add_shadow(img, x, y, w, h)
    _speckle(img, 150, seed)
    gt = [{"bbox": [x, y, w, h], "demo_class": "Ghost Net",
           "note": "Elongated net-like mesh with acoustic shadow (synthetic)."}]
    info = {"scenario": "Ghost Net", "description": "Drifting net snag with mesh texture and shadow."}
    return img, gt, info


def pipeline_scenario(seed: int = 22) -> Tuple[np.ndarray, List[Dict], Dict]:
    img = _seabed(seed=seed)
    x, y, w, h = 60, 280, 680, 34
    cv2.rectangle(img, (x, y), (x + w, y + h), (205, 205, 205), -1)
    cv2.line(img, (x, y + h // 2), (x + w, y + h // 2), (150, 150, 150), 2)
    _add_shadow(img, x, y, w, h, dx=0, dy=16)
    _speckle(img, 150, seed)
    gt = [{"bbox": [x, y, w, h], "demo_class": "Pipeline",
           "note": "Long linear high-contrast feature spanning swath (synthetic)."}]
    info = {"scenario": "Pipeline", "description": "Subsea pipeline crossing the survey swath."}
    return img, gt, info


def cylinder(seed: int = 33) -> Tuple[np.ndarray, List[Dict], Dict]:
    img = _seabed(seed=seed)
    x, y, w, h = 350, 250, 120, 55
    cv2.ellipse(img, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (215, 215, 215), -1)
    cv2.ellipse(img, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (160, 160, 160), 2)
    _add_shadow(img, x, y, w, h)
    _speckle(img, 150, seed)
    gt = [{"bbox": [x, y, w, h], "demo_class": "Cylinder",
           "note": "Compact cylindrical highlight with shadow (synthetic)."}]
    info = {"scenario": "Cylinder", "description": "Possible barrel / cylindrical debris object."}
    return img, gt, info


def shipwreck(seed: int = 44) -> Tuple[np.ndarray, List[Dict], Dict]:
    img = _seabed(seed=seed)
    pts = np.array([[250, 200], [520, 180], [580, 320], [300, 360]], np.int32)
    cv2.fillPoly(img, [pts], (200, 200, 200))
    cv2.polylines(img, [pts], True, (140, 140, 140), 3)
    _add_shadow(img, 250, 180, 330, 180, dx=16, dy=20)
    _speckle(img, 150, seed)
    gt = [{"bbox": [250, 180, 330, 180], "demo_class": "Shipwreck",
           "note": "Large angular wreck-like structure with shadow (synthetic)."}]
    info = {"scenario": "Shipwreck", "description": "Large wreck-like anomaly for triage demo."}
    return img, gt, info


def natural_rock(seed: int = 55) -> Tuple[np.ndarray, List[Dict], Dict]:
    img = _seabed(seed=seed)
    rng = np.random.default_rng(seed)
    gt = []
    for cx, cy, r in [(200, 350, 40), (560, 420, 55), (430, 180, 30)]:
        cv2.circle(img, (cx, cy), r, (95, 95, 95), -1)
        cv2.circle(img, (cx, cy), r, (70, 70, 70), 2)
        gt.append({"bbox": [cx - r, cy - r, 2 * r, 2 * r], "demo_class": "Natural Rock",
                   "note": "Low-contrast rock-like formation, similar to seabed (synthetic)."})
    _speckle(img, 120, seed)
    info = {"scenario": "Natural Rock / Seabed",
            "description": "Natural formations to exercise the false-positive filter."}
    return img, gt, info


def unknown_anomaly(seed: int = 66) -> Tuple[np.ndarray, List[Dict], Dict]:
    img = _seabed(seed=seed)
    pts = np.array([[360, 220], [470, 250], [440, 360], [330, 330]], np.int32)
    cv2.fillPoly(img, [pts], (225, 225, 225))
    cv2.polylines(img, [pts], True, (150, 150, 150), 2)
    _add_shadow(img, 330, 220, 140, 140)
    _speckle(img, 150, seed)
    gt = [{"bbox": [330, 220, 140, 140], "demo_class": "Unknown Anomaly",
           "note": "Irregular object matching no sonar class (synthetic)."}]
    info = {"scenario": "Unknown Anomaly",
            "description": "Deliberately unclassifiable shape for UNKNOWN ANOMALY path."}
    return img, gt, info


def low_quality_dropout(seed: int = 77) -> Tuple[np.ndarray, List[Dict], Dict]:
    img = _seabed(seed=seed)
    x, y, w, h = 300, 230, 130, 70
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), -1)
    _add_shadow(img, x, y, w, h)
    # dropout: uniform dark acquisition gaps
    img[80:200, 100:700] = 6
    img[430:520, 250:650] = 6
    _speckle(img, 80, seed)
    gt = [{"bbox": [x, y, w, h], "demo_class": "Cylinder",
           "note": "Target near dropout gap; expect DATA QUALITY WARNING (synthetic)."}]
    info = {"scenario": "Low-quality / dropout",
            "description": "Acquisition gaps plus a nearby target to test dropout warnings."}
    return img, gt, info


SCENARIOS = {
    "Ghost Net": ghost_net,
    "Pipeline": pipeline_scenario,
    "Cylinder": cylinder,
    "Shipwreck": shipwreck,
    "Natural Rock / Seabed": natural_rock,
    "Unknown Anomaly": unknown_anomaly,
    "Low-quality / dropout": low_quality_dropout,
}


def generate_scenario(name: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Generate a named synthetic demo scenario."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown demo scenario: {name}")
    return SCENARIOS[name]()
