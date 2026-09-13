"""SONARIS-X Multi-Scale Detection Module.

Implements object detection using YOLOv8 with support for:
- Full-image inference
- Sliding-window/tiled inference
- Overlapping tiles
- Confidence threshold
- NMS merging

Architecture supports custom sonar-trained weights.
"""

import os
import json
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

logger = logging.getLogger(__name__)
if YOLO is None:
    logger.warning("ultralytics not installed. Detector will run in DEMO mode.")


@dataclass
class Detection:
    """A single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]
    area: int
    tile_id: Optional[int] = None
    model_source: str = "default"


@dataclass
class DetectionResult:
    """Result of a detection run."""
    detections: List[Detection]
    inference_time_ms: float
    mode: str  # "full" or "tiled"
    num_tiles: int = 0
    tile_overlap: int = 0
    model_path: str = ""
    model_name: str = ""
    warnings: List[str] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)


@dataclass
class TileConfig:
    """Configuration for tiled inference."""
    tile_size: int = 512
    overlap: int = 64
    stride: Optional[int] = None


class DetectionError(Exception):
    """Custom exception for detection errors."""
    pass


# Sonar object taxonomy. These names are ONLY advertised as recognizable
# when custom sonar-trained weights are loaded (see sonar_classes()).
# The COCO fallback and classical saliency proposer NEVER claim these.
SONAR_CLASS_TAXONOMY = [
    "Ghost Net",
    "Pipeline",
    "Cylinder",
    "Shipwreck",
    "Debris",
    "Natural Rock",
]


class MultiScaleDetector:
    """Multi-scale object detector using YOLOv8."""

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[YOLO] = None
        self.model_path = model_path or ""
        self.model_name = "DEMO/UNTRAINED"
        self._load_model(model_path)

    def model_status(self) -> str:
        """Honest model status string for UI display."""
        if self.model is None:
            return "DEMO / UNTRAINED — CUSTOM WEIGHTS NOT LOADED"
        if self.model_path and os.path.exists(self.model_path):
            return "CUSTOM SONAR WEIGHTS LOADED"
        return "COCO FALLBACK — NOT SONAR-TRAINED"

    def sonar_classes(self) -> List[str]:
        """Sonar classes the loaded model can actually recognize.

        Only custom sonar-trained weights advertise the sonar taxonomy.
        COCO fallback and demo modes return [] so the UI never shows
        sonar object names as model predictions.
        """
        if self.model is not None and self.model_path and os.path.exists(self.model_path):
            try:
                names = list(self.model.names.values()) if hasattr(self.model, "names") else []
                # Advertise intersection with sonar taxonomy only
                known = [n for n in names if n in SONAR_CLASS_TAXONOMY]
                return known
            except Exception:
                return []
        return []

    def _load_model(self, model_path: Optional[str]):
        """Load YOLO model with fallback. Never crashes offline — falls back to DEMO mode."""
        if YOLO is None:
            logger.warning("ultralytics unavailable. Running in DEMO mode (synthetic detections, labeled DEMO/UNTRAINED).")
            self.model = None
            self.model_name = "NO MODEL - DEMO MODE"
            return
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.model_path = model_path
                self.model_name = os.path.basename(model_path)
                logger.info(f"Loaded model from {model_path}")
            else:
                # Fallback to YOLOv8n pre-trained (may require download; offline → DEMO mode)
                # NOTE: COCO weights are NOT sonar-trained. UI must show COCO FALLBACK status.
                self.model = YOLO('yolov8n.pt')
                self.model_name = "yolov8n.pt (COCO fallback — NOT sonar-trained)"
                logger.warning("No custom sonar model found. Using COCO YOLOv8n fallback (NOT sonar-trained).")
        except Exception as e:
            logger.error(f"Failed to load model (offline or missing weights, using DEMO mode): {e}")
            # Create a minimal fallback - will use demo detection mode
            self.model = None
            self.model_name = "NO MODEL - DEMO MODE [SIMULATED]"

    def _run_full_inference(self, img: np.ndarray, conf_threshold: float) -> List[Detection]:
        """Run full-image inference."""
        if self.model is None:
            return self._demo_detections(img, conf_threshold)

        try:
            results = self.model(img, conf=conf_threshold, verbose=False)
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    w, h = x2 - x1, y2 - y1
                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, w, h),
                        center=(x1 + w/2, y1 + h/2),
                        area=w * h,
                        model_source=self.model_name
                    ))
            return detections
        except Exception as e:
            logger.error(f"Full inference error: {e}")
            return self._demo_detections(img, conf_threshold)

    def _run_tiled_inference(self, img: np.ndarray, conf_threshold: float,
                              tile_config: TileConfig) -> List[Detection]:
        """Run tiled inference with overlapping tiles."""
        if self.model is None:
            return self._demo_detections(img, conf_threshold)

        h, w = img.shape[:2]
        tile_size = tile_config.tile_size
        overlap = tile_config.overlap
        stride = tile_config.stride or (tile_size - overlap)

        all_detections = []
        tile_id = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = img[y:y+tile_size, x:x+tile_size]
                if tile.size == 0:
                    continue

                tile_dets = self._run_full_inference(tile, conf_threshold)
                for det in tile_dets:
                    det.tile_id = tile_id
                    # Adjust bbox coordinates to full image space
                    bx, by, bw, bh = det.bbox
                    det.bbox = (x + bx, y + by, bw, bh)
                    det.center = (x + bx + bw/2, y + by + bh/2)
                    all_detections.append(det)

                tile_id += 1

        # Handle edge tiles
        if h % stride != 0 or w % stride != 0:
            y = max(0, h - tile_size)
            x = max(0, w - tile_size)
            if y < h - tile_size or x < w - tile_size:
                tile = img[y:y+tile_size, x:x+tile_size]
                if tile.size > 0:
                    tile_dets = self._run_full_inference(tile, conf_threshold)
                    for det in tile_dets:
                        det.tile_id = tile_id
                        bx, by, bw, bh = det.bbox
                        det.bbox = (x + bx, y + by, bw, bh)
                        det.center = (x + bx + bw/2, y + by + bh/2)
                        all_detections.append(det)
                    tile_id += 1

        logger.info(f"Tiled inference: {tile_id} tiles processed")
        return all_detections

    def _merge_detections(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Merge overlapping detections from tiles using NMS."""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Group by class and apply NMS
        kept = []
        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        for cls_name, cls_dets in by_class.items():
            while cls_dets:
                best = cls_dets.pop(0)
                kept.append(best)
                cls_dets = [
                    d for d in cls_dets
                    if self._calculate_iou(best.bbox, d.bbox) < iou_threshold
                ]

        return kept

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)

    def _demo_detections(self, img: np.ndarray, conf_threshold: float) -> List[Detection]:
        """Generate synthetic detections for demo mode."""
        h, w = img.shape[:2]
        # Create a few synthetic detections at random positions
        np.random.seed(42)
        detections = []
        num_detections = np.random.randint(2, 6)
        for i in range(num_detections):
            x = np.random.randint(10, max(10, w - 100))
            y = np.random.randint(10, max(10, h - 100))
            bw = np.random.randint(20, 80)
            bh = np.random.randint(20, 80)
            conf = np.random.uniform(conf_threshold, 0.95)
            detections.append(Detection(
                class_id=0,
                class_name="object",
                confidence=conf,
                bbox=(x, y, bw, bh),
                center=(x + bw/2, y + bh/2),
                area=bw * bh,
                model_source="DEMO/UNTRAINED"
            ))
        return detections

    def detect(self, img: np.ndarray, conf_threshold: float = 0.5,
               use_tiled: bool = False,
               tile_config: Optional[TileConfig] = None,
               use_saliency: bool = True) -> DetectionResult:
        """Run multi-scale detection on an image.

        Combines (a) YOLO inference when weights are available and
        (b) the classical sonar saliency proposer (DEMO/UNTRAINED).
        Saliency candidates are labeled "sonar_candidate" with a computed
        saliency score — never a sonar object name, never a calibrated
        probability.
        """
        if img is None or img.size == 0:
            raise ValueError("Cannot detect on null or empty image")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        tile_config = tile_config or TileConfig()
        start_time = time.time()

        if use_tiled:
            detections = self._run_tiled_inference(img, conf_threshold, tile_config)
            merged = self._merge_detections(detections)
            mode = "tiled"
            h, w = img.shape[:2]
            stride = tile_config.stride or (tile_config.tile_size - tile_config.overlap)
            stride = max(1, stride)
            nx = max(1, (max(0, w - tile_config.tile_size) + stride - 1) // stride + 1)
            ny = max(1, (max(0, h - tile_config.tile_size) + stride - 1) // stride + 1)
            num_tiles = int(nx * ny)
        else:
            detections = self._run_full_inference(img, conf_threshold)
            merged = self._merge_detections(detections)
            mode = "full"
            num_tiles = 0

        warnings: List[str] = []
        # Classical saliency proposer: honest candidate source when no
        # sonar-trained weights exist. Deferred import avoids circulars.
        if use_saliency:
            try:
                from src.detection.saliency import propose_sonar_candidates
                saliency_dets = propose_sonar_candidates(img)
                saliency_dets = [d for d in saliency_dets if d.confidence >= conf_threshold * 0.6]
                if saliency_dets:
                    merged = self._merge_detections(merged + saliency_dets)
                    warnings.append(
                        f"Saliency proposer added {len(saliency_dets)} classical candidates "
                        f"(DEMO/UNTRAINED, class=sonar_candidate)."
                    )
            except Exception as e:
                warnings.append(f"Saliency proposer unavailable: {e}")

        if self.model is None:
            warnings.append("MODEL STATUS: DEMO / UNTRAINED — CUSTOM WEIGHTS NOT LOADED")
        elif not (self.model_path and os.path.exists(self.model_path)):
            warnings.append("MODEL STATUS: COCO FALLBACK — NOT SONAR-TRAINED")

        inference_time = (time.time() - start_time) * 1000

        result = DetectionResult(
            detections=merged,
            inference_time_ms=inference_time,
            mode=mode,
            num_tiles=num_tiles,
            tile_overlap=tile_config.overlap,
            model_path=self.model_path,
            model_name=self.model_name,
            warnings=warnings,
            image_shape=img.shape[:2]
        )

        logger.info(f"Detection complete: {len(merged)} detections in {inference_time:.1f}ms ({mode} mode)")
        return result
