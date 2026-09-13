"""SONARIS-X dataset adapters (common internal format + honest registry).

Common internal record:
    {"image_path": str, "width": int, "height": int,
     "boxes": [{"bbox": [x,y,w,h], "class": str}], "group": str,
     "source": str, "license": str}

Adapters convert external layouts into these records. No dataset facts are
invented: every registry entry documents source / license / classes /
annotation format / environment / limitations, and external datasets require
manual download (licenses forbid redistribution).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2


def to_record(image_path: str, boxes: Optional[List[Dict]] = None,
              group: str = "default", source: str = "",
              license: str = "") -> Dict:
    """Build one internal record, reading dimensions from disk."""
    img = cv2.imread(image_path)
    h, w = (img.shape[:2] if img is not None else (0, 0))
    return {"image_path": image_path, "width": w, "height": h,
            "boxes": boxes or [], "group": group,
            "source": source, "license": license}


class YoloAdapter:
    """YOLO-format dataset: images/ + labels/*.txt + data.yaml (manual setup).

    Expected layout:
        <root>/images/*.png|jpg + <root>/labels/*.txt + <root>/data.yaml
    Label lines: <class_id> <cx> <cy> <w> <h> (normalized).
    group = relative parent dir (mission/survey) for leakage-safe splits.
    """

    def __init__(self, root: str, source: str = "user-supplied",
                 license: str = "see dataset provider"):
        self.root = root
        self.source = source
        self.license = license

    def records(self) -> List[Dict]:
        import yaml
        names: Dict[int, str] = {}
        yaml_path = os.path.join(self.root, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                names = {int(k): str(v) for k, v in (yaml.safe_load(f).get("names", {}) or {}).items()}
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        out = []
        for img_path in sorted(Path(self.root, "images").rglob("*")):
            if img_path.suffix.lower() not in exts:
                continue
            label = Path(self.root, "labels", img_path.stem + ".txt")
            boxes = []
            if label.exists():
                img = cv2.imread(str(img_path))
                h, w = (img.shape[:2] if img is not None else (0, 0))
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cid, cx, cy, bw, bh = float(parts[0]), *[float(v) for v in parts[1:]]
                    boxes.append({"bbox": [int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                                           int(bw * w), int(bh * h)],
                                  "class": names.get(int(cid), f"class_{int(cid)}")})
            out.append(to_record(str(img_path), boxes, group=img_path.parent.name,
                                 source=self.source, license=self.license))
        return out


class ManifestAdapter:
    """Built-in synthetic demo manifest (data/demo/manifest.json)."""

    def __init__(self, manifest: str = "data/demo/manifest.json"):
        self.manifest = manifest

    def records(self) -> List[Dict]:
        if not os.path.exists(self.manifest):
            return []
        with open(self.manifest) as f:
            entries = json.load(f)
        out = []
        for e in entries:
            boxes = [{"bbox": g["bbox"], "class": g["demo_class"]}
                     for g in e.get("ground_truth", [])]
            out.append(to_record(e["file"], boxes, group=e.get("scenario", "demo"),
                                 source="SONARIS-X synthetic generator",
                                 license="built-in demo, free to use"))
        return out


# Honest registry: only what ships or is manually installed is listed.
# External public sonar datasets are NOT auto-downloaded; add an entry here
# (source/license/classes/format/environment/limitations) after manual setup.
DATASET_REGISTRY: List[Dict] = [
    {"name": "sonarisx-synthetic-demo", "source": "built-in generator (src/demo/synthetic.py)",
     "license": "built-in demo, free to use",
     "classes": ["Ghost Net", "Pipeline", "Cylinder", "Shipwreck", "Natural Rock",
                 "Unknown Anomaly (demo ground truth labels, not model predictions)"],
     "annotation_format": "manifest.json (synthetic ground truth)",
     "environment": "fully synthetic seabed + painted targets",
     "limitations": "Not real sonar; validates software pipeline only, not detection accuracy.",
     "setup": "Analysis Console → Dataset Explorer → GENERATE DEMO SET"},
]
