"""SONARIS-X domain-generalization probe (measured, no adaptation claimed).

Treats each demo scenario (or each subdirectory of --dir) as a domain and
measures in-domain vs cross-domain candidate counts and support stability.

Usage:
    python scripts/domain.py [--dir data/demo]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.detection.detector import MultiScaleDetector
from src.demo.synthetic import generate_scenario, SCENARIOS


def main():
    ap = argparse.ArgumentParser(description="Domain-generalization measurement probe.")
    ap.add_argument("--dir", default=None, help="Optional image dir (subdirs = domains)")
    args = ap.parse_args()

    detector = MultiScaleDetector()
    rows = []
    if args.dir and os.path.isdir(args.dir):
        import cv2
        from pathlib import Path
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        for sub in sorted(Path(args.dir).iterdir()):
            if not sub.is_dir():
                continue
            files = [p for p in sub.iterdir() if p.suffix.lower() in exts]
            counts, supports = [], []
            for f in files[:20]:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                r = detector.detect(img)
                counts.append(len(r.detections))
                supports.extend(d.confidence for d in r.detections)
            rows.append({"domain": sub.name, "images": len(counts),
                         "mean_candidates": round(float(np.mean(counts)), 2) if counts else 0.0,
                         "mean_support": round(float(np.mean(supports)), 3) if supports else 0.0})
    else:
        for name in SCENARIOS:
            img, _, _ = generate_scenario(name)
            r = detector.detect(img)
            supports = [d.confidence for d in r.detections]
            rows.append({"domain": name, "candidates": len(r.detections),
                         "mean_support": round(float(np.mean(supports)), 3) if supports else 0.0})
    for row in rows:
        print(row)
    counts = [r.get("candidates", r.get("mean_candidates", 0)) for r in rows]
    if counts:
        print(f"GAP: candidate range {min(counts)}-{max(counts)} across {len(rows)} domains. "
              f"No domain adaptation implemented; preprocessing/augmentation/context are "
              f"mechanisms under test, not claimed fixes.")


if __name__ == "__main__":
    main()
