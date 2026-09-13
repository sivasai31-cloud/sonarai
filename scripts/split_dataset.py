"""SONARIS-X leakage-safe dataset splitting CLI.

Prefers splitting by mission/survey/location/session when a metadata
CSV/JSON with those columns is supplied; falls back to seeded random split.

Usage:
    python scripts/split_dataset.py --dir data/raw [--meta survey.csv] [--strategy mission]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import TrainingPipeline, TrainingConfig


def load_metadata(meta_path):
    if not meta_path or not os.path.exists(meta_path):
        return None
    if meta_path.endswith(".json"):
        with open(meta_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    rows = []
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Leakage-safe dataset splitting.")
    ap.add_argument("--dir", default="data/raw")
    ap.add_argument("--meta", default=None, help="CSV/JSON metadata with mission/survey/location/session columns")
    ap.add_argument("--strategy", default="mission",
                    choices=["mission", "survey", "location", "session", "random"])
    args = ap.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")
    paths = sorted(str(p) for p in Path(args.dir).rglob("*") if p.suffix.lower() in exts)
    print(f"Found {len(paths)} images in {args.dir}")
    meta = load_metadata(args.meta)
    strategy = "random" if meta is None else args.strategy
    cfg = TrainingConfig(split_strategy=strategy)
    pipe = TrainingPipeline(cfg)
    # align metadata rows to sorted paths by filename when possible
    splits = pipe.split_dataset(paths, meta)
    print(json.dumps({k: v for k, v in splits.items() if not k.endswith("paths") and k != "train"
                      and k != "validation" and k != "test"}, indent=2))
    print(f"train={splits['train_count']} val={splits['val_count']} test={splits['test_count']} "
          f"(strategy={splits['split_strategy']})")


if __name__ == "__main__":
    main()
