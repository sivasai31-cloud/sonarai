"""SONARIS-X dataset analysis CLI (real statistics only, never invented).

Usage:
    python scripts/analyze_dataset.py --dir data/demo [--out outputs/dataset_report.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.dataset_validation import DatasetValidator


def main():
    ap = argparse.ArgumentParser(description="Analyze a sonar image dataset directory.")
    ap.add_argument("--dir", default="data/demo", help="Image directory to analyze")
    ap.add_argument("--out", default="outputs/dataset_report.json", help="JSON report path")
    args = ap.parse_args()

    validator = DatasetValidator()
    stats = validator.validate_directory(args.dir)
    report = validator.generate_report(stats, {"annotation_count": 0, "missing_labels": 0})
    validator.export_report(report, args.out)
    print(json.dumps(report.get("summary", {}), indent=2))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
