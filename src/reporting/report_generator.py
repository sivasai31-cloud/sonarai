"""SONARIS-X Reporting Module.

Generates CSV, JSON, and human-readable mission reports.
Each target includes: ID, classification, confidence, anomaly score, risk,
evidence, bounding box, coordinates, timestamp, source image, processing info.
"""

import json
import csv
import os
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


@dataclass
class TargetReportEntry:
    """Report entry for a single target."""
    target_id: str
    classification: str
    class_name: str
    model_confidence: float
    anomaly_score: float
    risk_level: str
    risk_score: float
    bbox: List[int]
    center_x: float
    center_y: float
    area: int
    latitude: float
    longitude: float
    coordinate_source: str
    is_simulated_coords: bool
    evidence: List[str]
    image_width: int
    image_height: int
    processing_time_ms: float
    fingerprint_area: int
    fingerprint_aspect_ratio: float
    timestamp: str = ""
    source_image: str = ""
    model_status: str = ""
    quality_score: float = 0.0
    quality_rating: str = ""
    overlaps_dropout: bool = False
    fingerprint_summary: str = ""
    filter_decision: str = ""


class ReportGenerator:
    """Generates mission reports in multiple formats."""

    def __init__(self):
        self._reports: List[TargetReportEntry] = []
        self._mission_info: Dict[str, any] = {}

    def set_mission_info(self, mission_id: str, operator: str,
                            mission_date: str, equipment: str = ""):
        """Set mission metadata."""
        self._mission_info = {
            'mission_id': mission_id,
            'operator': operator,
            'mission_date': mission_date,
            'equipment': equipment,
            'generated_at': datetime.now().isoformat()
        }

    def add_target(self, entry: TargetReportEntry):
        """Add a target to the report."""
        entry.timestamp = datetime.now().isoformat()
        self._reports.append(entry)

    def generate_json(self, filepath: str) -> Dict:
        """Generate JSON report."""
        report = {
            'mission_info': self._mission_info,
            'generated_at': datetime.now().isoformat(),
            'total_targets': len(self._reports),
            'targets': [asdict(entry) for entry in self._reports]
        }

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def generate_csv(self, filepath: str) -> List[Dict]:
        """Generate CSV report."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        if not self._reports:
            return []

        fieldnames = list(asdict(self._reports[0]).keys())
        rows = [asdict(entry) for entry in self._reports]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return rows

    def generate_text_report(self, filepath: str) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("SONARIS-X MISSION REPORT")
        lines.append("=" * 70)
        lines.append("")

        if self._mission_info:
            lines.append(f"Mission ID: {self._mission_info.get('mission_id', 'N/A')}")
            lines.append(f"Operator: {self._mission_info.get('operator', 'N/A')}")
            lines.append(f"Date: {self._mission_info.get('mission_date', 'N/A')}")
            lines.append(f"Equipment: {self._mission_info.get('equipment', 'N/A')}")
            lines.append("")

        lines.append(f"Total Targets: {len(self._reports)}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("-" * 70)

        for entry in self._reports:
            lines.append("")
            lines.append(f"TARGET {entry.target_id}")
            lines.append(f"  Class: {entry.class_name} ({entry.classification})")
            lines.append(f"  Model Confidence: {entry.model_confidence:.3f}")
            lines.append(f"  Anomaly Score: {entry.anomaly_score:.1f}")
            lines.append(f"  Risk Level: {entry.risk_level} (score: {entry.risk_score:.1f})")
            lines.append(f"  Position: ({entry.latitude:.6f}, {entry.longitude:.6f}) [{entry.coordinate_source}]")
            if entry.is_simulated_coords:
                lines.append(f"  NOTE: Coordinates are SIMULATED")
            lines.append(f"  BBox: [{entry.bbox[0]}, {entry.bbox[1]}, {entry.bbox[2]}, {entry.bbox[3]}]")
            lines.append(f"  Area: {entry.fingerprint_area}, Aspect Ratio: {entry.fingerprint_aspect_ratio:.2f}")
            lines.append(f"  Evidence: {'; '.join(entry.evidence[:5])}")
            lines.append(f"  Processing Time: {entry.processing_time_ms:.1f}ms")
            lines.append("-" * 70)

        lines.append("")
        lines.append("END OF REPORT")

        report_text = "\n".join(lines)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(report_text)

        return report_text

    def get_summary_stats(self) -> Dict:
        """Get summary statistics from all reports."""
        if not self._reports:
            return {'total': 0}

        high_risk = sum(1 for r in self._reports if r.risk_level == 'HIGH')
        medium_risk = sum(1 for r in self._reports if r.risk_level == 'MEDIUM')
        low_risk = sum(1 for r in self._reports if r.risk_level == 'LOW')
        unknown = sum(1 for r in self._reports if 'UNKNOWN' in r.classification)
        confirmed = sum(1 for r in self._reports if 'CONFIRMED' in r.classification)
        simulated_coords = sum(1 for r in self._reports if r.is_simulated_coords)

        avg_confidence = sum(r.model_confidence for r in self._reports) / len(self._reports)
        avg_anomaly = sum(r.anomaly_score for r in self._reports) / len(self._reports)

        return {
            'total_targets': len(self._reports),
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'unknown_anomalies': unknown,
            'confirmed_targets': confirmed,
            'simulated_coordinates': simulated_coords,
            'average_confidence': round(avg_confidence, 3),
            'average_anomaly_score': round(avg_anomaly, 1)
        }
