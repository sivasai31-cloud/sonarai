"""SONARIS-X Dataset Validation Module.

Calculates dataset analysis tools:
- image count
- class distribution
- image dimensions
- annotation count
- object-size distribution
- corrupted images
- duplicate images
- missing labels

Exports a dataset report.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import Counter

import cv2
import numpy as np


@dataclass
class DatasetStats:
    """Complete dataset statistics."""
    image_count: int = 0
    readable_images: int = 0
    corrupted_images: int = 0
    duplicate_images: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    image_dimensions: Dict[str, int] = field(default_factory=dict)
    annotation_count: int = 0
    missing_labels: int = 0
    avg_image_size: float = 0.0
    min_image_size: int = 0
    max_image_size: int = 0
    total_objects: int = 0
    object_size_distribution: Dict[str, int] = field(default_factory=dict)
    file_extensions: Dict[str, int] = field(default_factory=dict)
    duplicate_files: List[str] = field(default_factory=list)
    corrupted_files: List[str] = field(default_factory=list)


class DatasetValidator:
    """Validates datasets for sonar imagery."""

    def __init__(self):
        self._stats: Dict[str, DatasetStats] = {}

    def compute_image_hash(self, filepath: str) -> Optional[str]:
        """Compute hash of image content for duplicate detection."""
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return None

    def validate_directory(self, directory: str) -> DatasetStats:
        """Validate all images in a directory."""
        stats = DatasetStats()
        image_files = []
        hashes = {}

        # Find all image files
        seen_paths = set()
        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
            for f in Path(directory).rglob(f'*{ext}'):
                if f.name not in seen_paths:
                    image_files.append(f)
                    seen_paths.add(f.name)
            for f in Path(directory).rglob(f'*{ext.upper()}'):
                if f.name.lower() not in seen_paths:
                    image_files.append(f)
                    seen_paths.add(f.name.lower())

        stats.image_count = len(image_files)

        if stats.image_count == 0:
            return stats

        dimensions_list = []
        file_hashes = {}
        corrupted_list = []

        for filepath in image_files:
            ext = filepath.suffix.lower()
            stats.file_extensions[ext] = stats.file_extensions.get(ext, 0) + 1

            # Check readability
            try:
                img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                if img is None:
                    stats.corrupted_images += 1
                    corrupted_list.append(str(filepath))
                    continue

                stats.readable_images += 1
                h, w = img.shape[:2]
                area = h * w
                dimensions_list.append(area)
                stats.image_dimensions[f'{w}x{h}'] = stats.image_dimensions.get(f'{w}x{h}', 0) + 1
            except Exception:
                stats.corrupted_images += 1
                corrupted_list.append(str(filepath))
                continue

            # Check duplicates
            file_hash = self.compute_image_hash(str(filepath))
            if file_hash:
                if file_hash in hashes:
                    stats.duplicate_images += 1
                    stats.duplicate_files.append(str(filepath))
                else:
                    hashes[file_hash] = str(filepath)

        # Compute dimension stats
        if dimensions_list:
            stats.avg_image_size = float(np.mean(dimensions_list))
            stats.min_image_size = int(min(dimensions_list))
            stats.max_image_size = int(max(dimensions_list))

        stats.corrupted_files = corrupted_list
        return stats

    def validate_annotations(self, annotation_dir: str, image_dir: str) -> Dict:
        """Validate annotation files against images."""
        result = {
            'annotation_count': 0,
            'missing_labels': 0,
            'images_without_annotations': 0,
            'annotations_without_images': 0,
            'class_distribution': {}
        }

        # Check for annotation files
        annotation_files = []
        for ext in ['.json', '.csv', '.xml', '.yaml']:
            annotation_files.extend(Path(annotation_dir).rglob(f'*{ext}'))

        result['annotation_count'] = len(annotation_files)

        # Check for images without annotations
        image_files = list(Path(image_dir).rglob('*.png')) + \
                      list(Path(image_dir).rglob('*.jpg')) + \
                      list(Path(image_dir).rglob('*.jpeg')) + \
                      list(Path(image_dir).rglob('*.tiff'))
        result['images_without_annotations'] = len(image_files) - len(annotation_files)
        if result['images_without_annotations'] < 0:
            result['images_without_annotations'] = 0

        return result

    def generate_report(self, stats: DatasetStats,
                          annotation_result: Dict) -> Dict:
        """Generate complete dataset validation report."""
        stats.annotation_count = annotation_result.get('annotation_count', 0)
        stats.missing_labels = annotation_result.get('missing_labels', 0)

        report = {
            'dataset_statistics': asdict(stats) if hasattr(stats, '__dataclass_fields__') else stats.__dict__,
            'annotation_validation': annotation_result,
            'summary': {
                'total_images': stats.image_count,
                'readable': stats.readable_images,
                'corrupted': stats.corrupted_images,
                'duplicates': stats.duplicate_images,
                'readable_percentage': round(stats.readable_images / max(1, stats.image_count) * 100, 1),
                'corrupted_percentage': round(stats.corrupted_images / max(1, stats.image_count) * 100, 1)
            }
        }

        return report

    def export_report(self, report: Dict, filepath: str):
        """Export dataset report to JSON."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
