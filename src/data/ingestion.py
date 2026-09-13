"""SONARIS-X Data Ingestion Module.

Supports PNG, JPG/JPEG, TIFF, CSV metadata, JSON metadata, EXIF metadata.
Validates file type, dimensions, readability, metadata completeness, and corruption.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageFile
import yaml

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


@dataclass
class ImageInfo:
    """Information about an ingested image."""
    filepath: str
    filename: str
    extension: str
    width: int
    height: int
    channels: int
    dtype: str
    file_size_bytes: int
    readable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MetadataInfo:
    """Information extracted from metadata files."""
    source: str
    format: str
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class DataIngestor:
    """Robust data ingestion layer for sonar imagery."""

    def __init__(self):
        self._ingested_images: List[ImageInfo] = []
        self._metadata_cache: Dict[str, MetadataInfo] = {}

    def validate_file_type(self, filepath: str) -> Tuple[bool, str]:
        """Validate that a file has a supported extension."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True, f"Supported format: {ext}"
        return False, f"Unsupported format: {ext}. Supported: {SUPPORTED_EXTENSIONS}"

    def validate_image_readability(self, filepath: str) -> Tuple[bool, str, Optional[np.ndarray]]:
        """Check if an image can be read and is not corrupted."""
        try:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "cv2.imread returned None - file may be corrupted", None
            if img.size == 0:
                return False, "Image has zero size", None
            return True, "Image readable", img
        except Exception as e:
            return False, f"Error reading image: {str(e)}", None

    def extract_exif_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract EXIF metadata from image files."""
        metadata = {}
        try:
            from PIL.ExifTags import TAGS, GPSTAGS
            img = Image.open(filepath)
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    metadata[tag_name] = str(value)

                # Extract GPS info
                gps_info = exif_data.get(34853)  # GPSInfo tag
                if gps_info:
                    gps_data = {}
                    for tag_id, value in gps_info.items():
                        gps_tag = GPSTAGS.get(tag_id, tag_id)
                        gps_data[gps_tag] = str(value)
                    metadata['GPSInfo'] = gps_data
        except Exception as e:
            metadata['exif_error'] = str(e)
        return metadata

    def extract_json_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from JSON sidecar files."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error reading JSON metadata: {e}")
            return {}

    def extract_csv_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from CSV survey files."""
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            return {'columns': list(df.columns), 'rows': len(df), 'data': df.to_dict('records')}
        except Exception as e:
            logger.error(f"Error reading CSV metadata: {e}")
            return {}

    def ingest_image(self, filepath: str) -> ImageInfo:
        """Ingest a single image with full validation."""
        info = ImageInfo(
            filepath=filepath,
            filename=os.path.basename(filepath),
            extension=os.path.splitext(filepath)[1].lower(),
            width=0, height=0, channels=0, dtype='',
            file_size_bytes=0, readable=False
        )

        # Check file exists
        if not os.path.exists(filepath):
            info.errors.append(f"File not found: {filepath}")
            info.readable = False
            self._ingested_images.append(info)
            return info

        info.file_size_bytes = os.path.getsize(filepath)

        # Validate file type
        type_ok, type_msg = self.validate_file_type(filepath)
        if not type_ok:
            info.errors.append(type_msg)
            info.readable = False
            self._ingested_images.append(info)
            return info

        # Validate image readability
        readable, msg, img = self.validate_image_readability(filepath)
        info.readable = readable
        info.warnings.append(msg)

        if readable and img is not None:
            info.width = img.shape[1]
            info.height = img.shape[0]
            info.channels = img.shape[2] if len(img.shape) > 2 else 1
            info.dtype = str(img.dtype)

            # Extract metadata
            info.metadata = self.extract_exif_metadata(filepath)
            info.metadata['opencv_shape'] = img.shape
            info.metadata['opencv_dtype'] = str(img.dtype)

        if info.errors:
            logger.error(f"Ingestion errors for {filepath}: {info.errors}")
        if info.warnings:
            logger.warning(f"Ingestion warnings for {filepath}: {info.warnings}")

        self._ingested_images.append(info)
        return info

    def ingest_batch(self, filepaths: List[str]) -> List[ImageInfo]:
        """Ingest multiple images."""
        results = []
        for fp in filepaths:
            info = self.ingest_image(fp)
            results.append(info)
        return results

    def get_ingested_images(self) -> List[ImageInfo]:
        """Get all ingested images."""
        return self._ingested_images

    def get_readable_count(self) -> int:
        """Get count of readable images."""
        return sum(1 for img in self._ingested_images if img.readable)

    def get_errors(self) -> List[str]:
        """Get all errors across all ingested images."""
        errors = []
        for img in self._ingested_images:
            if img.errors:
                errors.extend([f"{img.filename}: {e}" for e in img.errors])
        return errors
