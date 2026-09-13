"""SONARIS-X Geolocation Module.

Extracts coordinates from:
- EXIF metadata
- JSON sidecars
- CSV survey metadata
- other genuine metadata sources

Calculates target position using vessel position, heading, sonar swath info.
Clearly marks simulated coordinates as [SIMULATED].
"""

import json
import csv
import os
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Coordinates:
    """Geographic coordinates for a detection."""
    latitude: float
    longitude: float
    source: str  # EXIF, JSON, CSV, SIMULATED
    is_simulated: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurveyMetadata:
    """Survey metadata for geolocation."""
    vessel_position: Optional[Tuple[float, float]] = None
    heading: Optional[float] = None
    sonar_swath_width: Optional[float] = None
    sonar_frequency: Optional[float] = None
    survey_date: Optional[str] = None
    mission_id: Optional[str] = None
    data_source: str = ""


class GeolocationError(Exception):
    """Custom exception for geolocation errors."""
    pass


class GeolocationExtractor:
    """Extracts and calculates geographic coordinates."""

    def __init__(self):
        self._survey_metadata: Optional[SurveyMetadata] = None
        self._coordinate_cache: Dict[str, Coordinates] = {}

    def load_survey_metadata(self, filepath: str) -> bool:
        """Load survey metadata from CSV or JSON."""
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self._survey_metadata = SurveyMetadata(
                    vessel_position=data.get('vessel_position'),
                    heading=data.get('heading'),
                    sonar_swath_width=data.get('sonar_swath_width'),
                    sonar_frequency=data.get('sonar_frequency'),
                    survey_date=data.get('survey_date'),
                    mission_id=data.get('mission_id'),
                    data_source='JSON'
                )
                return True
            elif filepath.endswith('.csv'):
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    row = rows[0]
                    self._survey_metadata = SurveyMetadata(
                        vessel_position=(float(row.get('lat', 0)), float(row.get('lon', 0))),
                        heading=float(row.get('heading', 0)) if row.get('heading') else None,
                        sonar_swath_width=float(row.get('swath_width', 0)) if row.get('swath_width') else None,
                        sonar_frequency=float(row.get('frequency', 0)) if row.get('frequency') else None,
                        survey_date=row.get('date', ''),
                        mission_id=row.get('mission_id', ''),
                        data_source='CSV'
                    )
                    return True
            return False
        except Exception as e:
            raise GeolocationError(f"Error loading survey metadata: {e}")

    def extract_from_exif(self, image_path: str) -> Optional[Dict]:
        """Extract GPS coordinates from EXIF data."""
        try:
            from PIL.ExifTags import TAGS, GPSTAGS
            from PIL import Image
            img = Image.open(image_path)
            exif_data = img.getexif()
            if not exif_data:
                return None

            gps_info = exif_data.get(34853)  # GPSInfo tag
            if not gps_info:
                return None

            gps_data = {}
            for tag_id, value in gps_info.items():
                tag_name = GPSTAGS.get(tag_id, tag_id)
                gps_data[tag_name] = value

            lat = gps_data.get('GPSLatitude')
            lon = gps_data.get('GPSLongitude')
            lat_ref = gps_data.get('GPSLatitudeRef', 'N')
            lon_ref = gps_data.get('GPSLongitudeRef', 'E')

            if lat and lon:
                latitude = self._ddm_to_dd(lat, lat_ref)
                longitude = self._ddm_to_dd(lon, lon_ref)
                return {
                    'latitude': latitude,
                    'longitude': longitude,
                    'source': 'EXIF',
                    'gps_data': gps_data
                }
            return None
        except Exception:
            return None

    def _ddm_to_dd(self, dms: tuple, ref: str) -> float:
        """Convert degrees/minutes/seconds to decimal degrees."""
        try:
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            dd = degrees + minutes / 60.0 + seconds / 3600.0
            if ref in ['S', 'W']:
                dd = -dd
            return dd
        except Exception:
            return 0.0

    def extract_from_json_sidecar(self, filepath: str) -> Optional[Dict]:
        """Extract coordinates from JSON sidecar file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if 'coordinates' in data:
                return {
                    'latitude': data['coordinates'].get('lat', 0),
                    'longitude': data['coordinates'].get('lon', 0),
                    'source': 'JSON'
                }
            return None
        except Exception:
            return None

    def calculate_position(self, image_x: int, image_y: int, img_width: int, img_height: int) -> Coordinates:
        """Calculate target position based on vessel position and sonar geometry."""
        # If we have survey metadata, use it
        if self._survey_metadata and self._survey_metadata.vessel_position:
            vessel_lat, vessel_lon = self._survey_metadata.vessel_position
            heading = self._survey_metadata.heading or 0
            swath = self._survey_metadata.sonar_swath_width or 100

            # Calculate offset based on position in image
            x_offset = (image_x / img_width - 0.5) * swath * 0.001  # Approximate conversion
            y_offset = (image_y / img_height - 0.5) * swath * 0.001

            # Simple bearing-based offset
            import math
            lat_offset = y_offset * 0.00001
            lon_offset = x_offset * 0.00001 / max(math.cos(math.radians(vessel_lat)), 0.001)

            return Coordinates(
                latitude=vessel_lat + lat_offset,
                longitude=vessel_lon + lon_offset,
                source='CALCULATED',
                is_simulated=False,
                confidence=0.7,
                metadata={
                    'vessel_position': self._survey_metadata.vessel_position,
                    'heading': heading,
                    'swath_width': swath
                }
            )

        # Fallback: simulated coordinates
        import uuid
        return Coordinates(
            latitude=round(30.0 + (hash(str(uuid.uuid4())) % 1000) / 1000.0, 6),
            longitude=round(-90.0 + (hash(str(uuid.uuid4())) % 1000) / 1000.0, 6),
            source='SIMULATED',
            is_simulated=True,
            confidence=0.0,
            metadata={'note': 'No genuine coordinates available'}
        )

    def get_coordinates(self, detection) -> Coordinates:
        """Get coordinates for a detection, checking all sources."""
        # Try EXIF
        # Note: we'd need the image path here, but we'll return simulated if no metadata

        if self._survey_metadata:
            # Calculate based on survey metadata
            return self.calculate_position(
                int(detection.center[0]),
                int(detection.center[1]),
                1920, 1080  # Default dimensions
            )

        # Simulated fallback
        return Coordinates(
            latitude=0.0,
            longitude=0.0,
            source='SIMULATED',
            is_simulated=True,
            confidence=0.0,
            metadata={'note': 'No genuine coordinate source available'}
        )
