import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  MapContainer,
  TileLayer,
  Polyline,
  Polygon,
  Marker,
  Popup,
  useMap,
} from 'react-leaflet';
import L from 'leaflet';
import {
  Map as MapIcon,
  Layers,
  Compass,
  Crosshair,
  Maximize2,
  Shield,
  AlertTriangle,
  CheckCircle2,
  Radio,
  Info,
  Navigation,
  ChevronUp,
  ChevronDown,
  ArrowUpRight,
  Eye,
  Sliders,
} from 'lucide-react';
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

export interface TargetCoordinates {
  latitude: number;
  longitude: number;
}

export interface AcousticFingerprint {
  shape?: number;
  aspect_ratio?: number;
  area?: number;
  intensity?: number;
  texture?: number;
  edge_density?: number;
  shadow?: number;
  shadow_present?: boolean;
  seabed_divergence?: number;
  texture_entropy?: number;
  [key: string]: any;
}

export interface MapTarget {
  target_id: string;
  display_name: string;
  class_name: string;
  class_description?: string;
  model_confidence: number;
  anomaly_score?: number;
  risk_level: string;
  risk_score?: number;
  coordinates: TargetCoordinates;
  coordinates_simulated?: boolean;
  coordinate_source?: string;
  coordinates_label?: string;
  acoustic_fingerprint?: AcousticFingerprint;
  realistic_view?: any;
  evidence?: string[];
  explanation?: {
    summary?: string;
    reasons?: string[];
  };
  status?: string;
  analysis_status?: string;
  source?: string;
  data_source_label?: string;
}

export interface Waypoint {
  name: string;
  latitude: number;
  longitude: number;
  depth_m?: number;
  speed_kts?: number;
  status?: string;
}

export interface MissionTrackData {
  mission_id: string;
  mission_name: string;
  status: string;
  vessel: string;
  sensor: string;
  track_type: string;
  track_label: string;
  waypoints: Waypoint[];
  survey_area: [number, number][];
}

interface AcousticMapProps {
  targets?: MapTarget[];
  activeMissionId?: string;
  modelStatus?: string;
  backendStatus?: 'online' | 'offline' | 'connecting';
  selectedTarget?: MapTarget | null;
  onSelectTarget?: (target: MapTarget) => void;
  onReviewTarget?: (decision: string) => void;
  onNavigate?: (section: string) => void;
  theme?: 'dark' | 'light';
}

// Fallback verified reference targets if cold start
const DEFAULT_DEMO_TARGETS: MapTarget[] = [
  {
    target_id: 'SX-014',
    display_name: 'MILCO',
    class_name: 'MILCO',
    class_description: 'Mine-Like Contact',
    model_confidence: 0.86,
    anomaly_score: 84.5,
    risk_level: 'MEDIUM',
    risk_score: 68.0,
    coordinates: { latitude: 54.8214, longitude: 3.4219 },
    coordinates_simulated: true,
    coordinate_source: 'SIMULATED',
    coordinates_label: 'SIMULATED LOCATION',
    status: 'REVIEW REQUIRED',
    acoustic_fingerprint: {
      shape: 0.68,
      aspect_ratio: 1.15,
      area: 580,
      intensity: 0.74,
      texture: 0.58,
      edge_density: 0.42,
      shadow: 145.0,
      shadow_present: true,
      seabed_divergence: 0.52,
    },
    evidence: [
      'Sonar-trained class match: MILCO',
      'High shadow length divergence confirms elevated contact',
      'Aspect ratio consistent with cylindrical metallic ordnance',
    ],
  },
  {
    target_id: 'SX-008',
    display_name: 'NOMBO',
    class_name: 'NOMBO',
    class_description: 'Non-Mine-like Bottom Object',
    model_confidence: 0.78,
    anomaly_score: 38.0,
    risk_level: 'LOW',
    risk_score: 24.0,
    coordinates: { latitude: 54.832, longitude: 3.435 },
    coordinates_simulated: true,
    coordinate_source: 'SIMULATED',
    coordinates_label: 'SIMULATED LOCATION',
    status: 'VERIFIED NATURAL',
    acoustic_fingerprint: {
      shape: 0.45,
      aspect_ratio: 1.85,
      area: 820,
      intensity: 0.42,
      texture: 0.75,
      edge_density: 0.28,
      shadow: 80.0,
      shadow_present: true,
      seabed_divergence: 0.31,
    },
    evidence: [
      'Sonar-trained class match: NOMBO',
      'Irregular geological boundary matches natural boulder outcrop',
    ],
  },
  {
    target_id: 'SX-022',
    display_name: 'Pipeline',
    class_name: 'Pipeline',
    class_description: 'Subsea Pipeline Infrastructure',
    model_confidence: 0.92,
    anomaly_score: 76.0,
    risk_level: 'HIGH',
    risk_score: 82.0,
    coordinates: { latitude: 54.815, longitude: 3.41 },
    coordinates_simulated: true,
    coordinate_source: 'SIMULATED',
    coordinates_label: 'SIMULATED LOCATION',
    status: 'REVIEW REQUIRED',
    acoustic_fingerprint: {
      shape: 0.88,
      aspect_ratio: 3.4,
      area: 1200,
      intensity: 0.81,
      texture: 0.35,
      edge_density: 0.65,
      shadow: 210.0,
      shadow_present: true,
      seabed_divergence: 0.68,
    },
    evidence: [
      'Continuous linear acoustic reflector across entire swath',
      'Uniform cylindrical specular return',
    ],
  },
];

const DEFAULT_TRACK_DATA: MissionTrackData = {
  mission_id: 'SONAR-01',
  mission_name: 'NORTH SEA SECTOR 04 SURVEY',
  status: 'ACTIVE',
  vessel: 'AUV ORCA-X (Simulated Telemetry)',
  sensor: 'Dual-Frequency Side-Scan Sonar (455/900 kHz)',
  track_type: 'SIMULATED',
  track_label: 'DEMO MISSION TRACK (SIMULATED)',
  waypoints: [
    { name: 'START: WP-01', latitude: 54.8105, longitude: 3.402, depth_m: 42.5, speed_kts: 3.2, status: 'PASSED' },
    { name: 'WP-02', latitude: 54.8162, longitude: 3.4118, depth_m: 45.0, speed_kts: 3.2, status: 'PASSED' },
    { name: 'WP-03 (CONTACT ZONE)', latitude: 54.8214, longitude: 3.4219, depth_m: 46.8, speed_kts: 3.1, status: 'ON STATION' },
    { name: 'WP-04', latitude: 54.827, longitude: 3.4335, depth_m: 48.2, speed_kts: 3.2, status: 'PLANNED' },
    { name: 'WP-05', latitude: 54.8322, longitude: 3.441, depth_m: 49.5, speed_kts: 3.3, status: 'PLANNED' },
    { name: 'END: WP-06', latitude: 54.838, longitude: 3.453, depth_m: 51.0, speed_kts: 3.0, status: 'PLANNED' },
  ],
  survey_area: [
    [54.805, 3.395],
    [54.842, 3.425],
    [54.835, 3.465],
    [54.798, 3.435],
  ],
};

// Create custom tactical SVG divIcon for markers
function createTacticalMarkerIcon(
  target: MapTarget,
  isSelected: boolean
): L.DivIcon {
  const riskColor =
    target.risk_level === 'HIGH'
      ? '#f45d65'
      : target.risk_level === 'MEDIUM'
      ? '#f3b544'
      : '#4ee29b';

  const selectedClass = isSelected ? 'selected-beacon' : '';

  const html = `
    <div class="tactical-marker-pin ${selectedClass}" style="--marker-color: ${riskColor}">
      ${
        isSelected
          ? `<div class="marker-pulse-ring" style="border-color: ${riskColor}; box-shadow: 0 0 12px ${riskColor};"></div>`
          : ''
      }
      <div class="marker-diamond" style="border-color: ${riskColor}; background: rgba(3, 14, 24, 0.92);">
        <div class="marker-dot" style="background: ${riskColor};"></div>
      </div>
      <div class="marker-id-badge" style="border-color: ${riskColor}; color: ${riskColor};">
        ${target.target_id.slice(-6)}
      </div>
    </div>
  `;

  return L.divIcon({
    className: 'custom-tactical-marker',
    html,
    iconSize: [44, 44],
    iconAnchor: [22, 22],
    popupAnchor: [0, -20],
  });
}

// Waypoint custom marker icons for START and END
function createWaypointIcon(name: string, isStart: boolean): L.DivIcon {
  const color = isStart ? '#4ee29b' : '#38bdf8';
  const label = isStart ? 'START' : 'END';
  const html = `
    <div class="waypoint-pin" style="--wp-color: ${color}">
      <div class="waypoint-beacon" style="background: ${color}; box-shadow: 0 0 10px ${color};"></div>
      <div class="waypoint-label" style="border-color: ${color}; color: ${color};">${label}</div>
    </div>
  `;
  return L.divIcon({
    className: 'custom-waypoint-marker',
    html,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -16],
  });
}

// Controller component to handle map programmatic viewport actions
function MapController({
  fitTrigger,
  targets,
  trackData,
  selectedTarget,
}: {
  fitTrigger: number;
  targets: MapTarget[];
  trackData: MissionTrackData;
  selectedTarget: MapTarget | null;
}) {
  const map = useMap();

  // Fit Mission bounds smoothly
  useEffect(() => {
    if (!map) return;
    const points: [number, number][] = [];

    // Collect track waypoints
    trackData.waypoints.forEach((wp) => {
      points.push([wp.latitude, wp.longitude]);
    });

    // Collect target points
    targets.forEach((t) => {
      if (t.coordinates && t.coordinates.latitude !== 0) {
        points.push([t.coordinates.latitude, t.coordinates.longitude]);
      }
    });

    if (points.length > 0) {
      map.flyToBounds(points, {
        padding: [60, 60],
        maxZoom: 14,
        duration: 1.2,
      });
    }
  }, [map, fitTrigger, trackData, targets]);

  // Center on selected target if changed specifically
  useEffect(() => {
    if (selectedTarget && selectedTarget.coordinates && map) {
      const { latitude, longitude } = selectedTarget.coordinates;
      if (latitude !== 0 || longitude !== 0) {
        map.panTo([latitude, longitude], { animate: true, duration: 0.8 });
      }
    }
  }, [map, selectedTarget]);

  return null;
}

export function AcousticMap({
  targets: initialTargets,
  activeMissionId = 'SONAR-01',
  modelStatus = 'CUSTOM SONAR WEIGHTS LOADED',
  backendStatus = 'online',
  selectedTarget: propSelectedTarget,
  onSelectTarget,
  onReviewTarget,
  onNavigate,
}: AcousticMapProps) {
  // Configurable Map Tile Providers via Environment Variables
  const envApiKey = import.meta.env.VITE_MAP_API_KEY || 'AIzaSyDJlWuel9pbfnbh375QP1dvXEZMFFJ_srU';
  const envTileUrl = import.meta.env.VITE_MAP_TILE_URL;
  const envSatelliteUrl =
    import.meta.env.VITE_MAP_SATELLITE_URL ||
    `https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&key=${envApiKey}`;
  const envAttribution =
    import.meta.env.VITE_MAP_ATTRIBUTION ||
    '&copy; Google Maps &copy; OpenStreetMap contributors &copy; CARTO';

  // Primary Tile Provider fallback (Dark Ocean Navigation)
  const defaultDarkTile =
    'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
  const defaultSatelliteTile =
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

  // Apply optional API key if provided and url has placeholder
  const formatTileUrl = (url: string) => {
    if (envApiKey && url.includes('{key}')) {
      return url.replace('{key}', envApiKey);
    }
    return url;
  };

  // Layer Visibility State
  const [activeBaseLayer, setActiveBaseLayer] = useState<'dark' | 'satellite'>('dark');
  const [showMissionTrack, setShowMissionTrack] = useState(true);
  const [showTargets, setShowTargets] = useState(true);
  const [highRiskOnly, setHighRiskOnly] = useState(false);
  const [showSurveyArea, setShowSurveyArea] = useState(true);
  const [showLayerMenu, setShowLayerMenu] = useState(false);

  const primaryTileUrl = formatTileUrl(envTileUrl || defaultDarkTile);
  const satelliteTileUrl = formatTileUrl(envSatelliteUrl || defaultSatelliteTile);
  const currentTileUrl = activeBaseLayer === 'dark' ? primaryTileUrl : satelliteTileUrl;
  const isGoogleTile = currentTileUrl.includes('google.com');
  const tileSubdomains = isGoogleTile ? ['0', '1', '2', '3'] : ['a', 'b', 'c', 'd'];
  const mapAttribution =
    envAttribution ||
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

  // Mobile Bottom Drawer State
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);

  // Targets state (ensures fallback to demo targets if empty)
  const [targetsList, setTargetsList] = useState<MapTarget[]>([]);
  const [trackData, setTrackData] = useState<MissionTrackData>(DEFAULT_TRACK_DATA);
  const [fitTrigger, setFitTrigger] = useState<number>(0);

  // Internal selected target
  const [selected, setSelected] = useState<MapTarget | null>(null);

  // Load mission track and targets from API or fallback
  useEffect(() => {
    let isMounted = true;

    async function loadMapData() {
      // 1. Fetch Track Data
      try {
        const trackRes = await fetch('/api/map/mission-track');
        if (trackRes.ok) {
          const data = await trackRes.json();
          if (isMounted) setTrackData(data);
        }
      } catch {
        // Retain DEFAULT_TRACK_DATA
      }

      // 2. Fetch or sync targets
      if (initialTargets && initialTargets.length > 0) {
        if (isMounted) setTargetsList(initialTargets);
      } else {
        try {
          const targetsRes = await fetch('/api/map/targets');
          if (targetsRes.ok) {
            const data = await targetsRes.json();
            if (isMounted && Array.isArray(data) && data.length > 0) {
              setTargetsList(data);
            } else if (isMounted) {
              setTargetsList(DEFAULT_DEMO_TARGETS);
            }
          } else if (isMounted) {
            setTargetsList(DEFAULT_DEMO_TARGETS);
          }
        } catch {
          if (isMounted) setTargetsList(DEFAULT_DEMO_TARGETS);
        }
      }
    }

    loadMapData();
    return () => {
      isMounted = false;
    };
  }, [initialTargets]);

  // Synchronize external prop selectedTarget
  useEffect(() => {
    if (propSelectedTarget) {
      setSelected(propSelectedTarget);
    } else if (!selected && targetsList.length > 0) {
      // Default select the highest risk target
      const priority =
        targetsList.find((t) => t.risk_level === 'HIGH') ||
        targetsList.find((t) => t.risk_level === 'MEDIUM') ||
        targetsList[0];
      setSelected(priority);
    }
  }, [propSelectedTarget, targetsList, selected]);

  const handleSelect = useCallback(
    (target: MapTarget) => {
      setSelected(target);
      if (onSelectTarget) onSelectTarget(target);
      // Open drawer on mobile when target is selected
      if (window.innerWidth < 1024) {
        setMobileDrawerOpen(true);
      }
    },
    [onSelectTarget]
  );

  // Filtered targets
  const displayTargets = useMemo(() => {
    if (!showTargets) return [];
    if (highRiskOnly) {
      return targetsList.filter((t) => t.risk_level === 'HIGH');
    }
    return targetsList;
  }, [targetsList, showTargets, highRiskOnly]);

  // Track polyline coordinates
  const trackPositions: [number, number][] = useMemo(() => {
    return trackData.waypoints.map((wp) => [wp.latitude, wp.longitude]);
  }, [trackData]);

  // Initial center position
  const initialCenter: [number, number] = useMemo(() => {
    if (targetsList.length > 0 && targetsList[0].coordinates.latitude !== 0) {
      return [targetsList[0].coordinates.latitude, targetsList[0].coordinates.longitude];
    }
    return [54.8214, 3.4219];
  }, [targetsList]);

  // Mini radar data for acoustic fingerprint
  const radarData = useMemo(() => {
    const fp = selected?.acoustic_fingerprint;
    if (!fp) {
      return [
        { subject: 'Shape', value: 0.5 },
        { subject: 'Aspect', value: 0.4 },
        { subject: 'Intensity', value: 0.6 },
        { subject: 'Texture', value: 0.5 },
        { subject: 'Edges', value: 0.4 },
        { subject: 'Shadow', value: 0.3 },
      ];
    }
    return [
      { subject: 'Shape', value: typeof fp.shape === 'number' ? Math.min(1, fp.shape) : 0.5 },
      { subject: 'Aspect', value: typeof fp.aspect_ratio === 'number' ? Math.min(1, fp.aspect_ratio / 3.5) : 0.4 },
      { subject: 'Intensity', value: typeof fp.intensity === 'number' ? Math.min(1, fp.intensity) : 0.6 },
      { subject: 'Texture', value: typeof fp.texture === 'number' ? Math.min(1, fp.texture) : 0.5 },
      { subject: 'Edges', value: typeof fp.edge_density === 'number' ? Math.min(1, fp.edge_density) : 0.4 },
      { subject: 'Shadow', value: typeof fp.shadow === 'number' ? Math.min(1, fp.shadow / 200) : 0.3 },
    ];
  }, [selected]);

  // Status string for top bar
  const isModelOnline = modelStatus.includes('LOADED');
  const isSimulated =
    selected?.coordinates_simulated ??
    targetsList.some((t) => t.coordinates_simulated ?? true);

  return (
    <div className="acoustic-map-layout">
      {/* ====================================================================
          1. TOP BAR — Marine Intelligence Telemetry
          ==================================================================== */}
      <header className="map-topbar">
        <div className="map-topbar-left">
          <div className="map-title-block">
            <span className="map-icon-badge">
              <MapIcon size={16} />
            </span>
            <div>
              <div className="map-title-main">ACOUSTIC MAP</div>
              <div className="map-title-sub">REAL-TIME TACTICAL GEO-INTELLIGENCE</div>
            </div>
          </div>

          <div className="map-telemetry-divider" />

          <div className="map-telemetry-item">
            <span className="telemetry-label">MISSION</span>
            <strong className="telemetry-value highlight-cyan">
              {activeMissionId || trackData.mission_id}
            </strong>
          </div>

          <div className="map-telemetry-item">
            <span className="telemetry-label">STATUS</span>
            <strong className="telemetry-value status-active">
              <span className="status-dot-pulse" />
              ACTIVE
            </strong>
          </div>

          <div className="map-telemetry-item">
            <span className="telemetry-label">TARGETS</span>
            <strong className="telemetry-value">
              {String(displayTargets.length).padStart(2, '0')}
            </strong>
          </div>
        </div>

        <div className="map-topbar-right">
          {/* System states */}
          <div className="map-status-pill-group">
            <span
              className={`map-status-pill ${
                backendStatus === 'offline'
                  ? 'pill-danger'
                  : isModelOnline
                  ? 'pill-success'
                  : 'pill-warning'
              }`}
              title={
                isModelOnline
                  ? 'YOLOv8 custom sonar model weights verified & active'
                  : 'Model offline. Preprocessing & verified reference contacts active'
              }
            >
              <Radio size={11} />
              {backendStatus === 'offline'
                ? 'BACKEND OFFLINE'
                : isModelOnline
                ? 'MODEL ONLINE'
                : 'MODEL OFFLINE'}
            </span>

            <span className="map-status-pill pill-cyan">
              <Compass size={11} />
              DEMO DATA AVAILABLE
            </span>

            <span className="map-status-pill pill-ready">
              <CheckCircle2 size={11} />
              MAP READY
            </span>
          </div>

          {/* Fit Mission Button */}
          <button
            className="btn-fit-mission"
            onClick={() => setFitTrigger((prev) => prev + 1)}
            title="Fit mission track and all target contacts in view"
          >
            <Maximize2 size={13} />
            <span>FIT MISSION</span>
          </button>
        </div>
      </header>

      {/* ====================================================================
          2. MAIN WORKSPACE: MAP VIEWPORT + INTEL PANEL
          ==================================================================== */}
      <div className="map-workspace-body">
        {/* Main Map Viewport */}
        <div className="map-canvas-container">
          <MapContainer
            center={initialCenter}
            zoom={12}
            scrollWheelZoom={true}
            className="leaflet-interactive-map"
            zoomControl={false}
          >
            {/* Dynamic Tile Layer (Dark Basemap vs Satellite) */}
            <TileLayer
              key={`${activeBaseLayer}-${currentTileUrl}`}
              attribution={mapAttribution}
              url={currentTileUrl}
              subdomains={tileSubdomains}
              maxZoom={20}
            />

            {/* Programmatic Map Controller */}
            <MapController
              fitTrigger={fitTrigger}
              targets={displayTargets}
              trackData={trackData}
              selectedTarget={selected}
            />

            {/* Survey Area Swath Polygon */}
            {showSurveyArea && trackData.survey_area && (
              <Polygon
                positions={trackData.survey_area}
                pathOptions={{
                  color: '#38bdf8',
                  weight: 1.5,
                  dashArray: '5, 5',
                  fillColor: '#0284c7',
                  fillOpacity: 0.08,
                }}
              >
                <Popup>
                  <div className="tactical-popup-body">
                    <strong>SURVEY AREA · NORTH SEA SECTOR 04</strong>
                    <span>Swath Bounding Polygon (Simulated Demo Area)</span>
                  </div>
                </Popup>
              </Polygon>
            )}

            {/* Mission Track Polyline */}
            {showMissionTrack && trackPositions.length > 1 && (
              <Polyline
                positions={trackPositions}
                pathOptions={{
                  color: '#00e5ff',
                  weight: 2.5,
                  dashArray: '8, 6',
                  opacity: 0.85,
                }}
              />
            )}

            {/* Mission Track Waypoints (Start & End) */}
            {showMissionTrack && trackData.waypoints.length > 0 && (
              <>
                <Marker
                  position={[
                    trackData.waypoints[0].latitude,
                    trackData.waypoints[0].longitude,
                  ]}
                  icon={createWaypointIcon(trackData.waypoints[0].name, true)}
                >
                  <Popup>
                    <div className="tactical-popup-body">
                      <b style={{ color: '#4ee29b' }}>{trackData.waypoints[0].name}</b>
                      <span>Depth: {trackData.waypoints[0].depth_m ?? 42} m</span>
                      <span>Speed: {trackData.waypoints[0].speed_kts ?? 3.2} kts</span>
                      <small style={{ color: 'var(--cyan)' }}>DEMO MISSION TRACK</small>
                    </div>
                  </Popup>
                </Marker>

                <Marker
                  position={[
                    trackData.waypoints[trackData.waypoints.length - 1].latitude,
                    trackData.waypoints[trackData.waypoints.length - 1].longitude,
                  ]}
                  icon={createWaypointIcon(
                    trackData.waypoints[trackData.waypoints.length - 1].name,
                    false
                  )}
                >
                  <Popup>
                    <div className="tactical-popup-body">
                      <b style={{ color: '#38bdf8' }}>
                        {trackData.waypoints[trackData.waypoints.length - 1].name}
                      </b>
                      <span>
                        Depth:{' '}
                        {trackData.waypoints[trackData.waypoints.length - 1].depth_m ?? 51}{' '}
                        m
                      </span>
                      <span>Planned Survey Recovery Fix</span>
                      <small style={{ color: 'var(--cyan)' }}>DEMO MISSION TRACK</small>
                    </div>
                  </Popup>
                </Marker>
              </>
            )}

            {/* Detected Target Markers */}
            {displayTargets.map((target) => {
              if (
                !target.coordinates ||
                (target.coordinates.latitude === 0 &&
                  target.coordinates.longitude === 0)
              ) {
                return null;
              }

              const isSelected = selected?.target_id === target.target_id;
              const markerIcon = createTacticalMarkerIcon(target, isSelected);

              return (
                <Marker
                  key={target.target_id}
                  position={[
                    target.coordinates.latitude,
                    target.coordinates.longitude,
                  ]}
                  icon={markerIcon}
                  eventHandlers={{
                    click: () => handleSelect(target),
                  }}
                >
                  <Popup>
                    <div className="tactical-popup-body">
                      <div className="popup-kicker">
                        <span className="popup-id">{target.target_id}</span>
                        <span
                          className={`popup-risk-tag ${
                            target.risk_level === 'HIGH'
                              ? 'risk-high'
                              : target.risk_level === 'MEDIUM'
                              ? 'risk-medium'
                              : 'risk-low'
                          }`}
                        >
                          {target.risk_level}
                        </span>
                      </div>

                      <div className="popup-class">{target.display_name}</div>
                      <div className="popup-taxonomy">
                        {target.class_description || 'Acoustic Seabed Target'}
                      </div>

                      <div className="popup-metrics-grid">
                        <div>
                          <small>CONFIDENCE</small>
                          <b>{(target.model_confidence * 100).toFixed(1)}%</b>
                        </div>
                        <div>
                          <small>COORDINATES</small>
                          <b style={{ fontSize: '9px' }}>
                            {target.coordinates.latitude.toFixed(4)}°,{' '}
                            {target.coordinates.longitude.toFixed(4)}°
                          </b>
                        </div>
                      </div>

                      <div className="popup-source-banner">
                        <Info size={11} />
                        <span>
                          {target.coordinates_simulated
                            ? 'SIMULATED LOCATION'
                            : 'GPS / SENSOR METADATA'}
                        </span>
                      </div>

                      <div className="popup-status-line">
                        <span>STATUS:</span>
                        <strong>{target.status || 'REVIEW REQUIRED'}</strong>
                      </div>

                      <button
                        className="btn-popup-inspect"
                        onClick={() => handleSelect(target)}
                      >
                        <Eye size={12} /> FOCUS TARGET IN INTEL PANEL
                      </button>
                    </div>
                  </Popup>
                </Marker>
              );
            })}
          </MapContainer>

          {/* Floating Map HUD Controls (Zoom + Layer Selector) */}
          <div className="map-floating-controls">
            {/* Layer Control Dropdown Trigger */}
            <div className="layer-control-wrap">
              <button
                className={`btn-map-control ${showLayerMenu ? 'active' : ''}`}
                onClick={() => setShowLayerMenu(!showLayerMenu)}
                title="Map Layer Controls"
              >
                <Layers size={15} />
              </button>

              {showLayerMenu && (
                <div className="map-layer-dropdown">
                  <div className="layer-dropdown-header">
                    <Sliders size={12} />
                    <span>MAP LAYERS</span>
                  </div>

                  <div className="layer-group">
                    <span className="layer-group-title">BASEMAP</span>
                    <label className="layer-option">
                      <input
                        type="radio"
                        name="baselayer"
                        checked={activeBaseLayer === 'dark'}
                        onChange={() => setActiveBaseLayer('dark')}
                      />
                      <span>DARK OCEAN NAVIGATION</span>
                    </label>
                    <label className="layer-option">
                      <input
                        type="radio"
                        name="baselayer"
                        checked={activeBaseLayer === 'satellite'}
                        onChange={() => setActiveBaseLayer('satellite')}
                      />
                      <span>SATELLITE / IMAGERY</span>
                    </label>
                  </div>

                  <div className="layer-group">
                    <span className="layer-group-title">TACTICAL OVERLAYS</span>
                    <label className="layer-option">
                      <input
                        type="checkbox"
                        checked={showMissionTrack}
                        onChange={(e) => setShowMissionTrack(e.target.checked)}
                      />
                      <span>MISSION TRACK</span>
                    </label>
                    <label className="layer-option">
                      <input
                        type="checkbox"
                        checked={showTargets}
                        onChange={(e) => setShowTargets(e.target.checked)}
                      />
                      <span>ALL TARGETS</span>
                    </label>
                    <label className="layer-option">
                      <input
                        type="checkbox"
                        checked={highRiskOnly}
                        onChange={(e) => setHighRiskOnly(e.target.checked)}
                      />
                      <span>HIGH-RISK TARGETS ONLY</span>
                    </label>
                    <label className="layer-option">
                      <input
                        type="checkbox"
                        checked={showSurveyArea}
                        onChange={(e) => setShowSurveyArea(e.target.checked)}
                      />
                      <span>SURVEY AREA (SWATH)</span>
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Fit Mission Button Floating */}
            <button
              className="btn-map-control"
              onClick={() => setFitTrigger((prev) => prev + 1)}
              title="Reset View / Fit Mission Track"
            >
              <Crosshair size={15} />
            </button>
          </div>

          {/* Bottom Coordinate Bar & Watermark Disclaimer */}
          <div className="map-bottom-telemetry-bar">
            <div className="telemetry-bar-item">
              <Navigation size={12} />
              <span>SURVEY REGION: NORTH SEA / SECTOR 04</span>
            </div>
            <div className="telemetry-bar-item">
              <Shield size={12} />
              <span>
                COORDINATE FIDELITY:{' '}
                <strong style={{ color: isSimulated ? '#f3b544' : '#4ee29b' }}>
                  {isSimulated ? 'SIMULATED LOCATION' : 'REAL GPS SENSOR METADATA'}
                </strong>
              </span>
            </div>
            <div className="telemetry-bar-item hide-on-mobile">
              <span>ACTIVE CONTACTS: {displayTargets.length}</span>
            </div>
          </div>
        </div>

        {/* ====================================================================
            3. RIGHT / BOTTOM INTELLIGENCE PANEL
            ==================================================================== */}
        <aside
          className={`target-intelligence-panel ${
            mobileDrawerOpen ? 'mobile-drawer-open' : ''
          }`}
        >
          {/* Mobile Handle to Toggle Bottom Drawer */}
          <button
            className="mobile-drawer-handle"
            onClick={() => setMobileDrawerOpen(!mobileDrawerOpen)}
          >
            <div className="drawer-notch" />
            <div className="drawer-title-row">
              <span>TARGET INTELLIGENCE BRIEF</span>
              {mobileDrawerOpen ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </div>
          </button>

          {selected ? (
            <div className="intel-panel-content">
              {/* Header with Target ID */}
              <div className="intel-panel-header">
                <div>
                  <div className="intel-eyebrow">SELECTED TARGET INTELLIGENCE</div>
                  <h2 className="intel-target-id">{selected.target_id}</h2>
                </div>
                <span
                  className={`intel-risk-badge ${
                    selected.risk_level === 'HIGH'
                      ? 'badge-high'
                      : selected.risk_level === 'MEDIUM'
                      ? 'badge-medium'
                      : 'badge-low'
                  }`}
                >
                  {selected.risk_level} RISK
                </span>
              </div>

              {/* Primary Taxonomy & Class Box */}
              <div className="intel-spec-card">
                <div className="spec-row">
                  <span className="spec-label">CLASS</span>
                  <strong className="spec-value class-accent">
                    {selected.class_name}
                  </strong>
                </div>
                <div className="spec-sub-taxonomy">
                  {selected.class_description ||
                    (selected.class_name === 'MILCO'
                      ? 'Mine-Like Contact'
                      : selected.class_name === 'NOMBO'
                      ? 'Non-Mine-like Bottom Object'
                      : selected.class_name === 'Pipeline'
                      ? 'Subsea Pipeline Infrastructure'
                      : 'Acoustic Seabed Target')}
                </div>

                <div className="spec-grid-stats">
                  <div className="spec-stat-item">
                    <small>CONFIDENCE</small>
                    <b className="stat-highlight">
                      {(selected.model_confidence * 100).toFixed(1)}%
                    </b>
                  </div>
                  <div className="spec-stat-item">
                    <small>RISK</small>
                    <b
                      style={{
                        color:
                          selected.risk_level === 'HIGH'
                            ? 'var(--risk-high)'
                            : selected.risk_level === 'MEDIUM'
                            ? 'var(--risk-medium)'
                            : 'var(--risk-low)',
                      }}
                    >
                      {selected.risk_level}
                    </b>
                  </div>
                  <div className="spec-stat-item">
                    <small>LOCATION</small>
                    <b style={{ color: '#f3b544', fontSize: '10px' }}>
                      {selected.coordinates_simulated ?? true
                        ? 'SIMULATED'
                        : 'GPS POSITION'}
                    </b>
                  </div>
                </div>

                <div className="coordinates-detail-box">
                  <span className="coord-label">COORDINATES</span>
                  <div className="coord-values">
                    <span>
                      LAT: <b>{selected.coordinates.latitude.toFixed(4)}° N</b>
                    </span>
                    <span>
                      LON: <b>{selected.coordinates.longitude.toFixed(4)}° E</b>
                    </span>
                  </div>
                  <div className="coord-badge-note">
                    <AlertTriangle size={11} />
                    <span>
                      {selected.coordinates_simulated ?? true
                        ? 'SIMULATED LOCATION (North Sea Sector 04 Demo)'
                        : 'GENUINE SENSOR / GPS TELEMETRY'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Acoustic Fingerprint Section */}
              <div className="intel-fingerprint-section">
                <div className="fingerprint-header">
                  <span className="fingerprint-title">ACOUSTIC FINGERPRINT</span>
                  <span className="fingerprint-status-tag">
                    {selected.acoustic_fingerprint ? 'AVAILABLE' : 'CALCULATING'}
                  </span>
                </div>

                {/* Radar Chart */}
                <div className="fingerprint-radar-wrap">
                  <ResponsiveContainer width="100%" height={140}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="rgba(33, 216, 238, 0.18)" />
                      <PolarAngleAxis
                        dataKey="subject"
                        tick={{
                          fill: 'var(--text-muted, #7dd3fc)',
                          fontSize: 9,
                          fontFamily: 'var(--font-mono, monospace)',
                        }}
                      />
                      <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                      <Radar
                        dataKey="value"
                        stroke="#21d8ee"
                        fill="#21d8ee"
                        fillOpacity={0.25}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>

                {/* Micro parameters */}
                <div className="fingerprint-micro-grid">
                  <div className="micro-param">
                    <small>SHAPE</small>
                    <b>
                      {(selected.acoustic_fingerprint?.shape ?? 0.68).toFixed(2)}
                    </b>
                  </div>
                  <div className="micro-param">
                    <small>INTENSITY</small>
                    <b>
                      {(selected.acoustic_fingerprint?.intensity ?? 0.74).toFixed(2)}
                    </b>
                  </div>
                  <div className="micro-param">
                    <small>SHADOW</small>
                    <b>
                      {selected.acoustic_fingerprint?.shadow_present ?? true
                        ? 'CONFIRMED'
                        : 'ABSENT'}
                    </b>
                  </div>
                  <div className="micro-param">
                    <small>DIVERGENCE</small>
                    <b>
                      {(
                        selected.acoustic_fingerprint?.seabed_divergence ?? 0.52
                      ).toFixed(2)}
                    </b>
                  </div>
                </div>
              </div>

              {/* Target Actions & Navigation */}
              <div className="intel-actions-cluster">
                <button
                  className="btn-intel-action primary-cyan"
                  onClick={() => onNavigate && onNavigate('Sonar Analysis')}
                >
                  <Eye size={13} />
                  <span>ANALYZE IN SONAR VIEW</span>
                  <ArrowUpRight size={13} />
                </button>

                <button
                  className="btn-intel-action secondary-glass"
                  onClick={() => onNavigate && onNavigate('Target Intelligence')}
                >
                  <Crosshair size={13} />
                  <span>INSPECT IN TARGET CATALOG</span>
                  <ArrowUpRight size={13} />
                </button>

                {onReviewTarget && (
                  <div className="operator-review-row">
                    <button
                      className="btn-review-mini confirm"
                      onClick={() => onReviewTarget('CONFIRMED')}
                      title="Confirm target as verified contact"
                    >
                      <CheckCircle2 size={12} /> CONFIRM
                    </button>
                    <button
                      className="btn-review-mini natural"
                      onClick={() => onReviewTarget('MARKED NATURAL')}
                      title="Mark as benign natural seabed feature"
                    >
                      <Shield size={12} /> NATURAL
                    </button>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="intel-empty-state">
              <div className="empty-icon-wrap">
                <Crosshair size={28} />
              </div>
              <h3>SELECT CONTACT</h3>
              <p>
                Click any contact marker on the tactical map to inspect multi-spectral acoustic telemetry, risk scoring, and classification.
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

export default AcousticMap;
