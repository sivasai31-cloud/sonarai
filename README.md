# SONARIS-X

**Offline-first AI-powered Side-Scan Sonar Intelligence System**

Detecting anthropogenic underwater debris and anomalies from side-scan sonar imagery.

## Overview

SONARIS-X is a complete pipeline for analyzing side-scan sonar imagery:

RAW SONAR → DATA INGESTION → QUALITY ASSESSMENT → PREPROCESSING → MULTI-SCALE DETECTION → TARGET FINGERPRINT → LOCAL SEABED CONTEXT → FALSE-POSITIVE FILTER → KNOWN/UNKNOWN → CONFIDENCE → RISK → EXPLAINABILITY → GEOLOCATION → TARGET CATALOG → MAP → REPORT

## Architecture

```
sonaris-x/
├── app.py                    # Streamlit main application
├── src/
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── data/
│   │   └── ingestion.py      # Data ingestion layer
│   ├── preprocessing/
│   │   ├── quality.py        # Image quality assessment
│   │   └── preprocess.py     # Sonar preprocessing pipeline
│   ├── detection/
│   │   └── detector.py       # Multi-scale YOLO detector
│   ├── fingerprint/
│   │   └── fingerprint.py    # Acoustic/image fingerprint extraction
│   ├── context/
│   │   └── target_context.py # Adaptive target context analysis
│   ├── anomaly/
│   │   ├── false_positive_filter.py  # False positive filtering
│   │   └── classification.py         # Known/Unknown classification
│   ├── risk/
│   │   └── risk_engine.py    # Evidence-based risk scoring
│   ├── explainability/
│   │   └── explainer.py      # Measurable explainability
│   ├── geolocation/
│   │   └── geolocation.py    # Coordinate extraction & calculation
│   ├── reporting/
│   │   └── report_generator.py  # CSV/JSON/text report generation
│   └── validation/
│       ├── dataset_validation.py  # Dataset analysis tools
│       └── system_diagnostics.py  # Performance diagnostics
├── scripts/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Baseline comparison
│   ├── ablation.py           # Ablation study
│   ├── small_target.py       # Small target experiment
│   ├── analyze_dataset.py    # Dataset analysis
│   └── split_dataset.py      # Leakage-safe splitting
├── tests/
│   └── test_pipeline.py      # Complete test suite
├── models/                   # Model weights
├── data/                     # Data directories
├── outputs/                  # Generated outputs
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Model status: DEMO / UNTRAINED — CUSTOM WEIGHTS NOT LOADED

No sonar-trained weights ship with this prototype. The detector combines a
COCO `yolov8n.pt` fallback (**NOT sonar-trained**) with a classical sonar
saliency proposer (`src/detection/saliency.py`, **DEMO/UNTRAINED**). The app
banner always shows this. Candidates are labeled `sonar_candidate` (or COCO
names) and classified only as UNKNOWN ANOMALY / LIKELY NATURAL / REVIEW —
never as sonar object names. See `models/README.md` for adding real weights.

## Demo scenarios (SYNTHETIC DEMO DATA)

Seven deterministic synthetic scenarios (`src/demo/synthetic.py`): Ghost Net,
Pipeline, Cylinder, Shipwreck, Natural Rock / Seabed, Unknown Anomaly,
Low-quality / dropout. Each carries labeled synthetic ground truth shown in
the demo panel for reference — never injected as detections. Generate files
via Analysis Console → Dataset Explorer → GENERATE DEMO SET (`data/demo/`).

## Pages

Analysis Console (landing) · Target Catalog · Tactical Map [SIMULATED] ·
Mission Dashboard · Mission Reports (JSON/CSV/TXT) · Review Queue ·
Dataset Explorer · Training Center · Model Validation (baseline, ablation,
small-target, domain probe) · Failure Analysis (real pipeline outputs) ·
System Diagnostics (environment + PIPELINE SELF TEST).

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### Start the Application

```bash
streamlit run app.py
```

The application opens directly to the **Analysis Console**.

### Use the Application

1. **Upload** a sonar image (PNG, JPG, TIFF) via the left panel
2. Click **RUN ANALYSIS** to process through the complete pipeline
3. Or click **START DEMO MISSION** for synthetic demo data
4. Switch between **ORIGINAL**, **PREPROCESSED**, and **DETECTION VIEW**
5. Click on detections to see fingerprint, context analysis, and risk

### Demo Mode

Click **START DEMO MISSION** in the Analysis Console. The system will:
- Generate synthetic sonar data (clearly labeled)
- Run the full pipeline
- Display all analysis results

## Features

### Analysis Console
- Image upload with validation
- Preprocessing controls (CLAHE, denoising, shadows)
- Sonar viewer with multiple view modes
- Detection display with detailed inspection

### Mission Dashboard
- Real-time statistics from actual processing
- Risk distribution charts
- Class distribution charts
- Processing timeline

### Target Catalog
- Searchable target database
- Filter by class, risk, confidence, status
- Click to inspect detailed information

### Tactical Map
- Visual display of detection locations
- Color-coded risk levels
- Simulated locations clearly marked

### Mission Reports
- Generate JSON, CSV, and text reports
- Each target includes classification, confidence, risk, evidence, coordinates

### Model Validation
- Baseline comparison
- Ablation study results
- Small target experiment results

### Failure Analysis
- Common failure cases documented
- Solutions for each failure type

### System Diagnostics
- Model size, inference time, memory usage
- Optimization path for edge deployment

## Dataset Setup

### Supported Formats
- Images: PNG, JPG/JPEG, TIFF, BMP
- Metadata: JSON, CSV, EXIF
- Annotations: YOLO format, COCO format

### Adding Custom Sonar Data
1. Place images in `data/raw/`
2. Place metadata in `data/` (JSON/CSV)
3. Annotations in YOLO format in `data/labels/`
4. Create `data/data.yaml` with class definitions

### Dataset Adapters
- `scripts/analyze_dataset.py` - Analyze any dataset
- `scripts/split_dataset.py` - Leakage-safe splitting
- Each adapter documents source, license, classes, format

## Training

```bash
python scripts/train.py
```

Training configuration is in `scripts/train.py`:
- Reproducible with random seed
- Records dataset, model, epochs, batch size, learning rate
- Sonar-specific augmentation (speckle noise, dropout, contrast)
- Leakage-safe splitting by mission/survey/location

## Evaluation

```bash
python scripts/evaluate.py
```

Compares baseline detector with SONARIS-X using:
- Precision, Recall, F1
- mAP50, mAP50-95
- False positives
- Inference time

## Ablation Study

```bash
python scripts/ablation.py
```

Experiments A-F:
- A: Baseline
- B: Baseline + preprocessing
- C: Baseline + tiling
- D: Baseline + fingerprint/context
- E: Baseline + false-positive filtering
- F: Full SONARIS-X

## Limitations

- **Demo Mode**: When no trained sonar weights are available, the system uses synthetic detections clearly labeled as [DEMO/UNTRAINED MODEL].
- **Coordinates**: Without genuine GPS metadata, coordinates are marked as [SIMULATED].
- **Training**: Full training requires properly formatted YOLO dataset with annotations.
- **Edge Deployment**: Model export to ONNX is available but requires actual testing.

## Scientific Integrity Statement

SONARIS-X adheres to strict scientific integrity:
- All metrics are computed from actual data processing
- Simulated coordinates are clearly labeled [SIMULATED]
- Demo data is clearly labeled [SYNTHETIC DEMO DATA]
- Untrained models are labeled [DEMO/UNTRAINED MODEL]
- No fabricated accuracy, precision, recall, F1, or mAP values
- No fake coordinates or inflated performance metrics

## License

See dataset-specific licenses. Individual datasets documented in their adapters.

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Ultralytics YOLOv8
- OpenCV 4.7+
- NumPy, SciPy, Scikit-learn
- Streamlit, Flask
- Pillow, PyYAML, Pandas

See `requirements.txt` for complete list.
