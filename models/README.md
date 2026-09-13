# SONARIS-X model weights

## Current status: DEMO / UNTRAINED — CUSTOM WEIGHTS NOT LOADED

No sonar-trained weights ship with this prototype. The detector runs:

1. COCO `yolov8n.pt` fallback (auto-downloaded on first run; **NOT sonar-trained**), plus
2. the classical sonar saliency proposer (`src/detection/saliency.py`, **DEMO/UNTRAINED**).

The UI banner always shows this status. Candidates are labeled `sonar_candidate`
or COCO names and classified only as UNKNOWN ANOMALY / LIKELY NATURAL / REVIEW —
never as sonar object names.

## Adding custom sonar weights

1. Train with `scripts/train.py` on a labeled YOLO-format sonar dataset, or
   place your weights at `models/sonar_yolov8.pt`.
2. Pass the path explicitly:
   `SonarisXPipeline(model_path="models/sonar_yolov8.pt")`
   or select it in the UI when the model selector is available.
3. The detector advertises sonar classes **only** from the loaded weights'
   `model.names` intersected with the sonar taxonomy
   (`Ghost Net, Pipeline, Cylinder, Shipwreck, Debris, Natural Rock`).
4. Only then may KNOWN OBJECT predictions show sonar object names.

Do not rename COCO outputs to sonar names. Do not claim production readiness
without measured precision/recall on held-out sonar data.
