# PPE Compliance Monitor

**Turning a construction-site camera feed into an auditable violation ledger — not just boxes on a video.**

An end-to-end computer-vision system built with [Ultralytics YOLO](https://docs.ultralytics.com/),
submitted as the capstone for **Computer Vision for Developers with Ultralytics** —
[SDAIA Academy](https://github.com/SDAIAAcademy), cohort **24–28 August 2026**.

> 🚧 **Status: in progress.** Milestones land as incremental commits; this README grows with them.

## The problem

Construction sites lose lives to preventable head injuries. Supervisors cannot watch every camera,
and a system that merely draws boxes on a video does not help them — it produces something else to
watch. What a safety officer can actually act on is a short, ranked list of **incidents**: who was
non-compliant, for how long, and a photograph proving it.

## What this system does

1. **Detects** PPE and workers — hard hats, masks, safety vests, and their `NO-*` counterparts —
   with a YOLO11 detector fine-tuned on real construction-site data.
2. Runs **instance segmentation** and **pose estimation** as tasks beyond plain detection, using
   pose keypoints to derive the head region where a helmet must sit.
3. **Tracks** each worker across frames, counts entries across a site-entrance line, and builds a
   **heatmap** of where activity concentrates.
4. Converts the flickering detection stream into a **debounced violation ledger** — a CSV of
   incidents with duration, severity and a cropped evidence image each.
5. Is **evaluated** on a held-out split with mAP, per-class precision/recall and a confusion
   matrix, with the shipped confidence threshold chosen for the asymmetric cost of errors.
6. **Deploys** as an ONNX export behind a small Streamlit app.

## The design axis: missed violations cost more than false alarms

A missed violation is a head injury. A false alarm costs a supervisor five seconds. Every
threshold decision in this project is made on that asymmetry, and the evaluation notebook shows
the working rather than accepting the default `conf=0.25`.

Note also that a violation here is a **positive detection** of a `NO-Hardhat` / `NO-Mask` /
`NO-Safety Vest` class — never the absence of a box. The model has to actively assert
non-compliance, which is what makes the false-negative analysis meaningful.

## Repository layout

```
ppe-compliance-monitor/
├── app.py                        # Streamlit demo: image/video -> detections + ledger
├── ppe_monitor/
│   ├── config.py                 # class semantics, thresholds, paths
│   ├── compliance.py             # per-track debounced incident state machine
│   └── video.py                  # OpenCV capture -> track -> annotate -> write
├── notebooks/
│   ├── 01_tasks_inference.ipynb  # detection + segmentation + pose
│   ├── 02_video_analytics.ipynb  # tracking, counting, heatmap, violation ledger
│   ├── 03_evaluation.ipynb       # model.val, threshold sweep, error analysis
│   └── 04_training_colab.ipynb   # custom fine-tuning on a T4 GPU
├── scripts/                      # asset fetch, ONNX export, training smoke test
├── data/                         # sample site images and video clip
├── outputs/                      # annotated evidence frames, violations.csv
├── logs/                         # captured run logs
└── docs/                         # DATASET, TRAINING, EVALUATION, DEPLOYMENT
```

## Quickstart

```bash
git clone https://github.com/Ksamaani/ppe-compliance-monitor.git
cd ppe-compliance-monitor
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
python scripts/fetch_assets.py
```

Then open the notebooks in order. Full per-stage instructions, dataset provenance and expected
output are documented as each milestone lands.

## Attribution

Completed under the training program **Computer Vision for Developers with Ultralytics**,
delivered by **SDAIA Academy** — cohort 24–28 August 2026.
Program GitHub: <https://github.com/SDAIAAcademy>

Sample imagery and video are CC-licensed works from Wikimedia Commons; every source URL, author
and licence is recorded in [`logs/01_assets.log`](logs/01_assets.log) and
[`docs/DATASET.md`](docs/DATASET.md).
