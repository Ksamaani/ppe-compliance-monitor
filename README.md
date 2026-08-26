# ⛑ PPE Compliance Monitor

**Turning a construction-site camera feed into an auditable violation ledger — not just boxes on a video.**

An end-to-end computer-vision system built with [Ultralytics YOLO](https://docs.ultralytics.com/):
detection, instance segmentation, pose estimation, multi-object tracking, region analytics and a
custom-trained PPE model, wired into one pipeline that produces something a safety officer can
act on.

Capstone project for **Computer Vision for Developers with Ultralytics** —
[SDAIA Academy](https://github.com/SDAIAAcademy), cohort **24–28 August 2026**.

---

## The problem

Construction sites lose lives to preventable head injuries. Supervisors cannot watch every camera,
and a system that merely draws boxes on a video has handed them *another screen to watch*. What is
actually actionable is a short, ranked list of **incidents**: which worker was non-compliant, for
how long, and a photograph proving it.

## What the system does

| # | Stage | Ultralytics API | Output |
|---|---|---|---|
| 1 | Locate workers and PPE | `model.predict` | boxes, masks, keypoints |
| 2 | Derive the head region | pose keypoints 0–4 | where a helmet must sit |
| 3 | Follow each worker | `model.track(persist=True)` | stable per-worker identity |
| 4 | Count site entries | `solutions.ObjectCounter` | crossings over a gate line |
| 5 | Map activity density | `solutions.Heatmap` | where work concentrates |
| 6 | **Log violations** | `ppe_monitor.compliance` | debounced incident CSV + evidence crops |
| 7 | Measure it | `model.val` | mAP, per-class P/R, confusion matrix |
| 8 | Ship it | `model.export` + Streamlit | ONNX model behind a web app |

### The design axis: missed violations cost more than false alarms

A missed violation is a head injury. A false alarm costs a supervisor five seconds. Every
threshold decision here is made on that asymmetry, and
[`docs/EVALUATION.md`](docs/EVALUATION.md) shows the working rather than accepting the library
default of `conf=0.25`.

A violation is also a **positive detection** of a `NO-Hardhat` / `NO-Mask` / `NO-Safety Vest`
class — never the absence of a box. The model has to actively assert non-compliance, which is
what makes the false-negative analysis meaningful. Had "no `Hardhat` found" counted as a
violation, every missed detection of any kind would silently become one.

### Why a ledger rather than per-frame alerts

Raw detections flicker — a missing hard hat is seen on frame 41, lost on 42, seen again on 43.
Logged directly that is three "incidents" for one event, and a supervisor learns to ignore the
system within a week. [`ppe_monitor/compliance.py`](ppe_monitor/compliance.py) runs a state
machine per `(track_id, class)`: a violation must persist **5 consecutive frames** to open an
incident and be clear for **15** to close one, with incidents backdated to the true start of the
streak. At 15 fps that is 0.33 s to open, 1.0 s to close — long enough to survive a worker
passing behind a stack of material, short enough to catch someone walking briskly through frame.

The behaviour is pinned by [12 unit tests](tests/test_compliance.py) in both directions: a
two-frame blip must **not** become an incident, and a one-frame dropout must **not** split one
incident into two.

---

## Results

| Deliverable | Evidence |
|---|---|
| Tasks & inference | [`01_tasks_inference.ipynb`](notebooks/01_tasks_inference.ipynb) — detect + `-seg` + `-pose` on 8 real site photos |
| Video analytics | [`02_video_analytics.ipynb`](notebooks/02_video_analytics.ipynb) — 510 frames, 6 tracked workers, gate counting, heatmap |
| Evaluation | [`03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) — ⏳ pending the training run |
| Custom training | [`04_training_colab.ipynb`](notebooks/04_training_colab.ipynb) — ⏳ pending the T4 run |
| Deployment | [`app.py`](app.py) + [`scripts/export_onnx.py`](scripts/export_onnx.py) |

**Model metrics:** ⏳ filled from the executed training and evaluation notebooks. Deliberately
left blank rather than populated with plausible-looking numbers.

---

## Repository layout

```
ppe-compliance-monitor/
├── app.py                          # Streamlit app: image + video, live ledger
├── ppe_monitor/
│   ├── config.py                   # class semantics, thresholds, paths
│   ├── compliance.py               # debounced incident state machine
│   ├── video.py                    # OpenCV capture -> track -> annotate -> write
│   └── dataset.py                  # dataset download, split discovery, data.yaml
├── notebooks/
│   ├── 01_tasks_inference.ipynb    # detection, segmentation, pose, head ROIs
│   ├── 02_video_analytics.ipynb    # tracking, counting, heatmap, ledger
│   ├── 03_evaluation.ipynb         # model.val, threshold sweep, error analysis
│   └── 04_training_colab.ipynb     # fine-tuning on a Colab T4
├── scripts/
│   ├── fetch_assets.py             # CC-licensed imagery, with provenance logging
│   ├── smoke_test_training.py      # proves the Colab pipeline on CPU first
│   └── export_onnx.py              # ONNX export + round-trip verification
├── tests/test_compliance.py        # ledger debounce behaviour
├── data/                           # sample site photos and video clip
├── outputs/                        # annotated evidence frames, CSVs
├── logs/                           # captured run logs
└── docs/                           # DATASET · TRAINING · EVALUATION · DEPLOYMENT
```

---

## Quickstart

**Prerequisites:** Python 3.11, ~2 GB disk. No GPU needed for anything except training.

```bash
git clone https://github.com/Ksamaani/ppe-compliance-monitor.git
cd ppe-compliance-monitor
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Fetch the sample imagery and video (CC-licensed, downloaded from Wikimedia Commons; every source
URL and licence is logged to `logs/01_assets.log`):

```bash
python scripts/fetch_assets.py
```

Run the tests:

```bash
python tests/test_compliance.py
```

Then open the notebooks in order. `01` and `02` run out of the box on COCO-pretrained weights,
which download automatically on first use.

### Getting the trained model

`models/best.pt` is **not committed** — trained weights are build output, not source. To obtain it:

1. Open [`notebooks/04_training_colab.ipynb`](notebooks/04_training_colab.ipynb) in Colab.
2. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all** (~40–60 min).
3. The final cell downloads a zip; unpack `best.pt` into `models/`.

The dataset downloads itself inside the notebook via `kagglehub` — anonymously, no API key or
Kaggle account required.

### Running the app

```bash
streamlit run app.py
```

Without `models/best.pt` the app still runs on COCO weights and says plainly that it cannot
assess compliance — rather than silently reporting everyone as compliant, which is the dangerous
failure mode for a safety tool.

---

## Expected output

- **Notebook 01** → `outputs/01_*.jpg`: annotated detection, segmentation, pose and head-ROI images
- **Notebook 02** → `outputs/02_track.mp4`, `02_gate.mp4`, `02_heatmap.mp4`, plus
  `02_gate_counts.csv` (per-frame crossings), `02_violations.csv` (the ledger) and sampled stills
- **Notebook 03** → confusion matrix, PR/F1 curves, per-class metrics, the selected threshold
- **Notebook 04** → `best.pt`, `results.csv`, training curves, per-run `metrics.json`
- **`export_onnx.py`** → `models/best.onnx` plus a PyTorch-vs-ONNX detection diff

Rendered video is gitignored as generated output; stills, CSVs and executed notebooks are
committed as evidence.

---

## Technical documentation

| Document | Contents |
|---|---|
| [`docs/DATASET.md`](docs/DATASET.md) | Both data sources, measured class distribution, licences, why the video window was chosen |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Training design, the A/B knob table with reasoning, the CPU smoke test |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Metrics, threshold selection, error analysis |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Export target justification, app notes, production gaps |

**Stack:** Python 3.11 · Ultralytics 8.4.129 · PyTorch 2.13 (CPU) · OpenCV 5.0 · Streamlit 1.62 ·
ONNX Runtime 1.29. Exact pins in [`requirements.txt`](requirements.txt); the environment the
committed outputs were produced in is captured in [`logs/00_environment.log`](logs/00_environment.log).

---

## Attribution

Completed under the training program **Computer Vision for Developers with Ultralytics**,
delivered by **SDAIA Academy** — cohort **24–28 August 2026**.

Program GitHub: <https://github.com/SDAIAAcademy>

**Data credits**

- Training data: [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
  (Roboflow Universe, CC BY 4.0)
- Sample photographs: Wikimedia Commons, CC BY-SA / public domain — per-file credits in
  [`logs/01_assets.log`](logs/01_assets.log)
- Sample video: *Building construction Moira Close Broadwater Farm Haringey 2025 16* by
  **Acabashi**, CC BY-SA 4.0
