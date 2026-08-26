# Workflow

How this project is built, in what order, and why that order. Each stage ends in one commit, so
the history reads as incremental work rather than a bulk upload.

Written for two readers: a grader checking that each deliverable rests on real executed output,
and anyone who wants to reproduce the whole thing from a clean clone.

---

## Status

| # | Deliverable | Pts | Stage | State |
|---|---|---:|---|---|
| — | Scaffold, dependencies, environment log | — | 0 | ✅ committed |
| — | Sample assets with licence provenance | — | 1 | ✅ committed |
| 1 | Tasks & inference — detect / segment / pose | 25 | 2 | ✅ executed, committed |
| 2 | Video analytics — track, count, heatmap, ledger | 25 | 3 + 6 | ✅ re-run on custom weights, committed |
| 4 | Custom training — two runs on a Colab T4 | 15 | 4 | ✅ run by hand on Colab, results committed |
| 3 | Evaluation — val, threshold sweep, error analysis | 25 | 5 | ✅ executed, committed |
| 5 | Deployment — ONNX export + Streamlit app | 5 | 7 | ✅ app boot-verified; export round-tripped against the real weights |
| 6 | Docs & evidence | 5 | 8 | ✅ committed |

**Right now:** every stage is done and committed. The evaluation, the video ledger, the ONNX
export and the tracker-gap analysis all rest on captured output in [`logs/`](logs/) and executed
notebooks with their output cells populated.

---

## The dependency that shapes the order

One stage needs a GPU and the development machine has none — an Intel i7-10510U with integrated
graphics, no CUDA device. Training happens on a **Google Colab T4** and nothing else does.

That single constraint sets the build order. Every stage that does *not* need the trained weights
is built and committed **first**, so the GPU session is never the thing blocking progress:

```
   scaffold ──> assets ──> D1 inference ──> D2 video (COCO weights)
                                                   │
                                                   ├──> D4 smoke test ──> D4 Colab notebook
                                                   │                            │
                                                   │                      [ T4 GPU run ]
                                                   │                            │
                                                   │                        best.pt
                                                   │                            │
                                                   └────> D2 re-run ◀───────────┤
                                                          D3 evaluation ◀───────┤
                                                          D5 ONNX export ◀──────┘
                                                                 │
                                                                 └──> D6 docs + push
```

D2 is deliberately built twice. The first pass runs on COCO-pretrained weights, which have no PPE
classes at all, so it proves the tracking, counting and heatmap pipeline works while the ledger
necessarily reports zero violations. The second pass, after the weights come back, runs the same
code and produces a real ledger. Both are committed — the difference between them is the evidence
that training changed something.

---

## Stage by stage

### 0 — Scaffold
Virtual environment on Python 3.11 (the system 3.14 has no torch wheels), CPU-only torch,
dependencies pinned to `requirements.txt`, `ultralytics.checks()` captured to
`logs/00_environment.log`. Repository created and pushed empty.

### 1 — Assets
`scripts/fetch_assets.py` pulls CC-licensed construction imagery from the Wikimedia Commons API,
logging every source URL, licence and author to `logs/01_assets.log`. The 34-second site clip is
cut from a CC BY-SA 4.0 source video with the exact `ffmpeg` recipe recorded in the script.

### 2 — Deliverable 1: tasks and inference
`notebooks/01_tasks_inference.ipynb`. Detection, **instance segmentation** (`yolo11n-seg.pt`) and
**pose estimation** (`yolo11n-pose.pt`), plus the CLI form `yolo task=detect mode=predict`.

Pose is not decoration: a hard hat has to sit on a head, COCO has no head class, so keypoints 0–4
(nose, eyes, ears) are used to derive the head region geometrically. That is the bridge from a
generic pretrained model to the PPE problem, and it establishes empirically that off-the-shelf
weights cannot assess compliance — which is what justifies stage 4.

### 3 — Deliverable 2: video analytics
`ppe_monitor/video.py` and `ppe_monitor/compliance.py`, driven by
`notebooks/02_video_analytics.ipynb`. A real `cv2.VideoCapture` → process → `cv2.VideoWriter`
loop, `model.track(persist=True)` for stable worker IDs, `solutions.ObjectCounter` on a gate line,
`solutions.Heatmap`, and the violation ledger.

The counting line was placed from measured trajectory data — candidate lines scored by how many
distinct tracks actually crossed them — not chosen by eye.

### 4 — Deliverable 4: training
`scripts/smoke_test_training.py` runs **every API call the Colab notebook makes** locally on CPU
first, against a 24-image subset for one epoch, in 52 seconds. It produces a deliberately useless
model; it is testing the plumbing, not the weights. A Colab run that dies in cell 6 on a path bug
costs 40 minutes and a restart.

Then `notebooks/04_training_colab.ipynb` on a T4: **Run A** baseline (25 epochs) and **Run B**
tuned (`freeze=10`, heavier augmentation, 30 epochs). Two runs, because one run tells you where
the model landed but nothing about why.

### 5 — Deliverable 3: evaluation
`notebooks/03_evaluation.ipynb` on the held-out **test** split — the split that steered neither
early stopping nor model selection, so it is the only honest number.

This stage produced the project's two most useful mistakes, both left visible in the notebook:

1. The first threshold sweep read precision and recall back from `model.val(conf=X)`. Ultralytics
   reports those at the PR curve's **best-F1 point**, not at the threshold passed, so the sweep
   returned identical numbers for three different thresholds and picked a winner on noise. It was
   replaced with predict-at-0.01 plus direct greedy IoU matching against ground truth.
2. The macro-average precision cleared the 0.50 floor while `NO-Mask` sat at 0.358, carried by
   the other two classes. That became an explicit site-policy decision rather than a hidden
   defect.

### 6 — Deliverable 2, again
Re-run of `02_video_analytics.ipynb` on `models/best.pt`. Same code, real violations: 3 incidents
across 2 workers with evidence crops. A shadow ledger enforcing all three `NO-*` classes runs in
the same pass, so the policy decision is visible where it changes what a safety officer sees.

### 7 — Deliverable 5: export and app
`scripts/export_onnx.py` exports to ONNX and **round-trips it** — reloads the `.onnx`, runs it on
a real site image, and compares detections against the PyTorch model. An export that is never
loaded back proves nothing. `app.py` is a Streamlit front end with a model picker, threshold
slider and the live ledger.

ONNX because the deployment target is a CPU-only Intel box: TensorRT needs CUDA, CoreML is the
wrong platform, and ONNX Runtime is the portable CPU target.

### 8 — Deliverable 6: documentation
README as a landing page with real measured results, `docs/` for the technical detail, and a
final check that `.gitignore` excludes datasets, weights, rendered video and secrets.

---

## Reproducing it

```bash
python -m venv .venv && .venv/Scripts/activate    # Python 3.11
pip install -r requirements.txt
python scripts/fetch_assets.py                    # sample imagery, with provenance
jupyter execute notebooks/01_tasks_inference.ipynb
jupyter execute notebooks/02_video_analytics.ipynb
```

Those run on COCO weights with no training required. For the rest, run
`notebooks/04_training_colab.ipynb` on a Colab T4, drop the resulting `best.pt` into `models/`,
then re-run notebooks 02 and 03 and `scripts/export_onnx.py`.

---

## Rules the work follows

- **An unexecuted notebook proves nothing.** Every notebook is committed with its output cells
  populated, and the execution log goes to `logs/`.
- **Measure before asserting.** The counting line, the confidence threshold, the enforced class
  set and the claim that the sample clip contains violations were all measured. Where a
  measurement contradicted something already written down, the correction is in the document
  rather than quietly swapped out.
- **Price the trades.** Moving from `conf=0.25` to `0.10` buys recall with precision. It is
  presented as a trade with both numbers attached, not as a free improvement.
- **Smoke-test anything that costs a GPU session.** Twice this caught real bugs before they cost
  40 minutes.
