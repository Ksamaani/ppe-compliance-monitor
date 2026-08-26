# Data & provenance

Two separate sources of data are used, for two separate purposes. Keeping them apart matters:
the training data is labelled and drives the model; the sample assets are unlabelled and only
exist to demonstrate the pipeline on imagery the model has never seen.

---

## 1. Training dataset — Construction Site Safety (Roboflow via Kaggle)

| | |
|---|---|
| **Name** | Construction Site Safety Image Dataset |
| **Kaggle** | [`snehilsanyal/construction-site-safety-image-dataset-roboflow`](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) |
| **Upstream origin** | Roboflow Universe — *Construction Site Safety* project |
| **Licence** | CC BY 4.0 |
| **Annotation format** | YOLO (`class cx cy w h`, normalised 0–1) |
| **Size** | 2,801 images — train 2,605 · valid 114 · test 82 (measured, version 3) |
| **Access** | Anonymous, no API key: `kagglehub.dataset_download("snehilsanyal/construction-site-safety-image-dataset-roboflow")` |

### Classes (10)

| id | name | id | name |
|---|---|---|---|
| 0 | `Hardhat` | 5 | `Person` |
| 1 | `Mask` | 6 | `Safety Cone` |
| 2 | `NO-Hardhat` | 7 | `Safety Vest` |
| 3 | `NO-Mask` | 8 | `machinery` |
| 4 | `NO-Safety Vest` | 9 | `vehicle` |

### Measured class distribution

Counted directly from the label files by `ppe_monitor.dataset.describe()` rather than taken from
the dataset card — 38,352 annotated instances in total.

| class | train | valid | test |
|---|---:|---:|---:|
| `Person` | 9,532 | 166 | 174 |
| `machinery` | 5,247 | 55 | 44 |
| `NO-Safety Vest` | 3,962 | 106 | 90 |
| `Safety Cone` | 3,366 | 44 | 92 |
| `Hardhat` | 3,145 | 79 | 110 |
| `NO-Mask` | 3,097 | 74 | 79 |
| `Safety Vest` | 3,033 | 41 | 61 |
| `NO-Hardhat` | 2,317 | 69 | 41 |
| `Mask` | 1,651 | 21 | 28 |
| `vehicle` | 1,545 | 42 | 41 |
| **total** | **36,895** | **697** | **760** |

`NO-Hardhat` — the class this project cares about most — has 2,317 training instances, which is
enough to learn, but only **41 test instances**. Every point of recall on that class is worth
~2.4%, so the test-split figures for it carry real uncertainty. That is stated here so the
evaluation numbers are not over-read.

### Why the paired classes matter

The dataset labels both compliance and non-compliance explicitly. A violation is therefore a
**positive detection** of `NO-Hardhat` / `NO-Mask` / `NO-Safety Vest` — not the absence of a
`Hardhat` box. This distinction is what makes the error analysis in
[`EVALUATION.md`](EVALUATION.md) meaningful:

- A **false negative** on `NO-Hardhat` is an unprotected worker the system waved through.
- A **false positive** on `NO-Hardhat` is a supervisor walking over to a compliant worker.

Had violations been modelled as "no `Hardhat` box found", every missed detection of any kind
would have silently become a violation, and the two failure modes would be impossible to separate.

### Known caveats

Recorded up front so the evaluation is read in context:

- **Small held-out splits.** 114 validation and 82 test images are enough for model selection and
  honest reporting, but too small for hyperparameter tuning — per-class metrics on rare classes
  swing on a handful of instances.
- **Class imbalance.** `Person` alone is a quarter of all training instances, and there are six
  times as many `Person` boxes as `Mask` boxes. Rare classes will score noticeably lower, and
  that is a property of the data rather than of the training run.
- **Domain skew.** Images skew towards daytime outdoor sites; night, rain and indoor scenes are
  under-represented, so real deployment would need site-specific data collection.

---

## 2. Sample assets — Wikimedia Commons

Fetched by [`scripts/fetch_assets.py`](../scripts/fetch_assets.py) via the MediaWiki API. No API
key or account is required. Every file's source page, author and licence is written to
[`logs/01_assets.log`](../logs/01_assets.log) at fetch time.

### Images

Eight CC-licensed construction-site photographs, searched by term and filtered to ≥1200 px wide.
They were chosen to span the failure modes the model has to survive: helmets worn and not worn,
hi-vis vests, workers at varying distance from the camera, and partial occlusion.

### Video clip

| | |
|---|---|
| **Source** | *Building construction Moira Close Broadwater Farm Haringey 2025 16.webm* |
| **Commons page** | <https://commons.wikimedia.org/wiki/File:Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_16.webm> |
| **Author** | Acabashi |
| **Licence** | CC BY-SA 4.0 |
| **Source properties** | 1920×1080, VP9, ~29.97 fps, 362 s |
| **Working window** | 236 s → 270 s (34 s) |
| **Output** | 1280×720, H.264, 15 fps, no audio (~9 MB) |

Reproduced with:

```bash
ffmpeg -y -ss 236 -t 34 -i source.webm \
  -vf "scale=1280:-2,fps=15" \
  -c:v libx264 -preset veryfast -crf 26 \
  -pix_fmt yuv420p -movflags +faststart -an \
  data/videos/site_clip.mp4
```

**Why this window.** The source is a timelapse with zooms and crossfade transitions, which would
wreck tracking. 236–270 s is a static, zoomed shot of a formwork deck where two to four workers
continuously walk across frame carrying materials. Workers are large in frame, hard hats of
several colours are clearly visible, the preceding scene transition has finished before 235 s,
and the clip ends before the camera zooms out at ~272 s. Stable camera plus large subjects is
what object tracking needs to produce meaningful track IDs.
