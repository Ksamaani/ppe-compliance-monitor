# Evaluation

How the model is measured, what the numbers mean for a safety system, and how the shipped
thresholds were chosen. Produced by [`notebooks/03_evaluation.ipynb`](../notebooks/03_evaluation.ipynb).

## What is measured, and on what

`model.val(split="test")` on the **held-out test split** — 82 images, 760 instances.

The split choice matters. The *validation* split steered early stopping and model selection during
training, so quoting metrics from it flatters the model. The test split was used for neither, so
it is the only honest number.

| Metric | What it answers |
|---|---|
| **mAP50** | Are the boxes roughly in the right place? (IoU > 0.50) |
| **mAP50-95** | Are they *tightly* in the right place? (averaged over IoU 0.50:0.95) |
| **Precision** | Of the alarms raised, how many were real? |
| **Recall** | Of the real violations, how many were caught? |
| **Confusion matrix** | *What* was confused with what — including with background |

mAP50-95 is the number worth quoting. Localisation quality matters here beyond the usual reasons:
the box is cropped and stored as photographic evidence of a violation, and a loose box that clips
the worker's head is a weaker record.

---

## The asymmetry that drives every threshold decision

| | Consequence |
|---|---|
| **False negative** on `NO-Hardhat` | A worker without a hard hat is waved through. This is the failure mode that kills people. |
| **False positive** on `NO-Hardhat` | A supervisor walks over and finds a compliant worker. Five seconds wasted. |

These costs are nowhere near equal, so **F1 is the wrong objective** — it weights precision and
recall identically. This project maximises **recall on the violation classes** instead.

But not without limit. A monitor that cries wolf constantly gets muted, and a muted monitor turns
every future true positive into a miss as well. So the decision rule is:

> **Maximise mean recall across `NO-Hardhat`, `NO-Mask` and `NO-Safety Vest`, subject to mean
> precision across those classes staying at or above 0.50** — at least half of the alarms raised
> must be real.

The precision floor is a judgement call, and it is the one number here that a site safety manager
should be asked to set rather than an engineer.

### Why "violation" is a positive detection

A violation is a positive detection of a `NO-*` class, never the absence of a `Hardhat` box. Had
it been the latter, every missed detection of any kind — a worker the model simply did not see —
would silently have become a "violation", and false negatives would have been indistinguishable
from false positives. The dataset's paired classes are what make this analysis possible at all.

---

## Results

> ⏳ **Pending the training run.** Filled from the executed notebook's captured output. Left
> empty rather than populated with plausible-looking numbers.

### Headline (test split)

| metric | value |
|---|---|
| mAP50 | _pending_ |
| mAP50-95 | _pending_ |
| precision | _pending_ |
| recall | _pending_ |

### Per class

_pending_ — see `outputs/03_per_class_metrics.csv` and `outputs/03_recall_by_class.png`.

### Selected operating point

| | value |
|---|---|
| confidence | _pending_ |
| NMS IoU | _pending_ |
| violation-class recall | _pending_ |
| violation-class precision | _pending_ |
| vs. the `conf=0.25` default | _pending_ |

Machine-readable copy: `outputs/03_operating_point.json`. `ppe_monitor/config.py` is set to
match, so the Streamlit app, the violation ledger and notebook 02 all run at the threshold
justified here rather than at a library default.

### Failure analysis

_pending_ — the notebook renders the actual images behind every false negative and false positive
on the violation classes.

---

## Two IoU thresholds, easily confused

- **NMS IoU** (`iou=` in `predict` / `val`) — how much two *predicted* boxes may overlap before
  one is suppressed. Too low and a worker standing in front of another loses their box; too high
  and one worker collects duplicates. This is a knob, and the notebook sweeps it (0.40–0.70)
  because site footage has heavy worker-on-worker occlusion.
- **Matching IoU** — how much a prediction must overlap ground truth to count as correct. This is
  what the "50" in mAP50 refers to. It is a property of the metric, not a setting.

---

## Caveats, stated up front

- **The test split is small.** 82 images, and only **41 `NO-Hardhat` instances**. Each individual
  miss moves recall on that class by roughly 2.4 points. These figures indicate direction, not
  precision, and a difference of a few points between configurations is noise.
- **Class imbalance shapes the averages.** `Person` is nearly a quarter of all instances, so
  overall mAP is largely a statement about how well the model finds people. That is exactly why
  the per-class table exists and why the violation classes are pulled out separately.
- **Rare classes will look bad.** `Mask` has 28 test instances. Its metrics are reported for
  completeness, not because they support conclusions.
- **One dataset, one domain.** Metrics here predict performance on *this* dataset's distribution:
  daytime, outdoor, mostly clear weather. A night shift, heavy rain, or an indoor fit-out is
  out-of-distribution, and per-site calibration would be needed before trusting these numbers on
  a real deployment.

## Known failure modes

The notebook shows the images; the recurring causes on this dataset are:

- **Scale.** Distant workers occupy a few dozen pixels, and at 640 px input the detail separating
  a bare head from a helmeted one is simply gone. This is a
  [SAHI](https://docs.ultralytics.com/guides/sahi-tiled-inference/) tiled-inference problem, not
  a training problem — more epochs will not fix it.
- **Occlusion and unusual pose.** A worker bent over or half behind machinery presents a shape
  the model has fewer examples of.
- **`Person` / `NO-Hardhat` overlap.** Both boxes cover much of the same worker, so a confident
  `Person` detection can crowd the violation box.
- **Genuine label ambiguity.** Soft caps, hoods and bare heads at the boundary are hard for
  annotators too, and the ground truth is not perfectly consistent about them. Some apparent
  false positives are arguably correct.
