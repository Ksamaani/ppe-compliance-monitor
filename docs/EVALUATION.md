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

> **Maximise mean recall across the enforced violation classes, subject to mean precision across
> those classes staying at or above 0.50** — at least half of the alarms raised must be real.

The precision floor is a judgement call, and it is the one number here that a site safety manager
should be asked to set rather than an engineer.

*Which* classes are enforced turned out to be a decision in its own right — see
[The average was hiding a failing class](#the-average-was-hiding-a-failing-class) below.

### Why "violation" is a positive detection

A violation is a positive detection of a `NO-*` class, never the absence of a `Hardhat` box. Had
it been the latter, every missed detection of any kind — a worker the model simply did not see —
would silently have become a "violation", and false negatives would have been indistinguishable
from false positives. The dataset's paired classes are what make this analysis possible at all.

---

## Results

Model: Run A from the training stage (`models/best.pt`). Split: **test**, 82 images, 760
instances, 210 of them violations.

### Headline (test split)

| metric | value |
|---|---|
| mAP50 | **0.6798** |
| mAP50-95 | **0.3771** |
| precision | 0.8546 |
| recall | 0.6061 |

### Per class

Full table in `outputs/03_per_class_metrics.csv`; chart in `outputs/03_recall_by_class.png`.
The short version: common classes (`Person`, `machinery`, `Safety Cone`) do well, and the three
violation classes — the ones this system exists for — sit lowest.

### How the operating point was chosen, and one methodological correction

The first version of this analysis swept `model.val(conf=X)` and read `box.mp` / `box.mr` back.
**That is the wrong instrument**, and the output said so plainly: precision and recall came back
*identical* for conf 0.05 / 0.10 / 0.15, and the "winner" was chosen on a 0.0005 difference that
was pure noise.

The reason is that Ultralytics reports precision and recall at the **maximum-F1 point of the PR
curve**, not at the confidence threshold passed in. `conf=X` merely discards detections below X
before the curve is built. So sweeping it answers *"what is the best achievable F1 if I throw away
everything under X?"* — not *"what will I get if I deploy at X?"*. Only the second is a
deployment question.

The published sweep therefore measures deployed behaviour directly: predict once at a score floor
of 0.01, then apply each candidate threshold in memory and **match predictions to ground truth by
IoU ≥ 0.50**, counting TP / FP / FN by hand. Confidence and NMS IoU are swept jointly, since NMS
decides which boxes survive and confidence then decides which survivors raise an alarm.

With that fixed, the curve behaves as it must — monotonic in both directions (`nms_iou = 0.50`):

| conf | precision | recall | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.5045 | 0.6713 | 144 | 156 | 66 |
| 0.10 | 0.6150 | 0.6544 | 140 | 90 | 70 |
| 0.15 | 0.6742 | 0.6428 | 137 | 66 | 73 |
| 0.25 | 0.7502 | 0.5990 | 126 | 39 | 84 |
| 0.40 | 0.8597 | 0.5389 | 113 | 15 | 97 |
| 0.60 | 0.9270 | 0.4190 | 88 | 5 | 122 |
| 0.70 | 0.9074 | 0.3263 | 69 | 5 | 141 |

### The average was hiding a failing class

At `conf=0.05` the mean precision across the three violation classes is 0.5045 — it clears the
0.50 floor. Per class, it does not:

| class | precision | true alarms | false alarms | verdict |
|---|---:|---:|---:|---|
| `NO-Hardhat` | 0.5208 | 25 | 23 | above the floor |
| **`NO-Mask`** | **0.3581** | 53 | **95** | **far below** |
| `NO-Safety Vest` | 0.6346 | 66 | 38 | above the floor |

`NO-Mask` was being carried by the other two. And there is a domain reason behind the number:
**the dataset treats "no mask" as a PPE violation, but an open-air formwork deck does not require
masks.** Respiratory protection is a demolition, grinding or silica-dust rule, not a general site
rule. Enforcing it everywhere generates alarms that are correct against the labels and wrong
against site policy — the fastest way to get a safety system switched off.

So whether `NO-Mask` is enforced is a **site policy decision, not a model property**. It lives in
`ppe_monitor.config.ENFORCED_CLASSES`, and a site that does require masks re-enables it there
with no other change.

### Selected operating point

Enforcing `NO-Hardhat` and `NO-Safety Vest`:

| conf | precision | recall |
|---:|---:|---:|
| 0.05 | 0.5777 | 0.6716 |
| **0.10** | **0.6799** | **0.6716** |
| 0.15 | 0.7290 | 0.6604 |
| 0.25 (default) | 0.7810 | 0.6327 |

| | value |
|---|---|
| **confidence** | **0.10** |
| **NMS IoU** | **0.50** |
| **enforced classes** | `NO-Hardhat`, `NO-Safety Vest` |
| violation recall | **0.6715** |
| violation precision | **0.6799** |

Two things are happening there, and they are worth separating:

1. **A free improvement.** Recall is *identical* at conf 0.05 and 0.10, but precision is 10 points
   better at 0.10. Same violations caught, fewer false alarms — take it. That plateau is invisible
   to a `val(conf=...)` sweep, which is the practical payoff of fixing the methodology.
2. **A priced trade.** Against the default `conf=0.25`, the shipped point buys **+3.9 points of
   recall for −10.1 points of precision**. That is not a free win and is not presented as one.
   Under the stated rule it is still the right call: precision clears the 0.50 floor with room to
   spare, and the recall gained is violations that would otherwise be missed entirely. A site that
   cannot absorb the extra false alarms raises the floor and lands at conf 0.15 or 0.20 — the
   table prices that choice instead of leaving it inherited from a library default.

Machine-readable copy: `outputs/03_operating_point.json`. `ppe_monitor/config.py` is set to match,
so the Streamlit app, the violation ledger and notebook 02 all run at the threshold justified
here.

### Failure analysis

Two different points get counted here, and conflating them would overstate the cost of what is
actually shipped.

**The candidate point** — conf 0.05, all three labelled violation classes — produces **66 false
negatives and 156 false positives** against 210 ground-truth instances:

| class | TP | FP | FN |
|---|---:|---:|---:|
| `NO-Hardhat` | 25 | 23 | 16 |
| `NO-Mask` | 53 | 95 | 26 |
| `NO-Safety Vest` | 66 | 38 | 24 |

**The shipped point** — conf 0.10, enforcing `NO-Hardhat` and `NO-Safety Vest` — is a different
threshold over a different class set, so its errors are recounted rather than inherited:

| class | TP | FP | FN |
|---|---:|---:|---:|
| `NO-Hardhat` | 25 | 16 | 16 |
| `NO-Safety Vest` | 66 | 22 | 24 |
| **total** | **91** | **38** | **40** |

131 ground-truth instances, **40 missed violations and 38 false alarms**. Dropping `NO-Mask` and
raising the threshold removes 118 of the 156 false alarms — and costs no true positives at all,
because the same 91 detections survive both moves. The alarms removed were correct against the
dataset's labels and wrong against site policy, which is the distinction that decides whether a
safety system stays switched on.

The error gallery in the notebook is rendered at the candidate point deliberately: the lower
threshold surfaces more of the failure modes, and the causes below are what those images show.

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
