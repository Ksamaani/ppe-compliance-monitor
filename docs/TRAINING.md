# Training

Fine-tuning `yolo11n` on the Construction Site Safety dataset so the system can see PPE at all.
Dataset provenance, class list and measured distribution: [`DATASET.md`](DATASET.md).

## Why training is necessary

Notebook 01 establishes the gap empirically rather than asserting it. COCO-pretrained weights
find *people* reliably when they are large in frame, and have no concept of a hard hat — there is
no `Hardhat` class, no `NO-Hardhat` class, nothing PPE-related among the 80 COCO categories.
Compliance cannot be assessed at all with off-the-shelf weights. That is what this stage fixes.

## Where it runs

**Google Colab, T4 GPU** — [`notebooks/04_training_colab.ipynb`](../notebooks/04_training_colab.ipynb).

The development machine has no CUDA device (Intel i7-10510U, integrated graphics). A 25-epoch run
over 2,605 images at 640 px would take many hours on this CPU; on a free T4 it is roughly 20
minutes.

### Guarding the GPU session

A Colab run that fails in cell 6 on a path bug costs 40 minutes and a restart. So
[`scripts/smoke_test_training.py`](../scripts/smoke_test_training.py) exercises **every API call
the notebook makes** locally on CPU first, against a 24-image subset at 320 px for one epoch:

| Step | Verified |
|---|---|
| `kagglehub.dataset_download` | ✅ anonymous, no API key |
| split discovery + `data.yaml` authoring | ✅ absolute paths, correct class order |
| `model.train` | ✅ produces `weights/best.pt`, `results.csv`, plots |
| `model.val` | ✅ `box.map50`, `box.map`, `box.mp`, `box.mr` |
| per-class metrics | ✅ `box.class_result(i)` ↔ `box.ap_class_index` |
| `results.csv` parsing | ✅ all 15 columns the curve plots need |
| confidence sweep | ✅ `val(conf=…)` at three thresholds |
| `model.export(format="onnx")` + reload | ✅ round-trip predict |

Total runtime 52 seconds. It produces a deliberately useless model — 1 epoch on 24 images,
mAP50 ≈ 0.014. It is testing the plumbing, not the weights. Full output:
[`logs/05_smoke_test.log`](../logs/05_smoke_test.log).

## Two runs, not one

A single training run tells you where the model landed but nothing about why. Both runs use
`seed=42` and `imgsz=640` so the comparison isolates the knobs below.

| knob | Run A — baseline | Run B — tuned | reasoning |
|---|---|---|---|
| `epochs` | 25 | 30 | give B headroom |
| `patience` | — | 10 | stop early if val mAP stalls |
| `batch` | −1 (AutoBatch) | −1 | fill the T4's memory |
| `freeze` | 0 | **10** | COCO's early layers already encode edges, corners and texture. Freezing them keeps those generic features and spends the gradient budget on the PPE-specific head — the standard move when the target dataset is small relative to the pretraining set. |
| `weight_decay` | 0.0005 | **0.001** | 2,605 images is small enough for a model this size to memorise; stronger L2 pushes back |
| `hsv_v` | 0.4 | **0.5** | site lighting swings between direct glare and deep shadow within one frame |
| `degrees` | 0.0 | **10.0** | fixed site cameras are never perfectly level, and neither are phone snaps |
| `translate` | 0.1 | **0.2** | workers appear anywhere in frame, not politely centred |
| `scale` | 0.5 | **0.7** | huge range of worker distances between foreground and the far end of a site |
| `close_mosaic` | 10 | 10 | mosaic augmentation off for the final 10 epochs so the model finishes on realistic whole images |

### What to look for in the curves

The notebook plots `train/box_loss` against `val/box_loss` for both runs, plus mAP50 and
mAP50-95:

- **Train loss falling while val loss turns upward** → overfitting. The extra regularisation and
  augmentation in Run B exist to delay this.
- **Both still falling at the last epoch** → underfitting; more epochs would have helped.
- **`NO-Hardhat` recall specifically**, not just overall mAP. Overall mAP is dominated by the
  common `Person` class; this system is judged on catching violations.

Model selection is on **mAP50-95** rather than mAP50, because localisation quality matters when
the box is cropped as photographic evidence of a violation — a loose box that clips the worker's
head is a weaker record.

---

## Results

Both runs completed on a Colab T4: Run A in 16.7 minutes, Run B in 19.3 minutes.

| run | epochs | mAP50 | mAP50-95 | precision | recall | final val box loss |
|---|---:|---:|---:|---:|---:|---:|
| **A baseline** | 25 | **0.7545** | **0.4323** | **0.8863** | **0.6682** | **1.505** |
| B tuned | 30 | 0.6964 | 0.2962 | 0.8153 | 0.6438 | 2.007 |

**The tuned run lost, on every single metric.** That is the interesting result, and it is worth
more than a confirmation would have been.

### Per-class recall (validation split)

| class | A baseline | B tuned |
|---|---:|---:|
| `Hardhat` | 0.7345 | 0.7301 |
| `Mask` | 0.8571 | 0.8571 |
| **`NO-Hardhat`** | **0.5131** | 0.5269 |
| **`NO-Mask`** | **0.5135** | 0.4324 |
| **`NO-Safety Vest`** | **0.5329** | 0.5377 |
| `Person` | 0.7108 | 0.6720 |
| `Safety Cone` | 0.8409 | 0.7531 |
| `Safety Vest` | 0.7084 | 0.7073 |
| `machinery` | 0.8178 | 0.7455 |
| `vehicle` | 0.4524 | 0.4762 |

The three violation classes sit around 0.51–0.53 recall while their precision is 0.88–0.92. The
model is **cautious**: when it calls a violation it is almost always right, but it stays quiet
about half the time. For a safety system that is exactly the wrong way round, and it is the
reason `03_evaluation.ipynb` sweeps the confidence threshold instead of shipping the default —
there is a large amount of recall available to buy with precision the model can spare.

### Over- or underfitting?

**Underfitting, both runs.** Not what the tuning was designed for.

- `val/box_loss` **falls monotonically** for both runs (A: 1.87 → 1.51, B: 2.16 → 2.01). It never
  turns upward, which is the signature of overfitting. There is none.
- mAP is **still climbing at the final epoch** for both — A's best mAP50 is at epoch 25 of 25,
  B's best mAP50-95 at epoch 30 of 30. Both wanted more epochs.
- B's `patience=10` never triggered, confirming it was still improving — just from a worse
  trajectory.

### Why the tuning backfired

Run B was built to fight overfitting that was not happening. Applying regularisation to an
underfitting model just slows it down, and the details show exactly how:

- **`freeze=10` was the main cost.** Freezing the backbone assumes COCO's early features already
  suit the target domain. They partly do — but hard hats at 30 metres, hi-vis fabric under glare
  and the fine distinction between a helmeted and a bare head are not things COCO's frozen layers
  encode well. The backbone needed to adapt, and it was not allowed to.
- **The mAP50-95 gap is far wider than the mAP50 gap** — 0.43 → 0.30 (−31%) against 0.75 → 0.70
  (−8%). B finds objects nearly as often but localises them *loosely*. That is the fingerprint of
  aggressive geometric augmentation (`degrees=10`, `scale=0.7`, `translate=0.2`) without enough
  epochs left to re-converge on tight boxes.
- **Five extra epochs did not compensate**, because the deficit was capacity and convergence
  speed, not training length.

### What would be tried next

In priority order, given more GPU budget:

1. **More epochs on the Run A recipe** — 50–75. Both curves say the cheapest gain is still there.
2. **Unfreeze entirely, keep the mild augmentation.** Test whether `freeze` alone caused the
   damage or the augmentation shared the blame.
3. **A larger backbone** (`yolo11s`/`yolo11m`). Nano was chosen for CPU inference, but the
   underfitting suggests capacity is a real constraint.
4. **Class-weighted loss or oversampling for the `NO-*` classes**, since violation recall is what
   the system is judged on and those classes are outnumbered by `Person` roughly four to one.

**Selected model:** **Run A (baseline)**, on mAP50-95 — `models/best.pt`.

**Shipped confidence threshold:** chosen in [`EVALUATION.md`](EVALUATION.md) by sweeping the
held-out test split, not set to the library default.

Raw artefacts for both runs — `results.csv`, `args.yaml`, confusion matrices, curves,
`metrics.json` — are committed under [`runs_artifacts/`](../runs_artifacts). Trained weights are
excluded as build output.

## Reproducing

1. Open [`notebooks/04_training_colab.ipynb`](../notebooks/04_training_colab.ipynb) in Colab
   (badge at the top of the notebook).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.** Roughly 40–60 minutes for both runs.
4. The last cell zips the artefacts and starts a download.
5. Unpack: `best.pt` → `models/best.pt`, the rest → `runs_artifacts/`.

Then locally:

```bash
jupyter execute notebooks/03_evaluation.ipynb    # metrics + threshold selection
jupyter execute notebooks/02_video_analytics.ipynb  # re-run the ledger on PPE classes
python scripts/export_onnx.py                    # deployment model
```
