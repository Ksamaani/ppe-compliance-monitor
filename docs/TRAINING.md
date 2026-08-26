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

> ⏳ **Pending the Colab run.** This section is filled from the executed notebook's captured
> output — `run_summary.csv`, `per_class_comparison.csv` and `metrics.json` in the downloaded
> artefacts. It is deliberately left empty rather than filled with plausible-looking numbers.

| run | epochs | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| A baseline | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| B tuned | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

**Per-class recall on the critical classes:** _pending_

**Over/underfitting read:** _pending_

**Selected model:** _pending_

**Shipped confidence threshold:** chosen in [`EVALUATION.md`](EVALUATION.md) by sweeping on the
held-out test split, not set to the library default.

---

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
