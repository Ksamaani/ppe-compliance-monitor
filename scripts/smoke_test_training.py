"""Prove the training pipeline works on CPU before spending a GPU session on it.

The real fine-tuning runs on a Colab T4 (notebook 04). A Colab run that dies in
cell 6 because of a path bug or a renamed argument costs 40 minutes and has to be
started over. This script exercises *every API call that notebook makes* against
a 24-image subset at 320px for one epoch, locally, in a couple of minutes:

    kagglehub download -> split discovery -> data.yaml authoring
    -> model.train -> model.val -> per-class metrics -> results.csv parsing

It produces a deliberately useless model. That is fine: it is testing the
plumbing, not the model.

Usage:
    python scripts/smoke_test_training.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppe_monitor import dataset  # noqa: E402
from ppe_monitor.dataset import CLASS_NAMES  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    started = time.time()

    rule("1. Dataset download (kagglehub, anonymous)")
    root = dataset.download()
    split_root = dataset.find_split_root(root)
    splits = dataset.resolve_splits(split_root)
    print(f"split root : {split_root}")
    for name, path in splits.items():
        print(f"  {name:<6} -> {path}")

    rule("2. Split contents and class distribution")
    report = dataset.describe(splits)
    for name, stats in report.items():
        print(f"{name}: {stats['images']} images, {stats['instances']} instances")
        for cls, n in sorted(stats["per_class"].items(), key=lambda kv: -kv[1]):
            print(f"    {cls:<16}{n:>7}")

    rule("3. data.yaml authoring (full dataset)")
    full_yaml = dataset.write_data_yaml(split_root, ROOT / "data" / "data.yaml", splits)
    print(f"wrote {full_yaml}")
    print(full_yaml.read_text(encoding="utf-8"))

    rule("4. Tiny subset for the CPU smoke run")
    subset_yaml = dataset.make_subset(
        splits, ROOT / "data" / "datasets" / "_smoke", n_train=24, n_valid=12
    )
    print(f"wrote {subset_yaml}")
    print(subset_yaml.read_text(encoding="utf-8"))

    from ultralytics import YOLO, settings

    settings.update({
        "runs_dir": str(ROOT / "runs"),
        "datasets_dir": str(ROOT / "data" / "datasets"),
    })

    rule("5. model.train - 1 epoch, 320px, CPU")
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(subset_yaml),
        epochs=1,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,          # Windows: dataloader workers deadlock without this
        seed=42,
        project=str(ROOT / "runs" / "smoke"),
        name="train",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    save_dir = Path(results.save_dir)
    print(f"\nsave_dir: {save_dir}")
    print("artefacts produced:")
    for p in sorted(save_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(save_dir)}  ({p.stat().st_size:,} bytes)")

    rule("6. model.val - metrics access")
    best = save_dir / "weights" / "best.pt"
    trained = YOLO(str(best))
    metrics = trained.val(
        data=str(subset_yaml),
        imgsz=320,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs" / "smoke"),
        name="val",
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    print(f"mAP50      : {metrics.box.map50:.5f}")
    print(f"mAP50-95   : {metrics.box.map:.5f}")
    print(f"precision  : {metrics.box.mp:.5f}")
    print(f"recall     : {metrics.box.mr:.5f}")

    print("\nper-class access (the API the evaluation notebook depends on):")
    print(f"  metrics.box.maps shape : {getattr(metrics.box.maps, 'shape', len(metrics.box.maps))}")
    for idx, cls_idx in enumerate(metrics.box.ap_class_index):
        name = CLASS_NAMES[int(cls_idx)]
        p, r, ap50, ap = metrics.box.class_result(idx)
        print(f"  {name:<16} P={p:.3f} R={r:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")

    print("\nresults_dict keys:", sorted(metrics.results_dict.keys()))

    rule("7. results.csv parsing")
    csv_path = save_dir / "results.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        print(f"{len(df)} epoch row(s); columns:")
        for c in df.columns:
            print(f"  {c}")
        print("\nlast row:")
        print(df.iloc[-1].to_string())
    else:
        print("!! results.csv missing - the training notebook's curve plots would fail")
        return 1

    rule("8. Confidence sweep API")
    sweep = {}
    for conf in (0.10, 0.25, 0.50):
        m = trained.val(
            data=str(subset_yaml), imgsz=320, conf=conf, device="cpu", workers=0,
            project=str(ROOT / "runs" / "smoke"), name=f"sweep_{int(conf * 100)}",
            exist_ok=True, plots=False, verbose=False,
        )
        sweep[conf] = {"P": round(float(m.box.mp), 4), "R": round(float(m.box.mr), 4)}
        print(f"  conf={conf:.2f} -> P={m.box.mp:.4f} R={m.box.mr:.4f}")

    rule("9. model.export - ONNX")
    onnx_path = trained.export(format="onnx", imgsz=320)
    print(f"exported: {onnx_path}")
    reloaded = YOLO(str(onnx_path))
    probe = reloaded.predict(source=str(ROOT / "data" / "images" / "site_07.jpg"),
                             imgsz=320, verbose=False)[0]
    print(f"ONNX round-trip predict -> {len(probe.boxes)} boxes, shape {tuple(probe.boxes.data.shape)}")

    rule("RESULT")
    print(json.dumps({
        "elapsed_s": round(time.time() - started, 1),
        "train_epochs": 1,
        "map50": round(float(metrics.box.map50), 5),
        "sweep": sweep,
        "onnx": Path(onnx_path).name,
    }, indent=2))
    print("\nEvery API call the Colab notebook makes has now run successfully on CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
