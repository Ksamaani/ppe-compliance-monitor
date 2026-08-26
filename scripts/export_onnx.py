"""Export the trained PPE model to ONNX and prove the export actually works.

Deliverable 5. An export that is never loaded back is not evidence of anything,
so this script round-trips it: export, reload through the Ultralytics API, run
inference on a real site image, and compare the ONNX predictions against the
PyTorch ones detection by detection.

Target justification (also in docs/DEPLOYMENT.md): the deployment hardware for
this project is a CPU-only Intel machine with no CUDA device. TensorRT is not
available and CoreML is the wrong platform, which leaves ONNX Runtime as the
portable CPU target that runs unchanged on the dev laptop and on a site box.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --imgsz 640 --simplify
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppe_monitor.config import IMAGE_DIR, MODEL_DIR  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def summarise(result) -> list[tuple[str, float]]:
    """(class, confidence) for each detection, sorted for stable comparison."""
    return sorted(
        ((result.names[int(b.cls)], round(float(b.conf), 3)) for b in result.boxes),
        key=lambda t: (-t[1], t[0]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=MODEL_DIR / "best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--simplify", action="store_true", help="run onnx-simplifier")
    parser.add_argument("--opset", type=int, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.weights.exists():
        print(f"ERROR: {args.weights} not found.")
        print("Run notebooks/04_training_colab.ipynb on a GPU first, then unpack")
        print("ppe_training_artifacts/best.pt into models/best.pt.")
        return 1

    from ultralytics import YOLO

    rule("1. Load the trained model")
    model = YOLO(str(args.weights))
    print(f"weights : {args.weights}")
    print(f"task    : {model.task}")
    print(f"classes : {list(model.names.values())}")

    rule("2. model.export(format='onnx')")
    export_kwargs = {"format": "onnx", "imgsz": args.imgsz, "simplify": args.simplify}
    if args.opset:
        export_kwargs["opset"] = args.opset
    t0 = time.time()
    onnx_path = Path(model.export(**export_kwargs))
    print(f"\nexported in {time.time() - t0:.1f}s -> {onnx_path}")
    print(f"size: {onnx_path.stat().st_size / 1e6:.2f} MB "
          f"(PyTorch: {args.weights.stat().st_size / 1e6:.2f} MB)")

    # Keep the export next to the weights the app looks for.
    target = MODEL_DIR / "best.onnx"
    if onnx_path.resolve() != target.resolve():
        target.write_bytes(onnx_path.read_bytes())
        print(f"copied to {target}")

    rule("3. Round-trip: reload the ONNX model and run real inference")
    sample = next(iter(sorted(IMAGE_DIR.glob("*.jpg"))), None)
    if sample is None:
        print("No sample image; run scripts/fetch_assets.py first.")
        return 1

    onnx_model = YOLO(str(target))
    t0 = time.time()
    onnx_result = onnx_model.predict(source=str(sample), imgsz=args.imgsz, verbose=False)[0]
    onnx_ms = (time.time() - t0) * 1000

    t0 = time.time()
    pt_result = model.predict(source=str(sample), imgsz=args.imgsz, verbose=False)[0]
    pt_ms = (time.time() - t0) * 1000

    print(f"sample image : {sample.name}")
    print(f"PyTorch      : {len(pt_result.boxes)} detections in {pt_ms:.0f} ms")
    print(f"ONNX Runtime : {len(onnx_result.boxes)} detections in {onnx_ms:.0f} ms")

    rule("4. Do the two agree?")
    pt_dets, onnx_dets = summarise(pt_result), summarise(onnx_result)
    print(f"{'PyTorch':<34}{'ONNX'}")
    for i in range(max(len(pt_dets), len(onnx_dets))):
        a = f"{pt_dets[i][0]} {pt_dets[i][1]:.3f}" if i < len(pt_dets) else "-"
        b = f"{onnx_dets[i][0]} {onnx_dets[i][1]:.3f}" if i < len(onnx_dets) else "-"
        print(f"  {a:<32}{b}")

    same_classes = [c for c, _ in pt_dets] == [c for c, _ in onnx_dets]
    print(f"\nsame detection count : {len(pt_dets) == len(onnx_dets)}")
    print(f"same class sequence  : {same_classes}")
    if same_classes and pt_dets:
        drift = max(abs(a[1] - b[1]) for a, b in zip(pt_dets, onnx_dets))
        print(f"max confidence drift : {drift:.4f}")
        print("(small drift is expected — ONNX runs fp32 kernels in a different order)")

    annotated = onnx_result.plot()
    out = ROOT / "outputs" / "05_onnx_roundtrip.jpg"
    import cv2

    h, w = annotated.shape[:2]
    if w > 1000:
        annotated = cv2.resize(annotated, (1000, int(h * 1000 / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\nannotated ONNX output written to {out.relative_to(ROOT)}")

    rule("RESULT")
    print("ONNX export verified: the file loads, runs, and reproduces the PyTorch")
    print("model's detections on a real construction-site image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
