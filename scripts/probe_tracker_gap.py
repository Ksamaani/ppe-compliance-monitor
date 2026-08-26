"""Why the ledger logs fewer violations than the detector finds.

Every metric in `docs/EVALUATION.md` comes from single-image detection. The
deployed system is a *video* pipeline, and between the detector and the
violation ledger sits the tracker. This script measures the gap between the two
rather than assuming it, because the difference turns out to be the binding
constraint on two of the three violation classes.

Two passes over the same clip at the shipped operating point:

  Pass 1 - `predict` on every Nth frame. How much does the detector see at all?
  Pass 2 - `track` on every frame. How much of that survives association into a
           confirmed track ID, and does any track hold a violation long enough
           for the debounce in `ppe_monitor.compliance` to open an incident?

`model.track()` emits only confirmed tracks. A detection that never associates
with the same object across consecutive frames is dropped before the ledger can
see it, so it cannot open an incident no matter how confident the detector was.

Usage:
    python scripts/probe_tracker_gap.py
    python scripts/probe_tracker_gap.py --stride 10 --max-frames 510
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppe_monitor.config import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IOU,
    ENFORCED_CLASSES,
    OPEN_AFTER_FRAMES,
    SITE_CLIP,
    VIOLATION_CLASSES,
)

VIOLATION_ORDER = ("NO-Hardhat", "NO-Mask", "NO-Safety Vest")


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def detection_pass(model, clip: Path, stride: int) -> Counter:
    """Pass 1: what the detector finds, sampling every `stride`th frame."""
    cap = cv2.VideoCapture(str(clip))
    counts: Counter = Counter()
    peak: dict[str, float] = {}
    frames_with_violation = 0
    sampled = 0
    idx = -1

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        sampled += 1
        result = model.predict(frame, conf=DEFAULT_CONF, iou=DEFAULT_IOU, verbose=False)[0]
        labels = [result.names[int(b.cls)] for b in result.boxes]
        counts.update(labels)
        for box in result.boxes:
            label = result.names[int(box.cls)]
            peak[label] = max(peak.get(label, 0.0), float(box.conf))
        if any(label in VIOLATION_CLASSES for label in labels):
            frames_with_violation += 1

    cap.release()

    print(f"sampled {sampled} frames (every {stride}th) at conf={DEFAULT_CONF}, iou={DEFAULT_IOU}\n")
    print(f"{'class':<18}{'detections':>11}{'peak conf':>11}")
    print("-" * 40)
    for label, count in counts.most_common():
        if label in ENFORCED_CLASSES:
            mark = "  <- ENFORCED violation"
        elif label in VIOLATION_CLASSES:
            mark = "  <- violation, not enforced"
        else:
            mark = ""
        print(f"{label:<18}{count:>11}{peak[label]:>11.3f}{mark}")

    print()
    print(f"sampled frames containing any violation class : {frames_with_violation}/{sampled}")
    return counts


def tracking_pass(model, clip: Path, max_frames: int) -> None:
    """Pass 2: how much of that survives into confirmed tracks, and for how long."""
    cap = cv2.VideoCapture(str(clip))

    current: dict[tuple[int, str], int] = defaultdict(int)
    best: dict[tuple[int, str], int] = defaultdict(int)
    untracked: Counter = Counter()

    idx = -1
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx >= max_frames:
            break

        result = model.track(frame, persist=True, conf=DEFAULT_CONF, iou=DEFAULT_IOU, verbose=False)[0]
        boxes = result.boxes
        seen: set[tuple[int, str]] = set()

        if boxes is not None and len(boxes):
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            for track_id, cls in zip(ids, boxes.cls.int().tolist()):
                label = result.names[int(cls)]
                if label not in VIOLATION_CLASSES:
                    continue
                if track_id is None:
                    # ViolationLedger.update() skips frames where boxes.id is
                    # None, so an unassociated detection never reaches the ledger.
                    untracked[label] += 1
                    continue
                seen.add((int(track_id), label))

        for key in seen:
            current[key] += 1
            best[key] = max(best[key], current[key])
        for key in list(current):
            if key not in seen:
                current[key] = 0

    cap.release()

    print(f"tracked {idx + 1} frames at conf={DEFAULT_CONF}; the ledger opens an incident")
    print(f"after {OPEN_AFTER_FRAMES} consecutive violating frames on one track ID\n")

    for label in VIOLATION_ORDER:
        streaks = sorted((v for (_, lab), v in best.items() if lab == label), reverse=True)
        if not streaks:
            print(f"{label:<16} never held a confirmed track ID")
            if untracked[label]:
                print(f"{'':<16} {untracked[label]} detection(s) the tracker assigned no ID")
            print()
            continue
        reached = sum(1 for s in streaks if s >= OPEN_AFTER_FRAMES)
        print(f"{label:<16} {len(streaks):>3} distinct track IDs   "
              f"longest streak {streaks[0]:>3} frames   "
              f"{reached} reached the {OPEN_AFTER_FRAMES}-frame threshold")
        print(f"{'':<16} top streaks: {streaks[:8]}")
        if untracked[label]:
            print(f"{'':<16} {untracked[label]} detection(s) the tracker assigned no ID")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=10, help="detection-pass frame stride")
    parser.add_argument("--max-frames", type=int, default=510, help="frames for the tracking pass")
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"weights not found: {weights}")
        print("Run notebooks/04_training_colab.ipynb and place best.pt in models/ first.")
        return 1
    if not SITE_CLIP.exists():
        print(f"clip not found: {SITE_CLIP}")
        print("Run scripts/fetch_assets.py first.")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(SITE_CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    rule("1. What the DETECTOR finds")
    print(f"clip: {SITE_CLIP.name}, {total} frames\n")
    counts = detection_pass(model, SITE_CLIP, args.stride)

    rule("2. What survives into CONFIRMED TRACKS")
    tracking_pass(model, SITE_CLIP, args.max_frames)

    rule("READING")
    print("Detections that never associate into a stable track ID cannot open an")
    print("incident, however confident the detector was. Where a class is detected")
    print("often but holds no track, the ledger's recall is bounded by the TRACKER,")
    print("not by the detector - and a model improvement alone would not produce")
    print("more logged incidents on footage like this.")
    print()
    print("The lever that would: track the large, stable Person box and attach the")
    print("violation to it, rather than tracking the small violation box directly.")

    detected_violations = sum(c for lab, c in counts.items() if lab in VIOLATION_CLASSES)
    return 0 if detected_violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
