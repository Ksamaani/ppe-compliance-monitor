"""Tests for the violation ledger's debounce logic.

The ledger is the piece of this project that turns noisy per-frame detections
into something a safety officer can act on, so its state machine is worth
pinning down: a one-frame blip must not become an incident, and a one-frame
dropout must not split one incident into two.

Run:  python -m pytest tests/ -q      (or)      python tests/test_compliance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe_monitor.compliance import ViolationLedger  # noqa: E402

NAMES = {0: "Hardhat", 2: "NO-Hardhat", 5: "Person"}


class FakeBoxes:
    def __init__(self, ids, classes, confs):
        self.id = torch.tensor(ids, dtype=torch.float32) if ids else None
        self.cls = torch.tensor(classes, dtype=torch.float32)
        self.conf = torch.tensor(confs, dtype=torch.float32)
        self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 90.0]] * len(classes))


class FakeResult:
    """Minimal stand-in for an Ultralytics Results object."""

    def __init__(self, ids=(), classes=(), confs=()):
        self.names = NAMES
        self.boxes = FakeBoxes(list(ids), list(classes), list(confs))


def violating(track_id=1, conf=0.8):
    return FakeResult(ids=[track_id], classes=[2], confs=[conf])  # NO-Hardhat


def compliant(track_id=1, conf=0.8):
    return FakeResult(ids=[track_id], classes=[0], confs=[conf])  # Hardhat


def empty():
    return FakeResult()


def feed(ledger, frames, start=0):
    idx = start
    for result in frames:
        ledger.update(idx, result)
        idx += 1
    return idx - 1


def test_brief_blip_is_not_an_incident():
    """Two violating frames with open_after=5 must produce nothing."""
    ledger = ViolationLedger(open_after=5, close_after=3)
    last = feed(ledger, [violating(), violating()] + [empty()] * 10)
    ledger.close_all(last)
    assert ledger.incidents == [], f"expected no incidents, got {ledger.incidents}"


def test_sustained_violation_opens_one_incident():
    ledger = ViolationLedger(open_after=5, close_after=3)
    last = feed(ledger, [violating()] * 12 + [empty()] * 6)
    ledger.close_all(last)
    assert len(ledger.incidents) == 1, f"expected 1 incident, got {len(ledger.incidents)}"
    inc = ledger.incidents[0]
    assert inc.label == "NO-Hardhat"
    assert inc.track_id == 1
    # Backdated to the true start of the streak, not the frame it was confirmed.
    assert inc.first_frame == 0, f"expected first_frame 0, got {inc.first_frame}"
    assert inc.last_frame == 11, f"expected last_frame 11, got {inc.last_frame}"
    assert inc.duration_frames() == 12


def test_single_frame_dropout_does_not_split_incident():
    """A one-frame detection gap with close_after=3 stays one incident."""
    ledger = ViolationLedger(open_after=3, close_after=3)
    frames = [violating()] * 6 + [empty()] + [violating()] * 6 + [empty()] * 5
    last = feed(ledger, frames)
    ledger.close_all(last)
    assert len(ledger.incidents) == 1, f"expected 1 incident, got {len(ledger.incidents)}"


def test_long_gap_splits_into_two_incidents():
    ledger = ViolationLedger(open_after=3, close_after=3)
    frames = [violating()] * 6 + [empty()] * 5 + [violating()] * 6 + [empty()] * 5
    last = feed(ledger, frames)
    ledger.close_all(last)
    assert len(ledger.incidents) == 2, f"expected 2 incidents, got {len(ledger.incidents)}"


def test_compliant_detections_are_ignored():
    ledger = ViolationLedger(open_after=3, close_after=3)
    last = feed(ledger, [compliant()] * 20)
    ledger.close_all(last)
    assert ledger.incidents == []


def test_two_workers_tracked_independently():
    ledger = ViolationLedger(open_after=3, close_after=3)
    frames = [FakeResult(ids=[1, 2], classes=[2, 2], confs=[0.7, 0.9])] * 8 + [empty()] * 5
    last = feed(ledger, frames)
    ledger.close_all(last)
    assert len(ledger.incidents) == 2, f"expected 2 incidents, got {len(ledger.incidents)}"
    assert {i.track_id for i in ledger.incidents} == {1, 2}


def test_peak_confidence_is_retained():
    ledger = ViolationLedger(open_after=3, close_after=3)
    confs = [0.3, 0.4, 0.95, 0.5, 0.6, 0.4]
    last = feed(ledger, [violating(conf=c) for c in confs] + [empty()] * 5)
    ledger.close_all(last)
    assert abs(ledger.incidents[0].peak_conf - 0.95) < 1e-3


def test_open_incident_is_flushed_at_end_of_stream():
    """A worker still violating when the video ends must still be logged."""
    ledger = ViolationLedger(open_after=3, close_after=10)
    last = feed(ledger, [violating()] * 10)
    ledger.close_all(last)
    assert len(ledger.incidents) == 1
    assert ledger.incidents[0].last_frame == 9


def test_min_conf_filters_weak_detections():
    ledger = ViolationLedger(open_after=3, close_after=3, min_conf=0.5)
    last = feed(ledger, [violating(conf=0.2)] * 10 + [empty()] * 5)
    ledger.close_all(last)
    assert ledger.incidents == []


def test_triage_order_is_by_severity():
    ledger = ViolationLedger(open_after=1, close_after=1)
    # track 1 = NO-Hardhat (severity 3), track 2 = NO-Safety Vest (severity 2)
    ledger._states.clear()
    names = {2: "NO-Hardhat", 4: "NO-Safety Vest"}
    r = FakeResult()
    r.names = names
    r.boxes = FakeBoxes([1, 2], [4, 2], [0.8, 0.8])  # vest first, hardhat second
    ledger.update(0, r)
    ledger.update(1, r)
    ledger.close_all(1)
    order = [i.label for i in ledger.sorted_incidents()]
    assert order[0] == "NO-Hardhat", f"most severe must sort first, got {order}"


def test_csv_and_summary(tmp_path: Path | None = None):
    out = Path(tmp_path) if tmp_path else Path(__file__).resolve().parent / "_tmp"
    out.mkdir(parents=True, exist_ok=True)
    ledger = ViolationLedger(open_after=3, close_after=3)
    last = feed(ledger, [violating()] * 15 + [empty()] * 5)
    ledger.close_all(last)

    csv_path = ledger.to_csv(out / "violations.csv", fps=15.0)
    text = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2, "header + one incident row"
    assert "NO-Hardhat" in text[1]

    summary = ledger.summary(fps=15.0)
    assert summary["incidents"] == 1
    assert summary["workers_involved"] == 1
    assert summary["total_exposure_s"] == 1.0, summary  # 15 frames @ 15 fps

    csv_path.unlink()
    if not tmp_path:
        out.rmdir()


def test_crop_respects_row_column_indexing():
    """Crops must use frame[y1:y2, x1:x2] - the Module 0 indexing trap."""
    from ppe_monitor.compliance import _crop

    frame = np.zeros((100, 200, 3), np.uint8)  # h=100, w=200
    frame[10:90, 10:50] = 255
    crop = _crop(frame, np.array([10.0, 10.0, 50.0, 90.0]), pad=0.0)
    assert crop.shape[0] == 80 and crop.shape[1] == 40, f"got {crop.shape}"
    assert crop.mean() == 255.0, "crop landed on the wrong region"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
