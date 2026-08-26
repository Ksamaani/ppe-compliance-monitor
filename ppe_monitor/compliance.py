"""Per-track PPE violation ledger.

Raw per-frame detections flicker: a missing hard hat is detected on frame 41,
lost on 42, back on 43. Logging that directly produces three "incidents" for one
event and floods a safety officer with noise.

This module converts the detection stream into *incidents* using a small state
machine per track ID: a violation must persist OPEN_AFTER_FRAMES consecutive
frames before an incident opens, and must be clear for CLOSE_AFTER_FRAMES before
it closes. The result is an auditable CSV a human can act on, plus one cropped
evidence image per incident.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CLOSE_AFTER_FRAMES,
    ENFORCED_CLASSES,
    OPEN_AFTER_FRAMES,
    SEVERITY,
)


@dataclass
class Incident:
    """One continuous period of non-compliance by one tracked worker."""

    track_id: int
    label: str
    first_frame: int
    last_frame: int
    peak_conf: float
    evidence_path: str = ""

    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    def duration_seconds(self, fps: float) -> float:
        return self.duration_frames() / fps if fps else 0.0

    def severity(self) -> int:
        return SEVERITY.get(self.label, 0)


@dataclass
class _TrackState:
    """Debounce counters for a single (track_id, label) pair."""

    violating_streak: int = 0
    clear_streak: int = 0
    incident: Incident | None = None
    best_conf: float = 0.0
    best_crop: np.ndarray | None = field(default=None, repr=False)


class ViolationLedger:
    """Accumulates debounced PPE violations across a video.

    Usage::

        ledger = ViolationLedger(crop_dir=Path("outputs/crops"))
        for frame_idx, result in enumerate(model.track(..., stream=True)):
            ledger.update(frame_idx, result, result.orig_img)
        ledger.close_all(frame_idx)
        ledger.to_csv(Path("outputs/violations.csv"), fps=15.0)
    """

    def __init__(
        self,
        crop_dir: Path | None = None,
        open_after: int = OPEN_AFTER_FRAMES,
        close_after: int = CLOSE_AFTER_FRAMES,
        min_conf: float = 0.0,
        violation_classes: frozenset[str] = ENFORCED_CLASSES,
    ) -> None:
        self.crop_dir = crop_dir
        self.open_after = open_after
        self.close_after = close_after
        self.min_conf = min_conf
        self.violation_classes = violation_classes
        self._states: dict[tuple[int, str], _TrackState] = {}
        self.incidents: list[Incident] = []
        if crop_dir is not None:
            crop_dir.mkdir(parents=True, exist_ok=True)

    # -- ingest ---------------------------------------------------------------

    def update(self, frame_idx: int, result, frame: np.ndarray | None = None) -> None:
        """Feed one tracked frame of Ultralytics results into the ledger."""
        seen_this_frame: set[tuple[int, str]] = set()
        boxes = getattr(result, "boxes", None)

        if boxes is not None and boxes.id is not None:
            names = result.names
            ids = boxes.id.int().tolist()
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.cpu().numpy()

            for track_id, cls_idx, conf, box in zip(ids, classes, confs, xyxy):
                label = names[int(cls_idx)]
                if label not in self.violation_classes or conf < self.min_conf:
                    continue
                key = (int(track_id), label)
                seen_this_frame.add(key)
                self._register_violation(key, frame_idx, float(conf), box, frame)

        # Anything previously violating but absent this frame accrues clear time.
        for key, state in list(self._states.items()):
            if key not in seen_this_frame:
                self._register_clear(key, state, frame_idx)

    def _register_violation(
        self,
        key: tuple[int, str],
        frame_idx: int,
        conf: float,
        box: np.ndarray,
        frame: np.ndarray | None,
    ) -> None:
        state = self._states.setdefault(key, _TrackState())
        state.violating_streak += 1
        state.clear_streak = 0

        # Keep the most confident crop as the evidence image for this incident.
        if conf > state.best_conf and frame is not None:
            state.best_conf = conf
            state.best_crop = _crop(frame, box)

        if state.incident is None and state.violating_streak >= self.open_after:
            track_id, label = key
            # Backdate to when the streak started, not when it was confirmed.
            state.incident = Incident(
                track_id=track_id,
                label=label,
                first_frame=frame_idx - state.violating_streak + 1,
                last_frame=frame_idx,
                peak_conf=conf,
            )
        elif state.incident is not None:
            state.incident.last_frame = frame_idx
            state.incident.peak_conf = max(state.incident.peak_conf, conf)

    def _register_clear(self, key: tuple[int, str], state: _TrackState, frame_idx: int) -> None:
        state.clear_streak += 1
        state.violating_streak = 0
        if state.clear_streak >= self.close_after:
            self._finalise(key, state)
            del self._states[key]

    def _finalise(self, key: tuple[int, str], state: _TrackState) -> None:
        if state.incident is None:
            return
        incident = state.incident
        if self.crop_dir is not None and state.best_crop is not None and state.best_crop.size:
            safe_label = incident.label.replace(" ", "_").replace("-", "_")
            name = f"id{incident.track_id:03d}_{safe_label}_f{incident.first_frame:05d}.jpg"
            path = self.crop_dir / name
            cv2.imwrite(str(path), state.best_crop)
            incident.evidence_path = path.name
        self.incidents.append(incident)
        state.incident = None

    def close_all(self, final_frame: int) -> None:
        """Flush every still-open incident at end of stream."""
        for key, state in list(self._states.items()):
            if state.incident is not None:
                state.incident.last_frame = max(state.incident.last_frame, final_frame)
            self._finalise(key, state)
        self._states.clear()

    # -- report ---------------------------------------------------------------

    def sorted_incidents(self) -> list[Incident]:
        """Most severe first, then longest-running - the triage order."""
        return sorted(
            self.incidents,
            key=lambda i: (-i.severity(), -i.duration_frames(), i.first_frame),
        )

    def to_csv(self, path: Path, fps: float) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "incident_id",
                    "track_id",
                    "violation",
                    "severity",
                    "first_frame",
                    "last_frame",
                    "duration_frames",
                    "duration_s",
                    "peak_conf",
                    "evidence",
                ]
            )
            for n, inc in enumerate(self.sorted_incidents(), start=1):
                writer.writerow(
                    [
                        n,
                        inc.track_id,
                        inc.label,
                        inc.severity(),
                        inc.first_frame,
                        inc.last_frame,
                        inc.duration_frames(),
                        f"{inc.duration_seconds(fps):.2f}",
                        f"{inc.peak_conf:.3f}",
                        inc.evidence_path,
                    ]
                )
        return path

    def summary(self, fps: float) -> dict[str, object]:
        by_label: dict[str, int] = {}
        for inc in self.incidents:
            by_label[inc.label] = by_label.get(inc.label, 0) + 1
        total_s = sum(i.duration_seconds(fps) for i in self.incidents)
        return {
            "incidents": len(self.incidents),
            "workers_involved": len({i.track_id for i in self.incidents}),
            "by_violation": by_label,
            "total_exposure_s": round(total_s, 2),
        }


def _crop(frame: np.ndarray, box: np.ndarray, pad: float = 0.15) -> np.ndarray:
    """Crop a padded region around a box.

    NumPy indexes [row, col] = [y, x] while boxes are (x1, y1, x2, y2) - the
    exact trap covered in Module 0 of the course.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w, x2 + pad * bw))
    y2 = int(min(h, y2 + pad * bh))
    return frame[y1:y2, x1:x2].copy()
