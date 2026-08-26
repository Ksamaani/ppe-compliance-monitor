"""OpenCV video pipeline: capture -> YOLO track -> annotate -> write.

Everything here is a real streaming pipeline over a real video file. Frames are
read with cv2.VideoCapture, pushed through Ultralytics tracking or an
ultralytics.solutions module, annotated, and written back out with
cv2.VideoWriter, with periodic stills saved as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .compliance import ViolationLedger


@dataclass
class VideoInfo:
    """Basic properties of an opened video."""

    fps: float
    width: int
    height: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps:.2f} fps, "
            f"{self.frames} frames ({self.duration_s:.1f} s)"
        )


def probe(path: Path | str) -> VideoInfo:
    """Read video properties without consuming the stream."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def make_writer(path: Path | str, info: VideoInfo, fps: float | None = None) -> cv2.VideoWriter:
    """Create an mp4 writer matching the source geometry."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps or info.fps, (info.width, info.height))


def _plot_im(result, fallback: np.ndarray) -> np.ndarray:
    """Get the annotated frame from a result object across Ultralytics versions.

    Solutions return a SolutionResults carrying `plot_im`; plain predict/track
    results expose `.plot()`. Fall back to the raw frame if neither exists.
    """
    im = getattr(result, "plot_im", None)
    if im is not None:
        return im
    if hasattr(result, "plot"):
        return result.plot()
    return fallback


def run_tracking(
    source: Path | str,
    model,
    output_video: Path | str | None = None,
    ledger: ViolationLedger | None = None,
    still_dir: Path | str | None = None,
    still_every: int = 60,
    still_prefix: str = "track",
    conf: float = 0.25,
    iou: float = 0.5,
    max_frames: int | None = None,
    verbose_every: int = 60,
) -> dict[str, object]:
    """Track objects through a video, feeding the compliance ledger.

    Uses model.track(persist=True) frame by frame so that track IDs stay stable
    while we keep full control of the OpenCV loop (read -> process -> write).
    """
    source = Path(source)
    info = probe(source)
    cap = cv2.VideoCapture(str(source))
    writer = make_writer(output_video, info) if output_video else None
    if still_dir is not None:
        Path(still_dir).mkdir(parents=True, exist_ok=True)

    frame_idx = -1
    track_ids: set[int] = set()
    stills: list[str] = []

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

            result = model.track(
                frame, persist=True, conf=conf, iou=iou, verbose=False
            )[0]

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                track_ids.update(boxes.id.int().tolist())

            if ledger is not None:
                ledger.update(frame_idx, result, frame)

            annotated = _plot_im(result, frame)
            if writer is not None:
                writer.write(annotated)

            if still_dir is not None and frame_idx % still_every == 0:
                name = f"{still_prefix}_f{frame_idx:04d}.jpg"
                cv2.imwrite(str(Path(still_dir) / name), annotated)
                stills.append(name)

            if verbose_every and frame_idx % verbose_every == 0:
                print(f"  frame {frame_idx:4d}/{info.frames}  unique tracks so far: {len(track_ids)}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if ledger is not None:
        ledger.close_all(frame_idx)

    return {
        "info": info,
        "frames_processed": frame_idx + 1,
        "unique_tracks": len(track_ids),
        "stills": stills,
        "output_video": str(output_video) if output_video else "",
    }


def run_solution(
    source: Path | str,
    solution,
    output_video: Path | str | None = None,
    still_dir: Path | str | None = None,
    still_every: int = 120,
    still_prefix: str = "solution",
    max_frames: int | None = None,
    pass_frame_count: bool = False,
    on_result=None,
    verbose_every: int = 120,
) -> dict[str, object]:
    """Drive any ultralytics.solutions module over a video.

    ObjectCounter, Heatmap, QueueManager and friends are all callables taking a
    frame; Analytics additionally takes a frame counter, hence pass_frame_count.
    `on_result(frame_idx, result)` is invoked per frame so callers can harvest
    per-frame counts into a time series.
    """
    source = Path(source)
    info = probe(source)
    cap = cv2.VideoCapture(str(source))
    writer = make_writer(output_video, info) if output_video else None
    if still_dir is not None:
        Path(still_dir).mkdir(parents=True, exist_ok=True)

    frame_idx = -1
    stills: list[str] = []
    last_result = None

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

            if pass_frame_count:
                result = solution(frame, frame_idx + 1)
            else:
                result = solution(frame)
            last_result = result

            if on_result is not None:
                on_result(frame_idx, result)

            annotated = _plot_im(result, frame)
            if writer is not None:
                writer.write(annotated)

            if still_dir is not None and frame_idx % still_every == 0:
                name = f"{still_prefix}_f{frame_idx:04d}.jpg"
                cv2.imwrite(str(Path(still_dir) / name), annotated)
                stills.append(name)

            if verbose_every and frame_idx % verbose_every == 0:
                print(f"  frame {frame_idx:4d}/{info.frames}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # Always keep the final frame - it carries the cumulative counts/heat.
    if still_dir is not None and frame_idx >= 0 and last_result is not None:
        name = f"{still_prefix}_final_f{frame_idx:04d}.jpg"
        cv2.imwrite(str(Path(still_dir) / name), _plot_im(last_result, np.zeros((info.height, info.width, 3), np.uint8)))
        stills.append(name)

    return {
        "info": info,
        "frames_processed": frame_idx + 1,
        "stills": stills,
        "last_result": last_result,
        "output_video": str(output_video) if output_video else "",
    }


def head_roi_from_keypoints(kpts: np.ndarray, frame_shape: tuple[int, int], conf_min: float = 0.3) -> tuple[int, int, int, int] | None:
    """Derive the head region from pose keypoints 0-4 (nose, eyes, ears).

    This is the geometric bridge from a generic pose model to PPE compliance:
    a hard hat must occupy the region directly above the facial keypoints, so
    the head ROI is where a helmet check should look.

    kpts: (17, 3) array of (x, y, conf). Returns (x1, y1, x2, y2) or None.
    """
    face = kpts[:5]
    visible = face[face[:, 2] >= conf_min]
    if len(visible) < 2:
        return None

    xs, ys = visible[:, 0], visible[:, 1]
    w = max(float(xs.max() - xs.min()), 1.0)
    h = max(float(ys.max() - ys.min()), 1.0)
    span = max(w, h)

    cx = float(xs.mean())
    cy = float(ys.mean())

    # Widen generously and extend upward: the helmet sits above the eye line.
    half = span * 1.6
    x1 = cx - half
    x2 = cx + half
    y1 = cy - span * 2.4
    y2 = cy + span * 1.2

    fh, fw = frame_shape[:2]
    return (
        int(max(0, x1)),
        int(max(0, y1)),
        int(min(fw, x2)),
        int(min(fh, y2)),
    )
