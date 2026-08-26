"""Streamlit front end for the PPE compliance monitor.

Deliverable 5. Two modes:

  * Image  - upload a site photo, get detections and a compliance verdict.
  * Video  - upload a clip, run tracking through the debounced ViolationLedger,
             get a ranked incident table with evidence crops.

Serves the custom-trained model, its ONNX export, or COCO-pretrained weights as
a fallback so the app runs even before training has happened.

Run:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from ppe_monitor.compliance import ViolationLedger
from ppe_monitor.config import (
    CLOSE_AFTER_FRAMES,
    DEFAULT_CONF,
    DEFAULT_IOU,
    ENFORCED_CLASSES,
    FALLBACK_WEIGHTS,
    MODEL_DIR,
    OPEN_AFTER_FRAMES,
    SEVERITY,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PPE Compliance Monitor",
    page_icon="⛑",
    layout="wide",
    # The model picker and the confidence threshold are the two controls that
    # change the verdict, so they must be visible on arrival rather than behind
    # a toggle - Streamlit's "auto" state hides them entirely on load.
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model…")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def available_models() -> dict[str, str]:
    """Model path -> human label, best first."""
    options: dict[str, str] = {}
    if (MODEL_DIR / "best.pt").exists():
        options[str(MODEL_DIR / "best.pt")] = "Custom PPE model (PyTorch)"
    if (MODEL_DIR / "best.onnx").exists():
        options[str(MODEL_DIR / "best.onnx")] = "Custom PPE model (ONNX, CPU-optimised)"
    options[FALLBACK_WEIGHTS] = "COCO-pretrained yolo11n (fallback — no PPE classes)"
    return options


def detections_frame(result) -> pd.DataFrame:
    rows = []
    for box in result.boxes:
        label = result.names[int(box.cls)]
        rows.append(
            {
                "class": label,
                "violation": label in ENFORCED_CLASSES,
                "severity": SEVERITY.get(label, 0),
                "confidence": round(float(box.conf), 3),
                "x1": int(box.xyxy[0][0]),
                "y1": int(box.xyxy[0][1]),
                "x2": int(box.xyxy[0][2]),
                "y2": int(box.xyxy[0][3]),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["severity", "confidence"], ascending=False) if len(df) else df


def verdict(df: pd.DataFrame, is_custom: bool) -> None:
    if not is_custom:
        st.info(
            "Running on COCO-pretrained weights, which have no PPE classes — this locates "
            "people only. Train the custom model (notebook 04) for compliance verdicts."
        )
        return
    if not len(df):
        st.info("Nothing detected above the confidence threshold.")
        return

    violations = df[df["violation"]]
    if len(violations):
        kinds = ", ".join(sorted(violations["class"].unique()))
        st.error(f"**NON-COMPLIANT** — {len(violations)} violation(s) detected: {kinds}")
    else:
        st.success("**Compliant** — no PPE violations detected in this frame.")


# ---------------------------------------------------------------- sidebar ----

st.sidebar.title("⛑ Settings")

models = available_models()
weights = st.sidebar.selectbox("Model", list(models), format_func=lambda p: models[p])
is_custom = "best" in Path(weights).stem

conf = st.sidebar.slider(
    "Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
    help="Lower catches more violations at the cost of more false alarms. "
         "The shipped default is chosen in notebook 03 by sweeping this on the held-out test split.",
)
iou = st.sidebar.slider("NMS IoU", 0.1, 0.9, DEFAULT_IOU, 0.05)

if not is_custom:
    st.sidebar.warning("No custom weights found in `models/`. Using COCO fallback.")

st.sidebar.divider()
st.sidebar.caption(
    "Capstone — Computer Vision for Developers with Ultralytics · "
    "SDAIA Academy, cohort 24–28 August 2026 · "
    "[github.com/SDAIAAcademy](https://github.com/SDAIAAcademy)"
)

# ------------------------------------------------------------------- main ----

st.title("PPE Compliance Monitor")
st.caption(
    "Construction-site safety compliance from camera imagery — detections, and a debounced "
    "violation ledger for video."
)

model = load_model(weights)
image_tab, video_tab = st.tabs(["📷 Image", "🎬 Video"])

with image_tab:
    upload = st.file_uploader("Site photo", type=["jpg", "jpeg", "png"], key="img")
    if upload is None:
        st.info("Upload an image to run the pipeline. Samples live in `data/images/`.")
    else:
        img = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not decode that image.")
            st.stop()

        result = model.predict(source=img, conf=conf, iou=iou, verbose=False)[0]
        df = detections_frame(result)

        verdict(df, is_custom)

        left, right = st.columns(2)
        left.subheader("Input")
        left.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        right.subheader("Detections")
        right.image(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), use_container_width=True)

        if len(df):
            st.subheader(f"{len(df)} detection(s)")
            st.dataframe(df, use_container_width=True, hide_index=True)

with video_tab:
    st.caption(
        f"Tracking runs on CPU at roughly 4 frames/second. A violation must persist "
        f"{OPEN_AFTER_FRAMES} frames to open an incident and be clear for {CLOSE_AFTER_FRAMES} "
        "to close it, so one flickering detection does not become an alert."
    )
    clip = st.file_uploader("Site clip", type=["mp4", "avi", "mov", "mkv"], key="vid")
    max_frames = st.slider("Frames to process", 30, 600, 150, 30)

    if clip is None:
        st.info("Upload a clip to build a violation ledger. A sample lives in `data/videos/`.")
    elif st.button("Run analysis", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=Path(clip.name).suffix, delete=False) as fh:
            fh.write(clip.getvalue())
            tmp_path = Path(fh.name)

        cap = cv2.VideoCapture(str(tmp_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames, max_frames)

        crop_dir = Path(tempfile.mkdtemp()) / "crops"
        ledger = ViolationLedger(crop_dir=crop_dir)

        progress = st.progress(0.0, text="Tracking…")
        preview = st.empty()
        idx = -1
        try:
            while cap.isOpened() and idx + 1 < total:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                result = model.track(frame, persist=True, conf=conf, iou=iou, verbose=False)[0]
                ledger.update(idx, result, frame)
                if idx % 15 == 0:
                    preview.image(
                        cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB),
                        caption=f"frame {idx}",
                        use_container_width=True,
                    )
                progress.progress((idx + 1) / total, text=f"Tracking… frame {idx + 1}/{total}")
        finally:
            cap.release()
            tmp_path.unlink(missing_ok=True)

        ledger.close_all(idx)
        progress.empty()

        summary = ledger.summary(fps=fps)
        cols = st.columns(4)
        cols[0].metric("Incidents", summary["incidents"])
        cols[1].metric("Workers involved", summary["workers_involved"])
        cols[2].metric("Total exposure", f"{summary['total_exposure_s']:.1f} s")
        cols[3].metric("Frames processed", idx + 1)

        incidents = ledger.sorted_incidents()
        if incidents:
            st.subheader("Violation ledger — most severe first")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "track": i.track_id,
                            "violation": i.label,
                            "severity": i.severity(),
                            "first_frame": i.first_frame,
                            "duration_s": round(i.duration_seconds(fps), 2),
                            "peak_conf": round(i.peak_conf, 3),
                        }
                        for i in incidents
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            crops = sorted(crop_dir.glob("*.jpg"))
            if crops:
                st.subheader("Evidence")
                for col, crop in zip(st.columns(min(4, len(crops))), crops[:4]):
                    col.image(
                        cv2.cvtColor(cv2.imread(str(crop)), cv2.COLOR_BGR2RGB),
                        caption=crop.name,
                        use_container_width=True,
                    )
        elif is_custom:
            st.success(
                "No violation incidents. Either the crew is compliant, or the confidence "
                "threshold is too high — try lowering it in the sidebar."
            )
        else:
            st.info("COCO weights cannot detect PPE violations; there is no NO-Hardhat class.")
