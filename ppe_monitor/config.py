"""Project-wide configuration for the PPE compliance monitor.

Single source of truth for class semantics, thresholds and paths so the
notebooks, the Streamlit app and the scripts cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
VIDEO_DIR = DATA_DIR / "videos"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

SITE_CLIP = VIDEO_DIR / "site_clip.mp4"

# --- Dataset class semantics -------------------------------------------------
# Roboflow "Construction Site Safety" via Kaggle. Order is the dataset's own
# class order and MUST match data.yaml used for training.
CLASS_NAMES: tuple[str, ...] = (
    "Hardhat",          # 0
    "Mask",             # 1
    "NO-Hardhat",       # 2
    "NO-Mask",          # 3
    "NO-Safety Vest",   # 4
    "Person",           # 5
    "Safety Cone",      # 6
    "Safety Vest",      # 7
    "machinery",        # 8
    "vehicle",          # 9
)

# A violation is a POSITIVE detection of a NO-* class, never the absence of a
# box. This is what makes false-negative analysis meaningful: the model has to
# actively assert non-compliance.
VIOLATION_CLASSES: frozenset[str] = frozenset({"NO-Hardhat", "NO-Mask", "NO-Safety Vest"})

# Severity ranking used to sort the ledger. A missing hard hat is the failure
# mode that kills people, so it outranks the others.
SEVERITY: dict[str, int] = {
    "NO-Hardhat": 3,
    "NO-Safety Vest": 2,
    "NO-Mask": 1,
}

# --- Operating point ---------------------------------------------------------
# Provisional until notebooks/03_evaluation.ipynb sweeps the threshold on the
# held-out test split; that notebook writes the justified value back here.
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50

# Debounce for the incident state machine (see compliance.py).
OPEN_AFTER_FRAMES = 5    # consecutive violating frames before an incident opens
CLOSE_AFTER_FRAMES = 15  # consecutive clear frames before it closes

# COCO fallback weights used before the custom model exists.
FALLBACK_WEIGHTS = "yolo11n.pt"


def resolve_weights() -> tuple[str, bool]:
    """Return (weights_path, is_custom).

    Prefers the custom PPE model when it has been trained, otherwise falls back
    to COCO-pretrained weights so every notebook runs end-to-end at any stage of
    the project.
    """
    best = MODEL_DIR / "best.pt"
    if best.exists():
        return str(best), True
    return FALLBACK_WEIGHTS, False
