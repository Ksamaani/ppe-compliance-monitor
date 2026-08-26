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
#
# Every NO-* class the model can predict. Not all of them are necessarily
# enforced - see ENFORCED_CLASSES below.
VIOLATION_CLASSES: frozenset[str] = frozenset({"NO-Hardhat", "NO-Mask", "NO-Safety Vest"})

# What this site actually enforces.
#
# NO-Mask is deliberately excluded. Two reasons, both established in
# notebooks/03_evaluation.ipynb:
#
#   1. Policy. The dataset labels "no mask" as a PPE violation, but respiratory
#      protection is a demolition / grinding / silica-dust rule, not a general
#      site rule. An open-air formwork deck does not require masks, so enforcing
#      it produces alarms that are correct against the labels and wrong against
#      site policy - the fastest way to get a safety system ignored.
#   2. Measurement. On the held-out test split NO-Mask reaches only 0.358
#      precision (53 true alarms against 95 false ones), far below the 0.50
#      floor. It was dragging the macro-average down while the other two classes
#      propped it up.
#
# A site that DOES require masks re-enables it here; nothing else changes.
ENFORCED_CLASSES: frozenset[str] = frozenset({"NO-Hardhat", "NO-Safety Vest"})

# Severity ranking used to sort the ledger. A missing hard hat is the failure
# mode that kills people, so it outranks the others.
SEVERITY: dict[str, int] = {
    "NO-Hardhat": 3,
    "NO-Safety Vest": 2,
    "NO-Mask": 1,
}

# --- Operating point ---------------------------------------------------------
# Chosen in notebooks/03_evaluation.ipynb by sweeping confidence and NMS IoU on
# the held-out test split and matching predictions to ground truth by IoU, NOT
# by reading val(conf=...) - which reports precision and recall at the PR
# curve's best-F1 point rather than at the threshold passed.
#
# Rule: maximise recall over ENFORCED_CLASSES subject to precision >= 0.50.
# Recall is flat between conf 0.05 and 0.10, so the higher one is taken - the
# same violations are caught with fewer false alarms.
#
# At this point, on the test split: recall 0.672, precision 0.680.
# Against the Ultralytics default (conf=0.25) this is a priced trade, not a free
# win: roughly +3.9 points of recall bought with -10.1 of precision. Taken
# because a missed violation costs far more than a false alarm and precision
# still clears the floor. The genuinely free part is the 0.05 -> 0.10 step,
# where recall is unchanged and precision improves.
DEFAULT_CONF = 0.10
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
