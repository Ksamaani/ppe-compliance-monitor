"""Reproducibly fetch the sample assets used by notebooks 01 and 02.

Images come from Wikimedia Commons through the MediaWiki API (no API key, no
account). Every downloaded file's source page, author and licence are written to
logs/01_assets.log so attribution is preserved in the repository.

The site video is a 34-second window cut out of a CC BY-SA 4.0 construction
timelapse. The full source is ~169 MB, so the cut is scripted rather than
committed as a pipeline: see CLIP_* constants below for the exact recipe.

Usage:
    python scripts/fetch_assets.py                 # images + video (downloads source)
    python scripts/fetch_assets.py --images-only
    python scripts/fetch_assets.py --reuse-clip PATH   # skip the 169 MB download
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
VIDEO_DIR = ROOT / "data" / "videos"
LOG_PATH = ROOT / "logs" / "01_assets.log"

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "ppe-compliance-monitor/0.1 (SDAIA Academy capstone)"}
MIN_WIDTH = 1200

# (search term, how many images wanted) - chosen to cover the PPE failure modes
# the model has to handle: helmets on/off, hi-vis vests, distance, occlusion.
IMAGE_QUERIES: list[tuple[str, int]] = [
    ("construction workers hard hats site", 3),
    ("building construction site crane workers", 2),
    ("workers pouring concrete boom pump", 1),
    ("FEMA roofer working on a home in Oklahoma", 1),
    ("Seabee welds structure construction", 1),
]

# --- Video clip recipe -------------------------------------------------------
CLIP_TITLE = "File:Building construction Moira Close Broadwater Farm Haringey 2025 16.webm"
CLIP_PAGE = (
    "https://commons.wikimedia.org/wiki/"
    "File:Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_16.webm"
)
CLIP_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/5/54/"
    "Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_16.webm"
)
CLIP_AUTHOR = "Acabashi"
CLIP_LICENSE = "CC BY-SA 4.0"
CLIP_START_S = 236  # static zoomed shot of a formwork deck
CLIP_DURATION_S = 34  # ends before the camera zooms out at ~272 s
CLIP_SCALE_W = 1280
CLIP_FPS = 15
CLIP_DEST = VIDEO_DIR / "site_clip.mp4"


def api_search(term: str, limit: int = 10) -> list[dict]:
    """Search Commons for bitmap files matching a term."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{term} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1600,
    }
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    out = []
    for page in sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 999)):
        info = (page.get("imageinfo") or [None])[0]
        if not info or info.get("width", 0) < MIN_WIDTH:
            continue
        meta = info.get("extmetadata", {})
        out.append(
            {
                "title": page.get("title", ""),
                "url": info.get("thumburl") or info.get("url"),
                "descriptionurl": info.get("descriptionurl", ""),
                "artist": _strip_html(meta.get("Artist", {}).get("value", "")),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
            }
        )
    return out


def _strip_html(value: str) -> str:
    """Commons returns HTML fragments for the artist field."""
    out, depth = [], 0
    for ch in value:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())[:120]


def download(url: str, dest: Path, retries: int = 3) -> int:
    """Download a URL to dest, backing off on rate limits."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return dest.stat().st_size
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def fetch_images(log: list[str]) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    idx = 1
    for term, want in IMAGE_QUERIES:
        got = 0
        log.append(f"\n[search] {term!r} -> want {want}")
        try:
            candidates = api_search(term)
        except Exception as exc:  # noqa: BLE001 - network failures must not abort the rest
            log.append(f"  ERROR during search: {exc}")
            continue

        for cand in candidates:
            if got >= want:
                break
            dest = IMAGE_DIR / f"site_{idx:02d}.jpg"
            if dest.exists():
                log.append(f"  SKIP (exists): {dest.name}")
                idx, got = idx + 1, got + 1
                continue
            try:
                size = download(cand["url"], dest)
                log.append(f"  OK  {dest.name} <- {cand['title']} ({size:,} bytes)")
                log.append(f"      licence: {cand['license']} | author: {cand['artist']}")
                log.append(f"      source : {cand['descriptionurl']}")
                idx, got = idx + 1, got + 1
                time.sleep(4)  # be polite to the Commons API
            except Exception as exc:  # noqa: BLE001
                log.append(f"  FAIL {cand['title']}: {exc}")
                time.sleep(2)


def cut_clip(source: Path, dest: Path) -> None:
    """Cut, downscale and re-encode the working window out of the source video."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(CLIP_START_S),
        "-t", str(CLIP_DURATION_S),
        "-i", str(source),
        "-vf", f"scale={CLIP_SCALE_W}:-2,fps={CLIP_FPS}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def fetch_video(log: list[str], reuse_clip: Path | None) -> None:
    log.append("\n[video]")
    log.append(f"  title  : {CLIP_TITLE}")
    log.append(f"  author : {CLIP_AUTHOR} | licence: {CLIP_LICENSE}")
    log.append(f"  source : {CLIP_PAGE}")
    log.append(
        f"  window : {CLIP_START_S}-{CLIP_START_S + CLIP_DURATION_S} s "
        f"-> {CLIP_SCALE_W}px wide @ {CLIP_FPS} fps"
    )
    log.append(
        "  why    : static zoomed shot of a formwork deck; 2-4 workers walk across "
        "the frame carrying materials, hard hats of several colours clearly visible, "
        "and the camera zoom-out at ~272 s is excluded."
    )

    if CLIP_DEST.exists():
        log.append(f"  SKIP (exists): {CLIP_DEST.name} ({CLIP_DEST.stat().st_size:,} bytes)")
        return

    if reuse_clip is not None:
        if not reuse_clip.exists():
            log.append(f"  ERROR: --reuse-clip path not found: {reuse_clip}")
            return
        CLIP_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reuse_clip, CLIP_DEST)
        log.append(
            f"  REUSED a previously-cut copy of this exact window from {reuse_clip} "
            f"({CLIP_DEST.stat().st_size:,} bytes) instead of re-downloading the "
            "169 MB source. Drop --reuse-clip to reproduce it from scratch."
        )
        return

    tmp = ROOT / "data" / "videos" / "_source.webm"
    log.append(f"  downloading source (~169 MB) -> {tmp.name}")
    size = download(CLIP_URL, tmp)
    log.append(f"  downloaded {size:,} bytes; cutting with ffmpeg")
    cut_clip(tmp, CLIP_DEST)
    tmp.unlink(missing_ok=True)
    log.append(f"  OK  {CLIP_DEST.name} ({CLIP_DEST.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-only", action="store_true", help="skip the video")
    parser.add_argument("--video-only", action="store_true", help="skip the images")
    parser.add_argument(
        "--reuse-clip",
        type=Path,
        default=None,
        help="path to an already-cut copy of the clip, to avoid the 169 MB download",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    log = [f"=== ASSET FETCH {datetime.now(timezone.utc).isoformat(timespec='seconds')} ==="]

    if not args.video_only:
        fetch_images(log)
    if not args.images_only:
        fetch_video(log, args.reuse_clip)

    images = sorted(IMAGE_DIR.glob("*.jpg"))
    log.append(f"\nImages on disk : {len(images)}")
    for img in images:
        log.append(f"  {img.name:16s} {img.stat().st_size:>10,} bytes")
    log.append(f"Video on disk  : {'yes' if CLIP_DEST.exists() else 'no'}")

    report = "\n".join(log)
    print(report)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
