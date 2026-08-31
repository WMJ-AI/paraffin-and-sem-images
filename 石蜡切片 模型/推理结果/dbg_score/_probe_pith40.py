"""Probe pith detectors vs manual green starts on first 40 images."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from calibration.io_util import imread, parse_name
from calibration.region1_coords import load_coord_json
from calibration.stem import detect_pith, detect_stem_mask


def pale_pith(img, stem_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    like = ((s < 90) & (v > 130) & (gray > 125) & (stem_mask > 0)).astype(np.uint8) * 255
    like = cv2.morphologyEx(like, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    like = cv2.morphologyEx(like, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    if int(cv2.countNonZero(like)) < 80:
        dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return float(x), float(y)
    dist = cv2.distanceTransform(like, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y)


def inner_dark_pith(img, stem_mask):
    """Darker core, but only in the inner 45% of the stem disk."""
    dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    stem_r = float(dist.max())
    inner = (dist >= 0.55 * stem_r).astype(np.uint8) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    score = dist * (255.0 - gray) * (inner > 0)
    if float(score.max()) <= 0:
        return detect_pith(img, stem_mask)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(x), float(y)


def main():
    base = ROOT
    payload = load_coord_json(base / "推理结果" / "region1_coords_first40.json")
    orig_dir = base / "原图" / "第一区域"
    names = {
        2: ["dt", "dark", "pale", "inner_dark"],
        4: ["dt", "dark", "pale", "inner_dark"],
    }
    buckets = {2: {k: [] for k in names[2]}, 4: {k: [] for k in names[4]}}
    ring_from_green = {2: [], 4: []}

    for fname, rec in payload["images"].items():
        orig = imread(orig_dir / fname)
        if orig is None:
            continue
        info = detect_stem_mask(orig)
        if info is None:
            continue
        stem, _ = info
        mag = int(rec["magnification"])
        gx, gy = rec["green_start"]["x"], rec["green_start"]["y"]
        lx, ly = rec["layer_start"]["x"], rec["layer_start"]["y"]
        sr = rec["stem_r_px"]
        dist = cv2.distanceTransform(stem, cv2.DIST_L2, 5)
        yx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        dt = (float(yx[1]), float(yx[0]))
        dark = detect_pith(orig, stem)
        pale = pale_pith(orig, stem)
        inn = inner_dark_pith(orig, stem)
        cands = {"dt": dt, "dark": dark, "pale": pale, "inner_dark": inn}
        for k, p in cands.items():
            err = np.hypot(p[0] - gx, p[1] - gy) / sr
            buckets[mag][k].append(err)
        ring_from_green[mag].append(np.hypot(lx - gx, ly - gy) / sr)

    for mag in (2, 4):
        print(f"===== {mag}X pith error vs manual green_start (frac of stem_r) =====")
        for k, vals in buckets[mag].items():
            arr = np.array(vals)
            print(
                f"  {k:11s}  med={np.median(arr):.3f}  mean={arr.mean():.3f}  "
                f"p75={np.percentile(arr,75):.3f}  <=0.08={float((arr<=0.08).mean()):.0%}"
            )
        arr = np.array(ring_from_green[mag])
        print(
            f"  layer_start dist to green_start: med={np.median(arr):.3f}  "
            f"p25={np.percentile(arr,25):.3f} p75={np.percentile(arr,75):.3f}"
        )


if __name__ == "__main__":
    main()
