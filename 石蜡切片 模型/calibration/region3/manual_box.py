"""Parse manual cambium rectangles from 人工标定/第三区域."""

from __future__ import annotations

import cv2
import numpy as np


def parse_manual_cambium_box(orig: np.ndarray, manual: np.ndarray) -> tuple[int, int, int, int] | None:
    """Extract magenta filled rectangle (x0,y0,x1,y1) from manual vs original."""
    if orig.shape[:2] != manual.shape[:2]:
        manual = cv2.resize(manual, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff = cv2.absdiff(manual, orig)
    changed = np.any(diff > 20, axis=2)
    b, g, r = cv2.split(manual)
    # Strict magenta (matches filled-box extent ~30–45% height)
    mag = (r > 100) & (b > 80) & (g < 60) & changed
    h, w = mag.shape
    mag[: int(h * 0.06), int(w * 0.75) :] = False

    ys, xs = np.where(mag)
    if len(xs) < 100:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if (x1 - x0) < w * 0.35 or (y1 - y0) < h * 0.12:
        return None
    return x0, y0, x1, y1
