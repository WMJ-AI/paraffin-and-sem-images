"""Black 100 µm scale bar on region-2 10X images."""

from __future__ import annotations

import cv2
import numpy as np

from calibration.region2.rules import FALLBACK_BAR_PX, SCALE_MICRONS
from calibration.scale import ScaleInfo


def detect_black_scale_bar_px(img: np.ndarray) -> float | None:
    """Long horizontal black bar in the bottom-right (printed 100 µm)."""
    h, w = img.shape[:2]
    y0, x0 = int(h * 0.86), int(w * 0.62)
    roi = img[y0:, x0:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    black = (gray < 50).astype(np.uint8) * 255
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((3, 21), np.uint8))
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(black)
    best: float | None = None
    for i in range(1, n_lab):
        _x, _y, bw, bh, _area = stats[i]
        if not (150 <= bw <= 900 and 8 <= bh <= 55):
            continue
        if bw / max(bh, 1) < 5:
            continue
        if best is None or bw > best:
            best = float(bw)
    return best


def get_region2_scale(img: np.ndarray) -> ScaleInfo:
    bar_px = detect_black_scale_bar_px(img) or FALLBACK_BAR_PX
    return ScaleInfo(bar_pixels=bar_px, microns=SCALE_MICRONS, um_per_pixel=SCALE_MICRONS / bar_px)
