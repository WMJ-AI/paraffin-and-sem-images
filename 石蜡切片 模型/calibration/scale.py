"""Scale bar detection and pixel-to-micron conversion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Printed scale-bar lengths (µm) by magnification. Region-1 2X/4X bars
# read "500 µm" / "250 µm" on the micrographs (not 5000 / 2500).
MAGNIFICATION_TO_MICRONS: dict[int, float] = {
    2: 500.0,
    4: 250.0,
    10: 100.0,
    20: 50.0,
    40: 25.0,
}

# Typical black-bar width on this camera (region-2 10X is ~412 px).
FALLBACK_BAR_PX = 412.0


@dataclass
class ScaleInfo:
    bar_pixels: float
    microns: float
    um_per_pixel: float


def detect_scale_bar_pixels(img: np.ndarray) -> float | None:
    """Detect the printed black horizontal scale bar (bottom-right ROI)."""
    h, w = img.shape[:2]
    roi = img[int(h * 0.84) :, int(w * 0.68) :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    black = (gray < 80).astype(np.uint8) * 255
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(black)
    in_range: list[float] = []
    for i in range(1, n_lab):
        _x, _y, bw, bh, _area = stats[i]
        # 2X bars ~439 px, 4X ~414 px. Height can include the "250 µm" label.
        if not (350 <= bw <= 500 and 3 <= bh <= 130):
            continue
        if bw / max(bh, 1) < 3:
            continue
        if bw > roi.shape[1] * 0.98:
            continue
        in_range.append(float(bw))
    if in_range:
        return max(in_range)
    return None


def get_scale_info(img: np.ndarray, magnification: int) -> ScaleInfo | None:
    bar_px = detect_scale_bar_pixels(img)
    if bar_px is None or bar_px <= 0:
        bar_px = FALLBACK_BAR_PX
    microns = MAGNIFICATION_TO_MICRONS.get(magnification, 500.0)
    return ScaleInfo(bar_pixels=bar_px, microns=microns, um_per_pixel=microns / bar_px)


def pixels_to_microns(pixels: float, scale: ScaleInfo) -> float:
    return pixels * scale.um_per_pixel
