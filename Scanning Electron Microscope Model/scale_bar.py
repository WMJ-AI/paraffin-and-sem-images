"""Detect green 20 µm scale bar in bottom-left of SEM images."""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_rgb_gray(path) -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(path)
    rgb = np.array(im.convert("RGB"))
    gray = np.array(im.convert("L"))
    return rgb, gray


def detect_scale_um_per_px(
    rgb: np.ndarray,
    known_um: float = 20.0,
    fallback: float = 0.2233,
) -> dict:
    """Return µm/px from bottom-left green horizontal bar."""
    h, w = rgb.shape[:2]
    y0, y1 = int(h * 0.82), h
    x0, x1 = 0, int(w * 0.45)
    crop = rgb[y0:y1, x0:x1]
    r = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    b = crop[:, :, 2].astype(np.int16)
    green = (g > 140) & (g > r + 40) & (g > b + 40)
    if green.sum() < 20:
        return {
            "um_per_px": fallback,
            "bar_width_px": None,
            "method": "fallback",
            "known_um": known_um,
        }

    # Keep the densest horizontal band (the bar itself, not text).
    row_counts = green.sum(axis=1)
    best = int(np.argmax(row_counts))
    band = slice(max(0, best - 2), min(green.shape[0], best + 3))
    xs = np.where(green[band].any(axis=0))[0]
    if len(xs) < 10:
        ys, xs_all = np.where(green)
        width = int(xs_all.max() - xs_all.min() + 1)
    else:
        # Use continuous run near median to avoid text glyphs.
        runs = []
        start = int(xs[0])
        prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x - prev > 3:
                runs.append((start, prev))
                start = x
            prev = x
        runs.append((start, prev))
        start, end = max(runs, key=lambda t: t[1] - t[0])
        width = end - start + 1

    if width < 40:
        return {
            "um_per_px": fallback,
            "bar_width_px": width,
            "method": "fallback_narrow",
            "known_um": known_um,
        }

    um_per_px = known_um / float(width)
    return {
        "um_per_px": um_per_px,
        "bar_width_px": width,
        "method": "green_bar",
        "known_um": known_um,
    }
