"""Side-by-side merge: manual (left) vs auto on original (right)."""

from __future__ import annotations

import cv2
import numpy as np


def stitch_horizontal(
    left: np.ndarray,
    right: np.ndarray,
    gap: int = 12,
    left_label: str = "人工标定",
    right_label: str = "自动标定",
) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + gap + right.shape[1]
    canvas = np.full((h, w, 3), 255, np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, min(left.shape[1], right.shape[1]) / 3000)
    thickness = max(2, int(round(scale * 2)))
    y = int(40 * scale + 20)
    cv2.putText(canvas, left_label, (20, y), font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, left_label, (20, y), font, scale, (0, 128, 0), thickness, cv2.LINE_AA)
    x2 = left.shape[1] + gap + 20
    cv2.putText(canvas, right_label, (x2, y), font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, right_label, (x2, y), font, scale, (0, 0, 200), thickness, cv2.LINE_AA)
    return canvas
