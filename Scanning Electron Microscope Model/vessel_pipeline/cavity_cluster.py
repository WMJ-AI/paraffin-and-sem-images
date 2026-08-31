"""Reject contours that enclose a cluster of small cavities (not one lumen)."""
from __future__ import annotations

import cv2
import numpy as np

from segment import Vessel


def is_multi_cavity_cluster(
    gray: np.ndarray,
    v: Vessel,
    um_per_px: float,
    *,
    min_blobs: int = 3,
    max_top1_frac: float = 0.50,
    min_mean_gray: float = 48.0,
) -> bool:
    """
    True if the contour wraps several small dark pits separated by brighter tissue
    rather than one vessel lumen.

    Teaching (7(5) V03): green circle around a bunch of small cavities next to a
    real vessel — must not count as a lumen / pair partner.
    """
    if gray is None or um_per_px <= 0:
        return False
    h, w = gray.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [v.contour], -1, 255, -1)
    area = int(cv2.countNonZero(m))
    if area < 40:
        return False
    vals = gray[m > 0].astype(np.float64)
    if vals.size < 40:
        return False
    mean_g = float(vals.mean())
    thr = float(np.percentile(vals, 40))
    dark = ((gray.astype(np.float64) <= thr) & (m > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        1,
    )
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = 6.0 / (um_per_px**2)
    blob_areas = sorted(
        (float(cv2.contourArea(c)) for c in cnts if cv2.contourArea(c) >= min_a),
        reverse=True,
    )
    if len(blob_areas) < min_blobs:
        return False
    top1_frac = blob_areas[0] / max(float(area), 1.0)
    # Cluster: no single dominant lumen + interior not as dark as a true vessel
    if top1_frac <= max_top1_frac and mean_g >= min_mean_gray:
        return True
    # Alternate: many similar-sized blobs, none dominates
    if len(blob_areas) >= 4 and top1_frac <= 0.22:
        return True
    return False


def filter_multi_cavity_clusters(
    vessels: list[Vessel],
    gray: np.ndarray | None,
    um_per_px: float,
) -> list[Vessel]:
    if gray is None or not vessels:
        return vessels
    return [
        v
        for v in vessels
        if not is_multi_cavity_cluster(gray, v, um_per_px)
    ]
