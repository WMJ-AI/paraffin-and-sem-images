"""Require vessel lumens to be deep-black voids (not gray shallow pits)."""
from __future__ import annotations

import cv2
import numpy as np

from segment import Vessel


def lumen_darkness_metrics(
    gray: np.ndarray,
    v: Vessel,
) -> dict[str, float]:
    h, w = gray.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [v.contour], -1, 255, -1)
    vals = gray[m > 0].astype(np.float64)
    if vals.size < 30:
        return {
            "core_p50": 255.0,
            "very_dark_frac": 0.0,
            "dark_frac": 0.0,
            "contrast": 0.0,
            "mean": 255.0,
        }
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    core = cv2.erode(m, k, iterations=1)
    if int(cv2.countNonZero(core)) < 20:
        core = m
    cvals = gray[core > 0].astype(np.float64)
    outer = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    outer = cv2.bitwise_and(outer, cv2.bitwise_not(m))
    rvals = gray[outer > 0].astype(np.float64)
    core_p50 = float(np.percentile(cvals, 50))
    ring_p50 = float(np.percentile(rvals, 50)) if rvals.size >= 20 else core_p50 + 0.0
    # 自适应「很暗」门槛：固定 38 在偏亮 SEM（腔心~44）上会整批误杀
    very_thr = float(min(38.0, max(28.0, core_p50 - 6.0)))
    dark_thr = float(min(50.0, max(40.0, core_p50 + 2.0)))
    return {
        "core_p50": core_p50,
        "very_dark_frac": float((cvals <= very_thr).mean()),
        "dark_frac": float((vals <= dark_thr).mean()),
        "contrast": float(ring_p50 - core_p50),
        "mean": float(vals.mean()),
    }


def is_deep_black_lumen(
    gray: np.ndarray,
    v: Vessel,
    *,
    max_core_p50: float = 52.0,
    min_very_dark_frac: float = 0.15,
    min_dark_frac: float = 0.35,
    min_contrast: float = 12.0,
    soft_max_core_p50: float = 58.0,
    soft_min_dark_frac: float = 0.40,
    soft_min_contrast: float = 25.0,
) -> bool:
    """
    深黑空腔门控。支持两档：
    - 严格：core 够黑 + 暗像素占比 + 相对壁反差
    - 软通过：SEM 整体偏亮时，只要相对壁反差大且暗占比够，仍计为导管
    """
    m = lumen_darkness_metrics(gray, v)
    strict = (
        m["core_p50"] <= max_core_p50
        and m["very_dark_frac"] >= min_very_dark_frac
        and m["dark_frac"] >= min_dark_frac
        and m["contrast"] >= min_contrast
    )
    if strict:
        return True
    soft = (
        m["core_p50"] <= soft_max_core_p50
        and m["dark_frac"] >= soft_min_dark_frac
        and m["contrast"] >= soft_min_contrast
    )
    return soft


def filter_deep_black_lumens(
    vessels: list[Vessel],
    gray: np.ndarray | None,
    **kwargs,
) -> list[Vessel]:
    if gray is None or not vessels:
        return vessels
    return [v for v in vessels if is_deep_black_lumen(gray, v, **kwargs)]
