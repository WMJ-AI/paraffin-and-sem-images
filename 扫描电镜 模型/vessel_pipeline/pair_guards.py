"""Pair geometry guards: reject double-vessel if small cavities sit in the shared wall."""
from __future__ import annotations

import cv2
import numpy as np

from segment import Vessel


def _closest_points(
    v1: Vessel, v2: Vessel
) -> tuple[np.ndarray, np.ndarray, float] | None:
    pts1 = v1.contour.reshape(-1, 2).astype(np.float32)
    pts2 = v2.contour.reshape(-1, 2).astype(np.float32)
    step1 = max(1, len(pts1) // 400)
    step2 = max(1, len(pts2) // 400)
    dmin = 1e18
    a_best = b_best = None
    sub2 = pts2[::step2]
    for p in pts1[::step1]:
        dd = np.sum((sub2 - p) ** 2, axis=1)
        k = int(np.argmin(dd))
        d = float(dd[k])
        if d < dmin:
            dmin = d
            a_best = p
            b_best = sub2[k]
    if a_best is None or dmin < 1.0:
        return None
    return a_best, b_best, float(np.sqrt(dmin))


def _shared_wall_band(
    gray: np.ndarray,
    v1: Vessel,
    v2: Vessel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    Shared-wall mask = thick corridor between closest contour points,
    excluding vessel interiors.

    Prefer corridor over full dilation∩ (large gaps like 4(8)≈63px need coverage;
    unbounded dilation FP on true tight pairs).
    """
    h, w = gray.shape[:2]
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.drawContours(m1, [v1.contour], -1, 255, -1)
    cv2.drawContours(m2, [v2.contour], -1, 255, -1)

    closest = _closest_points(v1, v2)
    if closest is None:
        return None
    a, b, gap_px = closest
    if gap_px < 1.0:
        return None

    # Half-width of wall corridor (px): enough to catch mid-wall pits, not whole tissue
    half_w = int(np.clip(max(gap_px * 0.28, 12.0), 12, 20))
    strip = np.zeros((h, w), np.uint8)
    cv2.line(
        strip,
        (int(round(float(a[0]))), int(round(float(a[1])))),
        (int(round(float(b[0]))), int(round(float(b[1])))),
        255,
        max(2 * half_w, 1),
    )
    band = cv2.bitwise_and(strip, cv2.bitwise_not(cv2.bitwise_or(m1, m2)))
    if int(cv2.countNonZero(band)) < 15:
        return None
    return m1, m2, band, gap_px


def _has_compact_dark_pore(
    gray: np.ndarray,
    band: np.ndarray,
    thr: float,
    um_per_px: float,
    *,
    min_cavity_ai_um2: float,
    max_cavity_ai_um2: float,
    min_circ: float = 0.18,
    min_solid: float = 0.28,
) -> bool:
    h, w = gray.shape[:2]
    ys, xs = np.where(band > 0)
    if ys.size == 0:
        return False
    y0, y1 = max(0, int(ys.min())), min(h, int(ys.max()) + 1)
    x0, x1 = max(0, int(xs.min())), min(w, int(xs.max()) + 1)
    roi = gray[y0:y1, x0:x1]
    band_roi = band[y0:y1, x0:x1]
    dark = ((roi.astype(np.float64) <= thr) & (band_roi > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        1,
    )
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_cavity_ai_um2 / (um_per_px**2)
    max_area = max_cavity_ai_um2 / (um_per_px**2)
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        peri = float(cv2.arcLength(c, True))
        if peri < 1e-6:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        x, y, bw, bh = cv2.boundingRect(c)
        solid = area / max(float(bw * bh), 1.0)
        if circ < min_circ or solid < min_solid:
            continue
        return True
    return False


def pair_has_interstitial_lumens(
    gray: np.ndarray,
    v1: Vessel,
    v2: Vessel,
    um_per_px: float,
    *,
    min_cavity_ai_um2: float = 3.0,
    max_cavity_ai_um2: float = 120.0,
) -> bool:
    """
    True if discrete dark pores sit in the shared wall between two vessels.

    Teaching (9(9), 11(5), 4(8)): wall between two lumens contains small cavities
    → not a true double vessel; keep both as singles only.
    """
    if gray is None or um_per_px <= 0:
        return False
    packed = _shared_wall_band(gray, v1, v2)
    if packed is None:
        return False
    m1, m2, band, gap_px = packed
    gap_um = gap_px * um_per_px

    vessel_vals = np.concatenate(
        [gray[m1 > 0].astype(np.float64), gray[m2 > 0].astype(np.float64)]
    )
    band_vals = gray[band > 0].astype(np.float64)
    if vessel_vals.size < 20 or band_vals.size < 15:
        return False
    vessel_p50 = float(np.percentile(vessel_vals, 50))
    band_p50 = float(np.percentile(band_vals, 50))

    # Path A: lumen-dark compact pores in the wall corridor
    thr_lumen = vessel_p50 + 5.0
    dark_frac = float((band_vals <= thr_lumen).mean())
    if dark_frac >= 0.06 and _has_compact_dark_pore(
        gray,
        band,
        thr_lumen,
        um_per_px,
        min_cavity_ai_um2=min_cavity_ai_um2,
        max_cavity_ai_um2=max_cavity_ai_um2,
        min_circ=0.18,
        min_solid=0.28,
    ):
        return True

    # Path B: mid-gray compact pits (not fully lumen-black)
    span = band_p50 - vessel_p50
    if span >= 12.0:
        thr_wall = vessel_p50 + 0.55 * span
        if _has_compact_dark_pore(
            gray,
            band,
            thr_wall,
            um_per_px,
            min_cavity_ai_um2=1.0,
            max_cavity_ai_um2=min(40.0, max_cavity_ai_um2),
            min_circ=0.45,
            min_solid=0.45,
        ):
            return True

    # Path C: wall corridor has multiple intensity valleys / mid-dark pits
    # True shared wall is bright & smooth (1(1)/8(1)); false has valleys (18(8))
    if gap_um >= 4.0:
        closest = _closest_points(v1, v2)
        if closest is not None:
            a, b, _ = closest
            n = max(24, int(gap_px))
            xs = np.linspace(float(a[0]), float(b[0]), n)
            ys = np.linspace(float(a[1]), float(b[1]), n)
            h, w = gray.shape[:2]
            profile = []
            for x, y in zip(xs, ys):
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h and band[yi, xi]:
                    profile.append(float(gray[yi, xi]))
            if len(profile) >= 12:
                arr = np.asarray(profile, dtype=np.float64)
                valleys = 0
                for i in range(2, len(arr) - 2):
                    if (
                        arr[i] <= arr[i - 1]
                        and arr[i] <= arr[i + 1]
                        and arr[i] < arr[i - 2]
                        and arr[i] < arr[i + 2]
                        and arr[i] < 70.0
                    ):
                        valleys += 1
                med = float(np.median(arr))
                # ≥2 valleys ⇒ wall not a clean shared face (fiber / small cavities)
                if valleys >= 2 and (
                    gap_um >= 8.0 or dark_frac >= 0.10 or med <= 65.0
                ):
                    return True
                # also mid-threshold compact pores when wall not fully lumen-black
                if dark_frac >= 0.12 and _has_compact_dark_pore(
                    gray,
                    band,
                    vessel_p50 + 15.0,
                    um_per_px,
                    min_cavity_ai_um2=2.0,
                    max_cavity_ai_um2=min(80.0, max_cavity_ai_um2),
                    min_circ=0.20,
                    min_solid=0.30,
                ):
                    return True
    return False


def pair_has_intervening_vessel(
    v1: Vessel,
    v2: Vessel,
    others: list[Vessel],
    gray_shape: tuple[int, int] | None = None,
    *,
    max_dist_frac: float = 0.40,
    max_dist_px: float = 40.0,
) -> bool:
    """True if another lumen sits between the two candidates (breaks double vessel)."""
    ax, ay = float(v1.cx), float(v1.cy)
    bx, by = float(v2.cx), float(v2.cy)
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-6:
        return False
    ab_len = float(np.sqrt(ab2))
    thr = max(max_dist_px, max_dist_frac * ab_len)
    for v in others:
        if v.vessel_id in (v1.vessel_id, v2.vessel_id):
            continue
        t = ((v.cx - ax) * abx + (v.cy - ay) * aby) / ab2
        if t < 0.08 or t > 0.92:
            continue
        px, py = ax + t * abx, ay + t * aby
        dist = float(np.hypot(v.cx - px, v.cy - py))
        if dist <= thr:
            return True
    # Also: any other lumen overlapping the shared-wall corridor
    if gray_shape is not None and others:
        closest = _closest_points(v1, v2)
        if closest is None:
            return False
        a, b, gap_px = closest
        h, w = gray_shape[:2]
        half_w = int(np.clip(max(gap_px * 0.28, 12.0), 12, 20))
        strip = np.zeros((h, w), np.uint8)
        cv2.line(
            strip,
            (int(round(float(a[0]))), int(round(float(a[1])))),
            (int(round(float(b[0]))), int(round(float(b[1])))),
            255,
            max(2 * half_w, 1),
        )
        m1 = np.zeros((h, w), np.uint8)
        m2 = np.zeros((h, w), np.uint8)
        cv2.drawContours(m1, [v1.contour], -1, 255, -1)
        cv2.drawContours(m2, [v2.contour], -1, 255, -1)
        band = cv2.bitwise_and(strip, cv2.bitwise_not(cv2.bitwise_or(m1, m2)))
        for v in others:
            if v.vessel_id in (v1.vessel_id, v2.vessel_id):
                continue
            mv = np.zeros((h, w), np.uint8)
            cv2.drawContours(mv, [v.contour], -1, 255, -1)
            ov = int(cv2.countNonZero(cv2.bitwise_and(mv, band)))
            area = int(cv2.countNonZero(mv))
            if area > 0 and ov / area >= 0.20:
                return True
    return False


def filter_pairs_without_interstitial(
    pairs: list,
    gray: np.ndarray | None,
    um_per_px: float,
    *,
    min_cavity_ai_um2: float = 3.0,
    max_cavity_ai_um2: float = 120.0,
    all_vessels: list[Vessel] | None = None,
) -> list:
    if not pairs:
        return pairs
    kept = []
    shape = gray.shape[:2] if gray is not None else None
    for p in pairs:
        reject = False
        if all_vessels and pair_has_intervening_vessel(
            p.v1, p.v2, all_vessels, gray_shape=shape
        ):
            reject = True
        elif gray is not None and pair_has_interstitial_lumens(
            gray,
            p.v1,
            p.v2,
            um_per_px,
            min_cavity_ai_um2=min_cavity_ai_um2,
            max_cavity_ai_um2=max_cavity_ai_um2,
        ):
            reject = True
        if reject:
            p.v1.pair_id = None
            p.v2.pair_id = None
            continue
        kept.append(p)
    return kept
