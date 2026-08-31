"""Parse yellow 导管 example boxes from manuals and their lumen areas."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.io_util import imread
from calibration.region2.rules import DETECT_MIN_AREA_UM2, DETECT_MIN_DIAM_UM, MIN_AREA_UM2
from calibration.region2.scale_bar import get_region2_scale
from calibration.region2.vessels import (
    Vessel,
    _bright_mask,
    _center,
    _circularity,
    _collect_contours,
    _nms,
    _split_blob,
)


def parse_yellow_boxes(orig: np.ndarray, manual: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = orig.shape[:2]
    diff = cv2.absdiff(orig, manual).max(axis=2) > 35
    hsv = cv2.cvtColor(manual, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    yellow = diff & (hue >= 18) & (hue <= 42) & (sat >= 70)
    edge = cv2.dilate(yellow.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), 1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    edge[:280, :420] = 0
    edge[int(h * 0.90) :, int(w * 0.68) :] = 0
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 12 or bh < 12 or bw > w * 0.12 or bh > h * 0.12:
            continue
        if bw * bh < 80:
            continue
        boxes.append((x, y, bw, bh))
    return boxes


def lumen_area_um2(
    orig: np.ndarray,
    box: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    um_per_pixel: float | None = None,
) -> float | None:
    if um_per_pixel is None:
        um_per_pixel = get_region2_scale(orig).um_per_pixel
    if mask is None:
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        mask = _bright_mask(gray)
    x, y, bw, bh = box
    roi = mask[y : y + bh, x : x + bw]
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    area_px = float(cv2.contourArea(max(cnts, key=cv2.contourArea)))
    if area_px < 8:
        return None
    return area_px * um_per_pixel * um_per_pixel


def summarize_labeled_min_areas(base_dir: Path) -> dict:
    """Per-image min labeled lumen, then the mean of those mins."""
    orig_dir = base_dir / "原图" / "第二区域"
    man_dir = base_dir / "人工标定" / "第二区域"
    rows: list[dict] = []
    mins: list[float] = []
    for mp in sorted(man_dir.glob("*.jpg")):
        orig = imread(orig_dir / mp.name)
        manual = imread(mp)
        if orig is None or manual is None:
            continue
        umpp = get_region2_scale(orig).um_per_pixel
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        mask = _bright_mask(gray)
        areas = []
        for box in parse_yellow_boxes(orig, manual):
            a = lumen_area_um2(orig, box, mask=mask, um_per_pixel=umpp)
            if a is not None:
                areas.append(a)
        if not areas:
            continue
        med = float(np.median(areas))
        robust = [a for a in areas if a >= max(40.0, 0.20 * med)]
        mn_raw = float(min(areas))
        mn = float(min(robust)) if robust else mn_raw
        mins.append(mn)
        rows.append(
            {
                "filename": mp.name,
                "n_labeled": len(areas),
                "min_um2": round(mn_raw, 2),
                "min_robust_um2": round(mn, 2),
                "median_um2": round(med, 2),
                "max_um2": round(float(np.max(areas)), 2),
            }
        )
    mean_min = float(np.mean(mins)) if mins else 59.3
    return {"mean_min_um2": mean_min, "n_images": len(mins), "rows": rows, "mins": mins}


def apply_learned_min_area(base_dir: Path, min_manual_images: int = 5) -> float:
    """Write labeled-min CSV for diagnostics. Keep the rule floor in rules.py."""
    from calibration.region2 import rules

    summary = summarize_labeled_min_areas(base_dir)
    rows = summary["rows"]
    out_csv = base_dir / "推理结果" / "region2_labeled_min_area.csv"
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "n_labeled",
                    "min_um2",
                    "min_robust_um2",
                    "median_um2",
                    "max_um2",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
    return float(rules.MIN_AREA_UM2)


def _vessel_loose(cnt: np.ndarray, umpp: float) -> Vessel | None:
    """Keep every lumen inside a yellow box (no min-area / ring filters)."""
    area = float(cv2.contourArea(cnt))
    if area < 8:
        return None
    eq = float(2.0 * math.sqrt(area / math.pi))
    cen = _center(cnt)
    if cen is None:
        return None
    return Vessel(
        contour=cnt,
        center=cen,
        area_px=area,
        eq_diam_px=eq,
        area_um2=area * umpp * umpp,
        eq_diam_um=eq * umpp,
    )


def vessels_from_manual(orig: np.ndarray, manual: np.ndarray, scale=None) -> list[Vessel]:
    """All vessel lumens inside the yellow 导管 boxes (clusters kept as separate lumens)."""
    if scale is None:
        scale = get_region2_scale(orig)
    umpp = scale.um_per_pixel
    min_area_px = max(80.0, 0.55 * MIN_AREA_UM2) / (umpp * umpp)
    min_r = max(3.0, 0.45 * DETECT_MIN_DIAM_UM / umpp)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    mask = _bright_mask(gray)
    h, w = gray.shape
    found: list[Vessel] = []
    for x, y, bw, bh in parse_yellow_boxes(orig, manual):
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        roi = mask[y0:y1, x0:x1]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sized: list[tuple[float, np.ndarray]] = []
        for c in cnts:
            a = float(cv2.contourArea(c))
            if a >= min_area_px * 0.45:
                sized.append((a, c))
        if not sized:
            for c in _collect_contours(roi, min_area_px, min_r):
                a = float(cv2.contourArea(c))
                if a >= min_area_px * 0.45:
                    sized.append((a, c))
        if not sized:
            continue
        sized.sort(key=lambda t: t[0], reverse=True)
        top = sized[0][0]
        kept: list[np.ndarray] = []
        for a, c in sized:
            if len(kept) >= 6:
                break
            if a < max(min_area_px * 0.45, 0.18 * top):
                continue
            circ = _circularity(c, a)
            if circ < 0.62 and a > min_area_px:
                blob = np.zeros_like(roi)
                cv2.drawContours(blob, [c], -1, 255, -1)
                children = _split_blob(blob, min_r)
                kids = [ch for ch in children if cv2.contourArea(ch) >= min_area_px * 0.4]
                if 2 <= len(kids) <= 6:
                    kept.extend(kids)
                    continue
            kept.append(c)
        for c in kept:
            c = c + np.array([[[x0, y0]]])
            v = _vessel_loose(c, umpp)
            if v is not None:
                found.append(v)
    return _nms(found)
