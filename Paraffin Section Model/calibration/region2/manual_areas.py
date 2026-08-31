"""Parse 韧皮部 / 髓 overlays from burned-in manuals.

Manuals use stacked dashed boxes whose inner edge follows the tissue,
not a single full-height rectangle. We fill the union of those boxes
row by row so training sees the real outline.
"""

from __future__ import annotations

import cv2
import numpy as np


def _overlay(orig: np.ndarray, manual: np.ndarray) -> np.ndarray:
    return cv2.absdiff(orig, manual).max(axis=2) > 35


def _clusters(xs: np.ndarray, gap: int = 36) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x - prev > gap:
            groups.append((start, prev))
            start = x
        prev = x
    groups.append((start, prev))
    return groups


def _fill_side_from_box_edges(edge: np.ndarray, from_left: bool) -> np.ndarray:
    """Fill stacked side boxes using the dashed outlines (varying width).

    Dashed boxes leave both an outer and an inner edge. Always fill from the
    image border out to the farthest (inner) edge pixel in the side band —
    otherwise only the outer stroke is kept and the region looks shrunk.
    """
    h, w = edge.shape
    e = cv2.dilate((edge > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8), 1)
    e[:, int(w * 0.32) : int(w * 0.68)] = 0
    out = np.zeros((h, w), np.uint8)
    limit = int(w * 0.32)
    edge_tol = max(12, int(0.04 * w))
    max_w = int(w * 0.30)
    for y in range(h):
        if from_left:
            xs = np.where(e[y, :limit] > 0)[0]
            if xs.size < 4 or int(xs.min()) > edge_tol:
                continue
            groups = _clusters(xs)
            g0 = groups[0]
            # Span from the border-touching cluster to the farthest inner edge.
            inner = g0[1] + 1
            for g in groups[1:]:
                if g[0] - g0[0] <= max_w:
                    inner = max(inner, g[1] + 1)
            # Fallback: rightmost edge pixel in the left band (dashed gaps).
            span = int(xs.max()) + 1
            if span - int(xs.min()) <= max_w:
                inner = max(inner, span)
            inner = min(inner, limit)
            if inner >= 16:
                out[y, :inner] = 255
        else:
            xs = np.where(e[y, w - limit :] > 0)[0]
            if xs.size < 4 or int(xs.max()) < (limit - edge_tol):
                continue
            groups = _clusters(xs)
            g_last = groups[-1]
            inner_rel = g_last[0]
            for g in reversed(groups[:-1]):
                if g_last[1] - g[1] <= max_w:
                    inner_rel = min(inner_rel, g[0])
            span_rel = int(xs.min())
            if int(xs.max()) - span_rel <= max_w:
                inner_rel = min(inner_rel, span_rel)
            x0 = w - limit + inner_rel
            if w - x0 >= 16:
                out[y, x0:] = 255
    # close small gaps between stacked boxes; then smooth jagged inner edge
    k = np.ones((21, 7), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    # Horizontal close so thin "outer-edge only" rows expand to neighbors.
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((1, 31), np.uint8))
    out[int(h * 0.90) :, int(w * 0.68) :] = 0
    # drop specks
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(out)
    min_area = h * w * 0.004
    legend = (0, 0, min(w, 420), min(h, 280))
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 80 and bh < 80 and x < legend[2] and y < legend[3]:
            continue
        cv2.drawContours(clean, [c], -1, 255, -1)
    return clean


def parse_overlay_rects(
    orig: np.ndarray,
    manual: np.ndarray,
    hue_lo: int,
    hue_hi: int,
    sat_lo: int,
    sat_hi: int,
    max_w_frac: float = 0.36,
) -> list[tuple[int, int, int, int]]:
    """Dashed cyan/purple side boxes as (x, y, w, h)."""
    h, w = orig.shape[:2]
    diff = _overlay(orig, manual)
    hsv = cv2.cvtColor(manual, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    ink = diff & (hue >= hue_lo) & (hue <= hue_hi) & (sat >= sat_lo) & (sat <= sat_hi)
    edge = cv2.dilate(ink.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), 1)
    # Close dashes inside a box, but do not glue vertically stacked boxes together.
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 13), np.uint8))
    edge[:260, :500] = 0
    edge[int(h * 0.90) :, int(w * 0.68) :] = 0
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 16 or bh < 20 or bw > w * max_w_frac:
            continue
        if bh < 28 and bw < 48:
            continue
        cx = x + bw * 0.5
        if 0.34 * w < cx < 0.66 * w:
            continue
        edge_tol = int(0.04 * w)
        if not (x <= edge_tol or x + bw >= w - edge_tol):
            continue
        boxes.append((x, y, bw, bh))
    return boxes


def _fill_rects(h: int, w: int, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    mask[int(h * 0.90) :, int(w * 0.68) :] = 0
    return mask


def parse_manual_area_boxes(
    orig: np.ndarray, manual: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Phloem/pith masks and the dashed boxes copied from the manual overlay."""
    h, w = orig.shape[:2]
    phloem_boxes = parse_overlay_rects(orig, manual, 100, 116, 80, 255)
    pith_boxes = parse_overlay_rects(orig, manual, 118, 155, 40, 255)
    ph = _fill_rects(h, w, phloem_boxes)
    pi = _fill_rects(h, w, pith_boxes)
    ph2, pi2 = parse_manual_areas(orig, manual)
    # A 1–2 box parse is usually a fragment. Prefer the fuller partitioned band.
    min_px = int(h * w * 0.008)
    if len(phloem_boxes) < 4 or cv2.countNonZero(ph) < min_px:
        if cv2.countNonZero(ph2) > cv2.countNonZero(ph):
            ph = ph2
    if len(pith_boxes) < 4 or cv2.countNonZero(pi) < min_px:
        if cv2.countNonZero(pi2) > cv2.countNonZero(pi):
            pi = pi2
    max_w = int(0.32 * w)
    if cv2.countNonZero(ph) > 0:
        xs = np.where(ph > 0)[1]
        if float(xs.mean()) < 0.5 * w:
            ph[:, max_w:] = 0
        else:
            ph[:, : w - max_w] = 0
    if cv2.countNonZero(pi) > 0:
        xs = np.where(pi > 0)[1]
        if float(xs.mean()) < 0.5 * w:
            pi[:, max_w:] = 0
        else:
            pi[:, : w - max_w] = 0
    return ph, pi, phloem_boxes, pith_boxes


def parse_manual_areas(orig: np.ndarray, manual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (phloem_mask, pith_mask) following stacked boxes / outlines."""
    h, w = orig.shape[:2]
    diff = _overlay(orig, manual)
    hsv = cv2.cvtColor(manual, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    phloem = diff & (hue >= 100) & (hue <= 116) & (sat >= 80)
    pith = diff & (hue >= 118) & (hue <= 155) & (sat >= 40)
    left_p = _fill_side_from_box_edges(phloem, True)
    right_p = _fill_side_from_box_edges(phloem, False)
    left_m = _fill_side_from_box_edges(pith, True)
    right_m = _fill_side_from_box_edges(pith, False)
    ph = cv2.bitwise_or(left_p, right_p)
    pi = cv2.bitwise_or(left_m, right_m)
    # Keep each mask on its dominant side only (avoid full-width spill).
    max_w = int(0.32 * w)
    if cv2.countNonZero(ph) > 0:
        xs = np.where(ph > 0)[1]
        if float(xs.mean()) < 0.5 * w:
            ph[:, max_w:] = 0
        else:
            ph[:, : w - max_w] = 0
    if cv2.countNonZero(pi) > 0:
        xs = np.where(pi > 0)[1]
        if float(xs.mean()) < 0.5 * w:
            pi[:, max_w:] = 0
        else:
            pi[:, : w - max_w] = 0
    return ph, pi
