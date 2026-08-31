"""Detect vessel lumens: round bright cavities in 木质部, sized like 人工黄框样例."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

cv2.setNumThreads(4)

from calibration.region2.rules import (
    DETECT_MIN_AREA_UM2,
    DETECT_MIN_DIAM_UM,
    MAX_ASPECT,
    MAX_DIAM_UM,
    MIN_AREA_UM2,
    MIN_CIRCULARITY,
    MIN_DIAM_UM,
    RING_DELTA,
    SMALL_MIN_CIRCULARITY,
)
from calibration.region2.scale_bar import get_region2_scale
from calibration.scale import ScaleInfo


@dataclass
class Vessel:
    contour: np.ndarray
    center: tuple[float, float]
    area_px: float
    eq_diam_px: float
    area_um2: float
    eq_diam_um: float


@dataclass
class Region2Result:
    vessels: list[Vessel]
    scale: ScaleInfo
    xylem_area_px: float
    extra_area_px: float
    tissue_mask: np.ndarray
    extra_mask: np.ndarray
    phloem_mask: np.ndarray
    pith_mask: np.ndarray
    xylem_mask: np.ndarray
    phloem_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    pith_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    xylem_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.vessels)

    @property
    def lumen_area_um2(self) -> float:
        return float(sum(v.area_um2 for v in self.vessels))

    @property
    def xylem_area_um2(self) -> float:
        um = self.scale.um_per_pixel
        return float(self.xylem_area_px) * um * um

    @property
    def phloem_area_um2(self) -> float:
        um = self.scale.um_per_pixel
        return float(cv2.countNonZero(self.phloem_mask)) * um * um

    @property
    def pith_area_um2(self) -> float:
        um = self.scale.um_per_pixel
        return float(cv2.countNonZero(self.pith_mask)) * um * um


def _circularity(cnt: np.ndarray, area: float) -> float:
    peri = cv2.arcLength(cnt, True)
    return float(4.0 * np.pi * area / (peri * peri + 1e-6))


def _aspect(cnt: np.ndarray) -> float:
    x, y, bw, bh = cv2.boundingRect(cnt)
    return float(max(bw, bh) / max(min(bw, bh), 1))


def _center(cnt: np.ndarray) -> tuple[float, float] | None:
    m = cv2.moments(cnt)
    if m["m00"] <= 0:
        return None
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def _is_crack_blob(
    long_side: float,
    short_side: float,
    area: float,
    img_h: int,
    max_vessel_px: float,
    circ: float | None = None,
) -> bool:
    """Sectioning tear / jagged gap, not a 2–4 vessel multiple."""
    if long_side < 80:
        return False
    asp = long_side / max(short_side, 1.0)
    mean_thick = float(area) / max(float(long_side), 1.0)
    thin = max(32.0, float(max_vessel_px) * 0.12)
    # Through-going white tear: bbox can be wide from zigzags, still not a vessel.
    if long_side >= 0.28 * img_h and asp >= 1.6 and mean_thick <= 0.18 * img_h:
        return True
    if long_side >= 0.20 * img_h and mean_thick <= thin * 2.5 and asp >= 1.7:
        return True
    if long_side >= 150 and mean_thick <= thin * 1.2 and asp >= 3.0:
        return True
    if asp >= 4.0 and mean_thick <= thin and short_side <= thin * 1.8:
        return True
    if circ is not None and circ < 0.22 and asp >= 2.6 and long_side >= 160:
        return True
    return False


def _crack_streak_mask(mask: np.ndarray, max_vessel_px: float = 380.0) -> np.ndarray:
    """Long tears / sectioning gaps — not round vessel lumens."""
    crack = np.zeros_like(mask)
    h, _w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        _x, _y, bw, bh, area = stats[i]
        long_side = max(int(bw), int(bh))
        short_side = min(int(bw), int(bh))
        if _is_crack_blob(long_side, short_side, float(area), h, max_vessel_px):
            crack[labels == i] = 255
    if cv2.countNonZero(crack) == 0:
        return crack
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.dilate(crack, k, iterations=2)


def _long_void_mask(gray: np.ndarray, max_vessel_px: float = 380.0) -> np.ndarray:
    """White gaps that belong to a long tear, including leftover round bites."""
    bright = (gray > 198).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    return _crack_streak_mask(bright, max_vessel_px)


def _gray_tear_mask(gray: np.ndarray, max_vessel_px: float = 380.0) -> np.ndarray:
    """Bright vertical/horizontal tears on grayscale (before lumen thresholding)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 51))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 5))
    streak = np.maximum(
        cv2.morphologyEx(blur, cv2.MORPH_OPEN, kv),
        cv2.morphologyEx(blur, cv2.MORPH_OPEN, kh),
    )
    local = cv2.blur(blur, (71, 71))
    m = (
        (streak > 145) & (streak.astype(np.int16) - local.astype(np.int16) > 15)
    ).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _crack_streak_mask(m, max_vessel_px)


def _bright_mask(gray: np.ndarray, max_vessel_px: float = 380.0) -> np.ndarray:
    """Bright lumens at several scales so large overexposed vessels are kept."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    local31 = cv2.blur(blur, (31, 31))
    local71 = cv2.blur(blur, (71, 71))
    local151 = cv2.blur(blur, (151, 151))
    m31 = ((blur.astype(np.int16) - local31.astype(np.int16) > 10) & (blur > 125)).astype(np.uint8) * 255
    m71 = ((blur.astype(np.int16) - local71.astype(np.int16) > 7) & (blur > 112)).astype(np.uint8) * 255
    m151 = ((blur.astype(np.int16) - local151.astype(np.int16) > 4) & (blur > 98)).astype(np.uint8) * 255
    # Overexposed large cavities: brighter than a wide neighborhood even if the
    # interior is flat white.
    m_abs = ((blur > 170) & (blur.astype(np.int16) - local151.astype(np.int16) > 2)).astype(np.uint8) * 255
    mask = m31
    for extra in (m71, m151, m_abs):
        mask = cv2.bitwise_or(mask, extra)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    h, w = mask.shape
    mask[int(h * 0.90) :, int(w * 0.68) :] = 0
    flood = mask.copy()
    ffm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffm, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    cnts, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    max_hole = int(np.pi * (0.55 * max_vessel_px) ** 2)
    for cnt in cnts:
        if cv2.contourArea(cnt) <= max_hole:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
    mask = cv2.bitwise_or(mask, filled)
    cracks = cv2.bitwise_or(
        _crack_streak_mask(mask, max_vessel_px),
        _gray_tear_mask(gray, max_vessel_px),
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(cracks))
    return mask


def _ring_ok(gray: np.ndarray, cnt: np.ndarray, delta: float = RING_DELTA) -> bool:
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad = 16
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + bw + pad), min(gray.shape[0], y + bh + pad)
    roi = gray[y0:y1, x0:x1]
    local = cnt - np.array([[[x0, y0]]])
    lumen = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(lumen, [local], -1, 255, thickness=-1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ring = cv2.dilate(lumen, k, iterations=2)
    ring = cv2.subtract(ring, lumen)
    if int(lumen.sum()) < 255 * 20 or int(ring.sum()) < 255 * 20:
        return False
    lm = float(roi[lumen > 0].mean())
    rm = float(roi[ring > 0].mean())
    if lm >= rm + delta:
        return True
    # Large overexposed lumens: wall is also pale; still a vessel if the cavity is bright.
    area = float(cv2.countNonZero(lumen))
    return lm >= 165.0 and area >= 2500 and lm >= rm - 2.0


def _wall_closed(gray: np.ndarray, cnt: np.ndarray, lumen_mean: float | None = None) -> bool:
    """Reject crack bites: a closed vessel has a dark wall all around, not a bright gap."""
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad = 18
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + bw + pad), min(gray.shape[0], y + bh + pad)
    roi = gray[y0:y1, x0:x1]
    local = cnt - np.array([[[x0, y0]]])
    lumen = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(lumen, [local], -1, 255, thickness=-1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ring = cv2.subtract(cv2.dilate(lumen, k, iterations=2), lumen)
    ys, xs = np.where(ring > 0)
    if xs.size < 24:
        return False
    if lumen_mean is None:
        if int(lumen.sum()) < 255 * 20:
            return False
        lumen_mean = float(roi[lumen > 0].mean())
    m = cv2.moments(local)
    if m["m00"] <= 0:
        return False
    cx = float(m["m10"] / m["m00"])
    cy = float(m["m01"] / m["m00"])
    bins = 16
    ang = np.arctan2(ys.astype(np.float32) - cy, xs.astype(np.float32) - cx)
    idx = np.floor((ang + np.pi) * (bins / (2.0 * np.pi))).astype(np.int32) % bins
    open_bin = np.zeros(bins, dtype=bool)
    ring_vals = roi[ring > 0]
    wall_med = float(np.median(ring_vals))
    for b in range(bins):
        sel = idx == b
        if not np.any(sel):
            open_bin[b] = False
            continue
        mean = float(roi[ys[sel], xs[sel]].mean())
        # White tear sector: as bright as the lumen and clearly brighter than the wall.
        open_bin[b] = mean >= lumen_mean - 8.0 and mean >= wall_med + 18.0
    doubled = np.concatenate([open_bin, open_bin])
    run = 0
    max_run = 0
    for flag in doubled:
        if flag:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    # >= ~90° of ring as bright as the lumen → open to a tear.
    return max_run < 5


def _peaks_to_contours(mask: np.ndarray, min_r: float) -> list[np.ndarray]:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() < min_r * 0.45:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(cnts)
    ksz = max(9, int(round(min_r * 0.9)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    peaks = ((dist == cv2.dilate(dist, kernel)) & (dist >= min_r * 0.42)).astype(np.uint8)
    n_lab, _labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)
    out: list[np.ndarray] = []
    h, w = mask.shape
    for lab in range(1, n_lab):
        px, py = int(round(centroids[lab][0])), int(round(centroids[lab][1]))
        if not (0 <= px < w and 0 <= py < h):
            continue
        r = max(3, int(round(float(dist[py, px]) * 0.92)))
        x0, y0 = max(0, px - r), max(0, py - r)
        x1, y1 = min(w, px + r + 1), min(h, py + r + 1)
        roi = mask[y0:y1, x0:x1]
        disk = np.zeros_like(roi)
        cv2.circle(disk, (px - x0, py - y0), r, 255, thickness=-1)
        disk = cv2.bitwise_and(disk, roi)
        cnts, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea) + np.array([[[x0, y0]]])
            out.append(cnt)
    return out


def _split_blob(mask: np.ndarray, min_r: float) -> list[np.ndarray]:
    """Split touching vessel multiples. Peak disks on large ROIs; watershed on small."""
    h, w = mask.shape
    if h * w > 160_000:
        return _peaks_to_contours(mask, min_r)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() < min_r * 0.6:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(cnts)
    ksz = max(9, int(round(min_r * 0.9)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    local_max = (dist == cv2.dilate(dist, kernel)) & (dist >= min_r * 0.45)
    local_max = local_max.astype(np.uint8) * 255
    n_seed, seeds = cv2.connectedComponents(local_max)
    if n_seed <= 2:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(cnts)
    unknown = cv2.subtract(mask, local_max)
    markers = seeds + 1
    markers[unknown > 0] = 0
    markers[mask == 0] = 1
    color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    out: list[np.ndarray] = []
    for lab in range(2, int(markers.max()) + 1):
        part = (markers == lab).astype(np.uint8) * 255
        part = cv2.bitwise_and(part, mask)
        cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(cnts)
    return out


def _collect_contours(
    mask: np.ndarray,
    min_area: float,
    min_r: float,
    max_vessel_px: float = 380.0,
) -> list[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[np.ndarray] = []
    h, _w = mask.shape
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < min_area * 0.55:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        circ = _circularity(cnt, area)
        if _is_crack_blob(max(bw, bh), min(bw, bh), area, h, max_vessel_px, circ):
            continue
        if circ < 0.72 and area > min_area * 0.7:
            pad = 4
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(mask.shape[1], x + bw + pad)
            y1 = min(mask.shape[0], y + bh + pad)
            roi = mask[y0:y1, x0:x1]
            for child in _split_blob(roi, min_r):
                child = child + np.array([[[x0, y0]]])
                out.append(child)
        else:
            out.append(cnt)
    return out


def _nms(vessels: list[Vessel], iou_or_dist_frac: float = 0.45) -> list[Vessel]:
    vessels = sorted(vessels, key=lambda v: v.area_px, reverse=True)
    kept: list[Vessel] = []
    for v in vessels:
        ok = True
        for k in kept:
            d = float(np.hypot(v.center[0] - k.center[0], v.center[1] - k.center[1]))
            if d < iou_or_dist_frac * 0.42 * (v.eq_diam_px + k.eq_diam_px):
                ok = False
                break
        if ok:
            kept.append(v)
    return kept


def _drop_overlap_chains(vessels: list[Vessel], min_n: int = 4) -> list[Vessel]:
    """Drop detections lined up along a tear. Compact 2–4 vessel multiples stay."""
    n = len(vessels)
    if n < min_n:
        return list(vessels)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        a = vessels[i]
        for j in range(i + 1, n):
            b = vessels[j]
            d = float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]))
            r = 0.5 * (a.eq_diam_px + b.eq_diam_px)
            # Tile along a tear (centers ~1 diameter apart) or near-duplicates.
            if r > 1 and d < 1.12 * r:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    drop: set[int] = set()
    for idxs in groups.values():
        n_g = len(idxs)
        if n_g < min_n:
            continue
        xs = [vessels[i].center[0] for i in idxs]
        ys = [vessels[i].center[1] for i in idxs]
        bw = max(xs) - min(xs) + 1.0
        bh = max(ys) - min(ys) + 1.0
        asp = max(bw, bh) / max(min(bw, bh), 1.0)
        if n_g >= 6 and asp >= 1.8:
            drop.update(idxs)
        elif n_g >= 5 and asp >= 2.2:
            drop.update(idxs)
        elif n_g >= 4 and asp >= 3.2:
            drop.update(idxs)
    if not drop:
        return list(vessels)
    return [v for i, v in enumerate(vessels) if i not in drop]


def _vessel_from_contour(
    gray: np.ndarray,
    cnt: np.ndarray,
    umpp: float,
    min_d: float,
    max_d: float,
    min_area: float,
    max_area: float,
) -> Vessel | None:
    area = float(cv2.contourArea(cnt))
    if not (min_area <= area <= max_area):
        return None
    eq = float(2.0 * np.sqrt(area / np.pi))
    if not (min_d <= eq <= max_d):
        return None
    circ = _circularity(cnt, area)
    asp = _aspect(cnt)
    area_um2 = area * umpp * umpp
    min_circ = SMALL_MIN_CIRCULARITY if area_um2 < 250.0 else MIN_CIRCULARITY
    if circ < min_circ or asp > MAX_ASPECT:
        return None
    if asp > 1.85 and circ < 0.55:
        return None
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area > 0 and area / hull_area < 0.82:
        return None
    if not _ring_ok(gray, cnt):
        if circ < 0.62 or area_um2 < 350.0:
            return None
    if not _wall_closed(gray, cnt):
        return None
    cen = _center(cnt)
    if cen is None:
        return None
    return Vessel(
        contour=cnt,
        center=cen,
        area_px=area,
        eq_diam_px=eq,
        area_um2=area_um2,
        eq_diam_um=eq * umpp,
    )


def vessels_from_boxes(
    img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    scale: ScaleInfo | None = None,
) -> list[Vessel]:
    """Segment lumens inside boxes (kept for diagnostics)."""
    if scale is None:
        scale = get_region2_scale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    umpp = scale.um_per_pixel
    min_area = DETECT_MIN_AREA_UM2 / (umpp * umpp)
    max_area = np.pi * ((MAX_DIAM_UM / umpp) * 0.5) ** 2
    min_d = max(4.0, DETECT_MIN_DIAM_UM / umpp)
    max_d = MAX_DIAM_UM / umpp
    mask = _bright_mask(gray, max_d)
    h, w = gray.shape
    found: list[Vessel] = []
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        roi = mask[y0:y1, x0:x1]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cnt = cnt + np.array([[[x0, y0]]])
        v = _vessel_from_contour(gray, cnt, umpp, min_d, max_d, min_area, max_area)
        if v is not None:
            found.append(v)
    return found


def detect_vessels(img: np.ndarray, scale: ScaleInfo | None = None) -> list[Vessel]:
    """Find every 导管腔 that matches the size/shape of 人工黄框样例."""
    if scale is None:
        scale = get_region2_scale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    umpp = scale.um_per_pixel
    min_area = DETECT_MIN_AREA_UM2 / (umpp * umpp)
    max_area = np.pi * ((MAX_DIAM_UM / umpp) * 0.5) ** 2
    min_d = max(4.0, DETECT_MIN_DIAM_UM / umpp)
    max_d = MAX_DIAM_UM / umpp
    mask = _bright_mask(gray, max_d)
    min_r = 0.5 * min_d
    vessels: list[Vessel] = []
    for cnt in _collect_contours(mask, min_area, min_r, max_d):
        v = _vessel_from_contour(gray, cnt, umpp, min_d, max_d, min_area, max_area)
        if v is not None:
            vessels.append(v)
    kept = _nms(vessels)
    kept = filter_vessels_by_min_area(kept)
    kept = _drop_overlap_chains(kept)
    voids = cv2.bitwise_or(_gray_tear_mask(gray, max_d), _long_void_mask(gray, max_d))
    if cv2.countNonZero(voids) > 0:
        kept = [v for v in kept if not _vessel_hits_mask(v, voids, frac=0.18)]
    return kept


def _tissue_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tissue = np.ones((h, w), np.uint8) * 255
    tissue[int(h * 0.90) :, int(w * 0.68) :] = 0
    tissue[gray > 245] = 0
    return tissue


def _drop_mask(phloem: np.ndarray, pith: np.ndarray) -> np.ndarray:
    drop = np.zeros_like(phloem)
    k_pith = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_ph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    if cv2.countNonZero(pith) > 0:
        drop = cv2.bitwise_or(drop, cv2.dilate(pith, k_pith, iterations=2))
    if cv2.countNonZero(phloem) > 0:
        drop = cv2.bitwise_or(drop, cv2.dilate(phloem, k_ph, iterations=2))
    return drop


def _vessel_hits_mask(v: Vessel, mask: np.ndarray, frac: float = 0.12) -> bool:
    h, w = mask.shape[:2]
    cx, cy = int(round(v.center[0])), int(round(v.center[1]))
    if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
        return True
    x, y, bw, bh = cv2.boundingRect(v.contour)
    if bw < 1 or bh < 1:
        return False
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    roi = mask[y0:y1, x0:x1]
    if cv2.countNonZero(roi) == 0:
        return False
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cnt = v.contour - np.array([[[x0, y0]]])
    cv2.drawContours(local, [cnt], -1, 255, -1)
    area = float(cv2.countNonZero(local))
    if area < 1:
        return False
    return float(cv2.countNonZero(cv2.bitwise_and(local, roi))) / area >= frac


def filter_vessels_to_xylem(vessels: list[Vessel], phloem: np.ndarray, pith: np.ndarray) -> list[Vessel]:
    drop = _drop_mask(phloem, pith)
    if cv2.countNonZero(drop) == 0:
        return list(vessels)
    return [v for v in vessels if not _vessel_hits_mask(v, drop)]


def filter_vessels_by_min_area(
    vessels: list[Vessel], min_um2: float | None = None
) -> list[Vessel]:
    if min_um2 is None:
        min_um2 = float(MIN_AREA_UM2)
    return [v for v in vessels if v.area_um2 >= min_um2]


def analyze_region2(
    img: np.ndarray,
    area_clf=None,
    manual: np.ndarray | None = None,
    image_name: str | None = None,
) -> Region2Result:
    from calibration.region2.areas import stack_band_boxes, detect_phloem_pith, refine_side_mask

    scale = get_region2_scale(img)
    auto_vessels = detect_vessels(img, scale)
    tissue = _tissue_mask(img)
    h, w = img.shape[:2]
    phloem = np.zeros((h, w), np.uint8)
    pith = np.zeros((h, w), np.uint8)
    phloem_boxes: list[tuple[int, int, int, int]] = []
    pith_boxes: list[tuple[int, int, int, int]] = []

    # Stacked dashed boxes: partition first with the inner tissue line, then
    # cover the region with several rectangles that stay inside that line.
    if manual is not None:
        from calibration.region2.manual_areas import parse_manual_areas

        phloem, pith = parse_manual_areas(img, manual)
        if cv2.countNonZero(phloem):
            phloem = refine_side_mask(phloem)
        if cv2.countNonZero(pith):
            pith = refine_side_mask(pith)
    elif area_clf is not None:
        try:
            phloem, pith = detect_phloem_pith(img, area_clf, vessels=auto_vessels)
        except Exception:
            phloem = np.zeros((h, w), np.uint8)
            pith = np.zeros((h, w), np.uint8)

    phloem[int(h * 0.90) :, int(w * 0.68) :] = 0
    pith[int(h * 0.90) :, int(w * 0.68) :] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    phloem[gray > 245] = 0
    pith[gray > 245] = 0
    extra = cv2.bitwise_or(phloem, pith)
    xylem_tissue = cv2.subtract(tissue, extra)

    kept = filter_vessels_to_xylem(auto_vessels, phloem, pith)
    kept = _drop_overlap_chains(kept)
    kept = filter_vessels_by_min_area(kept)

    # Stacked dashed boxes stay inside the inner tissue line.
    phloem_boxes = stack_band_boxes(phloem)
    pith_boxes = stack_band_boxes(pith)
    # 局部木质部面积 = 整幅视野 (orange frame in manuals).
    xylem_boxes: list[tuple[int, int, int, int]] = [(0, 0, w, h)]
    return Region2Result(
        vessels=kept,
        scale=scale,
        xylem_area_px=float(h * w),
        extra_area_px=float(cv2.countNonZero(extra)),
        tissue_mask=tissue,
        extra_mask=extra,
        phloem_mask=phloem,
        pith_mask=pith,
        xylem_mask=xylem_tissue,
        phloem_boxes=phloem_boxes,
        pith_boxes=pith_boxes,
        xylem_boxes=xylem_boxes,
    )
