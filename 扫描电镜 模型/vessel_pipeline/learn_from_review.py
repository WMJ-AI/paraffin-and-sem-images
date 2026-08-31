"""
解析人工标定笔画（蓝/黄框、短线），仅用于离线归纳通用规则。

生产只依赖 output/learned_rules.json：按图几何识别多导管区并自动画 TVW 线。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DATA = Path(r"H:\尉明杰\扫描电镜 模型")
PIPELINE = Path(__file__).resolve().parent


@dataclass
class OrientedBox:
    cx: float
    cy: float
    w: float
    h: float
    angle: float

    def contains_point(self, x: float, y: float, pad: float = 8.0) -> bool:
        rad = np.deg2rad(self.angle)
        dx, dy = x - self.cx, y - self.cy
        lx = dx * np.cos(rad) + dy * np.sin(rad)
        ly = -dx * np.sin(rad) + dy * np.cos(rad)
        return abs(lx) <= self.w / 2 + pad and abs(ly) <= self.h / 2 + pad


@dataclass
class YellowGuide:
    image_key: str  # e.g. "1 (2).tif" or "1_(2)"
    boxes: list[OrientedBox]
    lines: list[tuple[np.ndarray, np.ndarray]]  # (p0, p1) each
    force_zero_pair: bool = False


def _review_root() -> Path | None:
    for p in DATA.iterdir():
        if p.is_dir() and "复查" in p.name:
            return p
    return None


def image_key_from_stem(stem: str) -> str:
    """1_(2)_自动标注 / 01_(02)_左原图_右自动 -> 1 (2).tif（去前导零以便匹配批次原图）"""
    stem = (
        stem.replace("_自动标注", "")
        .replace("_对照_原图_自动", "")
        .replace("_左原图_右自动", "")
    )
    m = re.match(r"(\d+)[_\s]*\((\d+)\)", stem)
    if m:
        return f"{int(m.group(1))} ({int(m.group(2))}).tif"
    return stem


def _yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (15, 70, 100), (45, 255, 255))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    m2 = ((r > 170) & (g > 150) & (b < 130) & (r > b + 35) & (g > b + 35)).astype(np.uint8) * 255
    return cv2.bitwise_or(m1, m2)


def _blue_mask(bgr: np.ndarray) -> np.ndarray:
    """Human marks on 已标定 often use bright blue / cyan (not yellow)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # cyan–blue, exclude green auto contours and purple Ai
    m1 = cv2.inRange(hsv, (85, 50, 70), (140, 255, 255))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    m2 = ((b > 140) & (b > r + 35) & (b > g + 15) & (r < 170)).astype(np.uint8) * 255
    return cv2.bitwise_or(m1, m2)


def _human_mark_mask(bgr: np.ndarray) -> np.ndarray:
    """Yellow (复查) or blue (人工标定对照左半) strokes."""
    m = cv2.bitwise_or(_yellow_mask(bgr), _blue_mask(bgr))
    # drop thin green auto leftovers if any leaked into left crop
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (40, 80, 80), (85, 255, 255))
    m = cv2.bitwise_and(m, cv2.bitwise_not(green))
    return m


def _expand_box(b: OrientedBox, scale: float = 1.2) -> OrientedBox:
    return OrientedBox(b.cx, b.cy, b.w * scale, b.h * scale, b.angle)


def _boxes_from_yellow(mask: np.ndarray, min_area: int = 800) -> list[OrientedBox]:
    """Yellow rectangle strokes → oriented boxes via hole fill / minAreaRect."""
    h, w = mask.shape
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    inv = cv2.bitwise_not(m)
    ff = inv.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[seed[1], seed[0]] != 0:
            cv2.floodFill(ff, flood, seed, 0)
    holes = ff
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    boxes: list[OrientedBox] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or min(bw, bh) < 40:
            continue
        if max(bw, bh) > max(w, h) * 0.9:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        comp = cv2.dilate(comp, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        boxes.append(OrientedBox(float(cx), float(cy), float(rw), float(rh), float(ang)))
    return boxes


def _lines_from_yellow(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Short yellow strokes as TVW guide segments."""
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    lines = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 40 or area > 2500:
            continue
        long, short = max(bw, bh), min(bw, bh)
        if long < 12 or long / max(short, 1) < 1.8:
            # very short mark still ok if elongated-ish or medium
            if area < 80 or long < 10:
                continue
        ys, xs = np.where(labels == i)
        if len(xs) < 8:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        axis = vt[0]
        proj = (pts - c) @ axis
        p0 = c + axis * proj.min()
        p1 = c + axis * proj.max()
        if np.linalg.norm(p1 - p0) < 8:
            continue
        lines.append((p0, p1))
    return lines


def _imread_unicode(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _ensure(guides: dict[str, YellowGuide], key: str) -> YellowGuide:
    if key not in guides:
        guides[key] = YellowGuide(image_key=key, boxes=[], lines=[])
    return guides[key]


def _panel_for_marks(bgr: np.ndarray, path: Path | None = None) -> np.ndarray:
    """对照图取左半（原图侧）；单幅原图整图。"""
    if bgr is None:
        return bgr
    h, w = bgr.shape[:2]
    name = path.name if path is not None else ""
    if "左原图_右自动" in name or "对照" in name or w >= int(h * 1.6):
        return bgr[:, : w // 2]
    return bgr


def filter_valid_multi_boxes(
    boxes: list[OrientedBox],
    min_side: float = 28.0,
    min_area: float = 1800.0,
) -> list[OrientedBox]:
    """丢掉高分标记噪声小框；保留多导管区尺度的框。"""
    out = []
    for b in boxes:
        if min(b.w, b.h) < min_side:
            continue
        if b.w * b.h < min_area:
            continue
        out.append(b)
    return out


def _ingest_marked_bgr(guides: dict[str, YellowGuide], key: str, bgr: np.ndarray) -> None:
    """Parse human boxes + TVW lines (yellow or blue)."""
    if bgr is None:
        return
    y = _human_mark_mask(bgr)
    if int((y > 0).sum()) < 40:
        return
    g = _ensure(guides, key)
    # 轻微外扩即可；过大易吞进远处第三腔
    boxes = filter_valid_multi_boxes(
        [_expand_box(b, 1.12) for b in _boxes_from_yellow(y)]
    )
    lines = _lines_from_yellow(y)
    if boxes:
        g.boxes = boxes
    if lines:
        g.lines = lines


def load_human_mark_guides(mark_root: Path | None) -> dict[str, YellowGuide]:
    """Load human-marked PNGs from mark_root (and optional 已标定/ subfolder)."""
    guides: dict[str, YellowGuide] = {}
    if mark_root is None or not mark_root.exists():
        return guides
    roots = []
    done = mark_root / "已标定"
    if done.exists():
        roots.append(done)
    roots.append(mark_root)
    seen_files: set[str] = set()
    for root in roots:
        for f in root.rglob("*.png"):
            if "待标定" in str(f):
                continue
            # skip output compare dumps if nested
            if "对照图" in str(f):
                continue
            rp = str(f.resolve())
            if rp in seen_files:
                continue
            seen_files.add(rp)
            key = image_key_from_stem(f.stem)
            bgr = _imread_unicode(f)
            if bgr is None:
                continue
            _ingest_marked_bgr(guides, key, _panel_for_marks(bgr, f))
            for side in (
                f.with_name(f.stem + "_无对.txt"),
                f.with_suffix(".无对.txt"),
                root / f"{f.stem}_无对.txt",
            ):
                if side.exists():
                    _ensure(guides, key).force_zero_pair = True
    return guides

def load_compare_left_guides(compare_root: Path | None) -> dict[str, YellowGuide]:
    """If user painted marks on left half of 对照图, parse that half."""
    guides: dict[str, YellowGuide] = {}
    if compare_root is None or not compare_root.exists():
        return guides
    for f in compare_root.rglob("*_左原图_右自动.png"):
        bgr = _imread_unicode(f)
        if bgr is None:
            continue
        left = _panel_for_marks(bgr, f)
        key = image_key_from_stem(f.stem)
        if int((_human_mark_mask(left) > 0).sum()) < 80:
            continue
        _ingest_marked_bgr(guides, key, left)
    return guides


def load_review_guides() -> dict[str, YellowGuide]:
    """Map 'N (M).tif' -> YellowGuide from 复查(1)."""
    root = _review_root()
    guides: dict[str, YellowGuide] = {}
    if root is None:
        g20 = _ensure(guides, "20 (1).tif")
        g20.force_zero_pair = True
        return guides

    region_dir = wall_dir = None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if "区域" in d.name:
            region_dir = d
        elif "壁厚" in d.name or "壁" in d.name:
            wall_dir = d
    if region_dir is None or wall_dir is None:
        dirs = [d for d in root.iterdir() if d.is_dir()]
        dirs.sort(key=lambda d: len(list(d.glob("*.png"))))
        if len(dirs) >= 2:
            if region_dir is None:
                region_dir = (
                    dirs[0]
                    if len(list(dirs[0].glob("*.png")))
                    <= len(list(dirs[-1].glob("*.png")))
                    else dirs[-1]
                )
            if wall_dir is None:
                wall_dir = dirs[-1] if dirs[-1] != region_dir else dirs[0]

    if region_dir:
        for f in region_dir.glob("*.png"):
            key = image_key_from_stem(f.stem)
            bgr = _imread_unicode(f)
            if bgr is None:
                continue
            boxes = _boxes_from_yellow(_yellow_mask(bgr))
            g = _ensure(guides, key)
            g.boxes.extend(boxes)

    if wall_dir:
        for f in wall_dir.glob("*.png"):
            key = image_key_from_stem(f.stem)
            bgr = _imread_unicode(f)
            if bgr is None:
                continue
            lines = _lines_from_yellow(_yellow_mask(bgr))
            g = _ensure(guides, key)
            g.lines.extend(lines)

    g20 = _ensure(guides, "20 (1).tif")
    g20.force_zero_pair = True
    return guides


def load_all_yellow_guides(
    mark_root: Path | None = None,
    compare_root: Path | None = None,
) -> dict[str, YellowGuide]:
    """合并笔画：复查 < 对照图左半 < 人工标定（不落盘、不按文件名缓存）。"""
    guides: dict[str, YellowGuide] = {}
    for src in (
        load_review_guides(),
        load_compare_left_guides(compare_root),
        load_human_mark_guides(mark_root),
    ):
        for key, g in src.items():
            dst = _ensure(guides, key)
            if g.boxes:
                dst.boxes = list(g.boxes)
            if g.lines:
                dst.lines = list(g.lines)
            if g.force_zero_pair:
                dst.force_zero_pair = True
    return guides


def _synthetic_pair(a, b, um_per_px: float, image_name: str, pair_idx: int):
    from pair_tvw import (
        VesselPair,
        _contour_points,
        lines_to_pair_fields,
        measure_nearest_distance_lines,
        min_contour_distance,
    )

    pts1 = _contour_points(a.contour)
    pts2 = _contour_points(b.contour)
    m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
    prefix = f"{m.group(1)}({m.group(2)})" if m else image_name
    pid = f"{prefix}-P{pair_idx:02d}"
    raw = measure_nearest_distance_lines(
        pts1,
        pts2,
        n_lines=5,
        min_len_px=1.5 / max(um_per_px, 1e-9),
        max_len_px=45.0 / max(um_per_px, 1e-9),
        min_sep_px=12.0,
    )
    if not raw:
        _, i1, i2 = min_contour_distance(pts1, pts2)
        p1, p2 = pts1[i1], pts2[i2]
        raw = [
            (
                float(p1[0]),
                float(p1[1]),
                float(p2[0]),
                float(p2[1]),
                float(np.linalg.norm(p1 - p2)),
                0.5,
            )
        ]
    lines, med, mean, sd, c1, c2, cp = lines_to_pair_fields(
        raw, a, b, um_per_px, pid, image_name
    )
    a.pair_id = pid
    b.pair_id = pid
    return VesselPair(
        pair_id=pid,
        image_name=image_name,
        v1=a,
        v2=b,
        lines=lines,
        tvw_median_um=med,
        tvw_mean_um=mean,
        tvw_sd_um=sd,
        cwr1=c1,
        cwr2=c2,
        cwr_pair=cp,
    )


def _vessels_in_box(vessels, box: OrientedBox, used: set[str], pad: float):
    """Strict-then-relaxed membership so large rotated boxes don't swallow distant lumens."""
    strict = [
        v
        for v in vessels
        if v.vessel_id not in used and box.contains_point(v.cx, v.cy, pad=min(pad, 18.0))
    ]
    if len(strict) >= 2:
        return strict
    loose = [
        v
        for v in vessels
        if v.vessel_id not in used and box.contains_point(v.cx, v.cy, pad=pad)
    ]
    return loose if len(loose) >= 2 else strict


def _guide_mids_near_box(box: OrientedBox, guide_lines, pad: float = 40.0):
    mids = []
    for p0, p1 in guide_lines or []:
        mid = 0.5 * (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float))
        if box.contains_point(float(mid[0]), float(mid[1]), pad=pad):
            mids.append(mid)
    return mids


def _pick_adjacent_pair_in_box(inside, um_per_px: float, guide_mids):
    """
    框内不止两个腔时：选共同壁最近、且靠近人工壁厚短线的一对。
    不再简单取 Ai 最大的两个（1(6) 会误配远处第三腔）。
    """
    from pair_tvw import _contour_points, min_contour_distance

    if len(inside) < 2:
        return None
    if len(inside) == 2:
        a, b = sorted(inside, key=lambda v: v.ai_um2 or 0, reverse=True)
        return a, b

    best = None  # (score, gap_um, a, b)
    for i in range(len(inside)):
        for j in range(i + 1, len(inside)):
            a, b = inside[i], inside[j]
            pts1 = _contour_points(a.contour)
            pts2 = _contour_points(b.contour)
            dist, ia, ib = min_contour_distance(pts1, pts2)
            gap_um = float(dist * um_per_px)
            # 允许贴合/微距（后续 ensure_pair_lumen_gap 会拉开）；拒远距假对
            if gap_um > 28.0:
                continue
            wall_mid = 0.5 * (pts1[ia] + pts2[ib])
            ai_min = float(min(a.ai_um2 or 0, b.ai_um2 or 0))
            ai_sum = float((a.ai_um2 or 0) + (b.ai_um2 or 0))
            # 人工短线落在共同壁附近 → 强加分
            line_term = 0.0
            if guide_mids:
                d_line = min(float(np.linalg.norm(m - wall_mid)) for m in guide_mids)
                if d_line > 90.0:
                    continue
                line_term = 500.0 / (1.0 + d_line)
            # 近壁优先，其次面积
            score = line_term + ai_min * 0.05 + ai_sum * 0.01 - gap_um * 8.0
            if best is None or score > best[0]:
                best = (score, gap_um, a, b)

    if best is not None:
        return best[2], best[3]

    # 无合格近壁对：退回框内 Ai 最大的两个
    ranked = sorted(inside, key=lambda v: v.ai_um2 or 0, reverse=True)
    return ranked[0], ranked[1]


def pair_from_two_largest_in_boxes(
    vessels,
    boxes: list[OrientedBox],
    um_per_px: float,
    image_name: str,
    pad: float = 55.0,
    guide_lines=None,
):
    """
    每个人工蓝框 → 1 个导管对。
    框内选共同壁最近的两腔（参考人工短线位置）；再测 3–5 条最近距离 TVW。
    """
    from pair_tvw import find_pairs

    if len(vessels) < 2 or not boxes:
        return []

    for v in vessels:
        v.pair_id = None

    out = []
    used: set[str] = set()
    pair_idx = 1
    boxes = filter_valid_multi_boxes(boxes)
    ordered = sorted(boxes, key=lambda b: (b.cx, b.cy))
    for box in ordered:
        inside = _vessels_in_box(vessels, box, used, pad=pad)
        # 有人工框时禁止“抓远处未占用腔”兜底，避免 2(2)/2(3) 误配
        if len(inside) < 2:
            continue
        guide_mids = _guide_mids_near_box(box, guide_lines)
        picked = _pick_adjacent_pair_in_box(inside, um_per_px, guide_mids)
        if picked is None:
            continue
        a, b = picked
        if (a.ai_um2 or 0) < (b.ai_um2 or 0):
            a, b = b, a
        from pair_tvw import min_contour_distance, _contour_points

        gap_um = (
            min_contour_distance(_contour_points(a.contour), _contour_points(b.contour))[0]
            * um_per_px
        )
        if gap_um > 28.0:
            continue
        raw = find_pairs(
            [a, b],
            um_per_px=um_per_px,
            image_name=image_name,
            min_gap_um=0.0,
            max_gap_um=30.0,
            min_interface_points=3,
            min_tvw_um=0.2,
            max_tvw_um=30.0,
            max_ai_ratio=12.0,
            min_pair_ai_um2=70.0,
        )
        if raw:
            p = raw[0]
            m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
            prefix = f"{m.group(1)}({m.group(2)})" if m else image_name
            pid = f"{prefix}-P{pair_idx:02d}"
            p.pair_id = pid
            p.v1.pair_id = pid
            p.v2.pair_id = pid
            for ln in p.lines:
                ln.pair_id = pid
            out.append(p)
        else:
            p = _synthetic_pair(a, b, um_per_px, image_name, pair_idx)
            if p.tvw_mean_um > 28.0 or not p.lines:
                continue
            out.append(p)
        used.add(a.vessel_id)
        used.add(b.vessel_id)
        pair_idx += 1
    return out


def filter_pairs_by_boxes(pairs, boxes: list[OrientedBox]):
    """Keep pairs whose midpoint lies in any yellow box."""
    if not boxes:
        return pairs
    kept = []
    for p in pairs:
        mx = 0.5 * (p.v1.cx + p.v2.cx)
        my = 0.5 * (p.v1.cy + p.v2.cy)
        if any(b.contains_point(mx, my) for b in boxes):
            kept.append(p)
    return kept


def _pair_wall_anchor(pair) -> np.ndarray:
    """Approximate common-wall midpoint for assigning human TVW strokes."""
    from pair_tvw import _contour_points

    pts1 = _contour_points(pair.v1.contour)
    pts2 = _contour_points(pair.v2.contour)
    # coarse nearest between contours
    best = None
    step = max(1, len(pts1) // 40)
    for p in pts1[::step]:
        d2 = np.sum((pts2[::step] - p) ** 2, axis=1)
        k = int(np.argmin(d2))
        d = float(d2[k])
        if best is None or d < best[0]:
            best = (d, p, pts2[::step][k])
    if best is None:
        return np.array([0.5 * (pair.v1.cx + pair.v2.cx), 0.5 * (pair.v1.cy + pair.v2.cy)])
    return 0.5 * (best[1] + best[2])


def _snap_guide_stroke_to_wall(pts1, pts2, p0, p1) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Keep the lateral position of a human short stroke; snap ends to the two lumen contours.
    Returns (q1, q2, length_px) or None.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    mid = 0.5 * (p0 + p1)
    axis = p1 - p0
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        axis = np.array([1.0, 0.0])
    else:
        axis = axis / L

    # Prefer sample on the stroke closest to both walls (sum of distances)
    best_s = mid
    best_cost = 1e18
    for t in np.linspace(0.0, 1.0, 9):
        s = p0 + t * (p1 - p0)
        d1 = float(np.min(np.sum((pts1 - s) ** 2, axis=1)))
        d2 = float(np.min(np.sum((pts2 - s) ** 2, axis=1)))
        cost = d1 + d2
        if cost < best_cost:
            best_cost = cost
            best_s = s

    i1 = int(np.argmin(np.sum((pts1 - best_s) ** 2, axis=1)))
    i2 = int(np.argmin(np.sum((pts2 - best_s) ** 2, axis=1)))
    q1, q2 = pts1[i1], pts2[i2]
    length_px = float(np.linalg.norm(q2 - q1))
    if length_px < 2.0 or length_px > 120.0:
        return None
    return q1, q2, length_px


def apply_human_tvw_guides(pairs, guide_lines, um_per_px: float):
    """
    把人工蓝/黄短线按最近导管对分配，吸附到两腔壁后作为 TVW 绘制与计算。
    每人造短线 → 一条壁厚线，位置与人工标定一致。
    """
    from pair_tvw import TVWLine, _contour_points

    if not pairs or not guide_lines:
        return pairs

    anchors = [_pair_wall_anchor(p) for p in pairs]
    buckets: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in pairs]
    for p0, p1 in guide_lines:
        mid = 0.5 * (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float))
        best_i, best_d = 0, 1e18
        for i, a in enumerate(anchors):
            d = float(np.linalg.norm(mid - a))
            if d < best_d:
                best_d = d
                best_i = i
        if best_d <= 70.0:
            buckets[best_i].append((np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)))

    for i, pair in enumerate(pairs):
        guides = buckets[i]
        if not guides:
            continue
        # order along wall tangent for stable L1..Ln
        a = anchors[i]
        # wall direction approx v1->v2 orthogonal
        tdir = np.array([-(pair.v2.cy - pair.v1.cy), pair.v2.cx - pair.v1.cx], dtype=float)
        nrm = np.linalg.norm(tdir)
        if nrm > 1e-6:
            tdir = tdir / nrm
        guides = sorted(guides, key=lambda g: float(np.dot(0.5 * (g[0] + g[1]) - a, tdir)))

        pts1 = _contour_points(pair.v1.contour)
        pts2 = _contour_points(pair.v2.contour)
        lines = []
        tvws = []
        for k, (p0, p1) in enumerate(guides, start=1):
            got = _snap_guide_stroke_to_wall(pts1, pts2, p0, p1)
            if got is None:
                continue
            q1, q2, length_px = got
            tvw_um = length_px * um_per_px
            tvws.append(tvw_um)
            lines.append(
                TVWLine(
                    pair_id=pair.pair_id,
                    image_name=pair.image_name,
                    line_id=f"L{k}",
                    x0=float(q1[0]),
                    y0=float(q1[1]),
                    x1=float(q2[0]),
                    y1=float(q2[1]),
                    length_px=length_px,
                    um_per_px=um_per_px,
                    tvw_um=tvw_um,
                    quantile=0.5,
                )
            )
        if not lines:
            continue
        tvw_mean = float(np.mean(tvws))
        pair.lines = lines
        pair.tvw_mean_um = tvw_mean
        pair.tvw_median_um = float(np.median(tvws))
        pair.tvw_sd_um = float(np.std(tvws, ddof=1)) if len(tvws) > 1 else 0.0
        pair.cwr1 = (tvw_mean / pair.v1.di_um) ** 2
        pair.cwr2 = (tvw_mean / pair.v2.di_um) ** 2
        pair.cwr_pair = 0.5 * (pair.cwr1 + pair.cwr2)
    return pairs


def refine_pair_tvw_with_lines(pair, guide_lines, um_per_px: float):
    """Backward-compatible single-pair wrapper."""
    out = apply_human_tvw_guides([pair], guide_lines, um_per_px)
    return out[0] if out else pair
