"""Convert messy JPG ink into the 6-point / two-ray annotation.

Point map (matches the recommended schematic):
  1 pith            green start, geometric center of the stem disk
  2 green_end       radius to outer bark
  3 seg_inner       layer ray starts at the pith–xylem ring (not the pith)
  4 xylem           xylem / phloem break, collinear with 3–6
  5 phloem          phloem / bark break, collinear with 3–6
  6 bark            layer ray ends on outer bark
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import LineGeometry, parse_manual_geometry
from calibration.io_util import imread
from calibration.region1_geom import (
    _inset_end,
    _nudge_seg_angle,
    _pick_seg_angle,
    _ray_to_bark,
    angle_sep,
    valid_angle_pair,
)
from calibration.stem import (
    detect_inner_ring_radius,
    detect_pith_center,
    detect_stem_mask,
    estimate_green_angle,
    point_on_ray,
)


def detect_pith_center_mask(stem_mask: np.ndarray) -> tuple[float, float]:
    """Fallback pith = distance-transform peak (disk center)."""
    dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y)


def geom_from_six_points(
    pith: tuple[float, float],
    green_end: tuple[float, float],
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    green_start: tuple[float, float] | None = None,
) -> LineGeometry:
    g0 = green_start if green_start is not None else pith
    cx, cy = g0
    green_ang = math.degrees(math.atan2(-(green_end[1] - cy), green_end[0] - cx))
    seg_ang = math.degrees(math.atan2(-(p3[1] - p0[1]), p3[0] - p0[0]))
    return LineGeometry(
        center=g0,
        green_start=g0,
        green_end=green_end,
        seg_y=float((p0[1] + p1[1]) / 2),
        seg_x0=float(p0[0]),
        xylem_end=float(p1[0]),
        phloem_end=float(p2[0]),
        bark_end=float(p3[0]),
        green_angle_deg=green_ang,
        seg_angle_deg=seg_ang,
        seg_y0=float(p0[1]),
        xylem_end_y=float(p1[1]),
        phloem_end_y=float(p2[1]),
        bark_end_y=float(p3[1]),
    )


def six_points(geom: LineGeometry) -> list[tuple[float, float]]:
    return [
        geom.green_start,
        geom.green_end,
        (geom.seg_x0, geom.seg_y0 or geom.seg_y),
        (geom.xylem_end, geom.xylem_end_y or geom.seg_y),
        (geom.phloem_end, geom.phloem_end_y or geom.seg_y),
        (geom.bark_end, geom.bark_end_y or geom.seg_y),
    ]


def geom_to_record(geom: LineGeometry) -> dict:
    pith = geom.center
    pts = six_points(geom)
    bark_r = math.hypot(pts[5][0] - pith[0], pts[5][1] - pith[1]) or 1.0

    def frac(pt: tuple[float, float]) -> float:
        return math.hypot(pt[0] - pith[0], pt[1] - pith[1]) / bark_r

    return {
        "pith": [pts[0][0], pts[0][1]],
        "green_end": [pts[1][0], pts[1][1]],
        "seg_inner": [pts[2][0], pts[2][1]],
        "xylem": [pts[3][0], pts[3][1]],
        "phloem": [pts[4][0], pts[4][1]],
        "bark": [pts[5][0], pts[5][1]],
        "green_angle_deg": float(geom.green_angle_deg),
        "seg_angle_deg": float(geom.seg_angle_deg),
        "ring_frac": frac(pts[2]),
        "xylem_frac": frac(pts[3]),
        "phloem_frac": frac(pts[4]),
    }


def geom_from_record(rec: dict) -> LineGeometry:
    return geom_from_six_points(
        tuple(rec["pith"]),
        tuple(rec["green_end"]),
        tuple(rec["seg_inner"]),
        tuple(rec["xylem"]),
        tuple(rec["phloem"]),
        tuple(rec["bark"]),
    )


def faithful_from_parse(parse: LineGeometry, stem_r: float) -> LineGeometry:
    """Keep pith (心) and orange inner; only force collinear layer points 3–6."""
    g0 = (float(parse.green_start[0]), float(parse.green_start[1]))
    g_end = parse.green_end
    p0 = (float(parse.seg_x0), float(parse.seg_y0 or parse.seg_y))
    p1 = (float(parse.xylem_end), float(parse.xylem_end_y or parse.seg_y))
    p2 = (float(parse.phloem_end), float(parse.phloem_end_y or parse.seg_y))
    p3 = (float(parse.bark_end), float(parse.bark_end_y or parse.seg_y))
    if math.hypot(p0[0] - g0[0], p0[1] - g0[1]) < 8.0:
        p0 = point_on_ray(g0, float(parse.seg_angle_deg), max(36.0, 0.10 * max(stem_r, 1.0)))
    vx, vy = p3[0] - p0[0], p3[1] - p0[1]
    n2 = vx * vx + vy * vy
    if n2 < 1:
        return geom_from_six_points(g0, g_end, p0, p1, p2, p3, green_start=g0)
    n = math.sqrt(n2)
    ux, uy = vx / n, vy / n

    def _along(pt: tuple[float, float]) -> float:
        return (pt[0] - p0[0]) * ux + (pt[1] - p0[1]) * uy

    d1 = max(4.0, _along(p1))
    d2 = max(d1 + 3.0, _along(p2))
    d3 = max(d2 + 3.0, n)
    p1 = (p0[0] + d1 * ux, p0[1] + d1 * uy)
    p2 = (p0[0] + d2 * ux, p0[1] + d2 * uy)
    p3 = (p0[0] + d3 * ux, p0[1] + d3 * uy)
    return geom_from_six_points(g0, g_end, p0, p1, p2, p3, green_start=g0)


def light_from_keypoints(
    kpts: np.ndarray,
    stem_mask: np.ndarray | None,
    mag: int = 2,
) -> LineGeometry | None:
    """YOLO keypoints with collinear 3–6, no bark reshoot — matches 人工墨线."""
    if kpts is None or kpts.shape[0] < 6:
        return None
    pith = (float(kpts[0, 0]), float(kpts[0, 1]))
    hint = geom_from_six_points(
        pith,
        (float(kpts[1, 0]), float(kpts[1, 1])),
        (float(kpts[2, 0]), float(kpts[2, 1])),
        (float(kpts[3, 0]), float(kpts[3, 1])),
        (float(kpts[4, 0]), float(kpts[4, 1])),
        (float(kpts[5, 0]), float(kpts[5, 1])),
    )
    stem_r = 1500.0
    if stem_mask is not None:
        dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
        stem_r = float(dist.max()) or stem_r
    return faithful_from_parse(hint, stem_r)


def _hint_radii(hint: LineGeometry, pith: tuple[float, float]) -> tuple[float, float, float, float] | None:
    """Return (ring, xylem, phloem, bark) distances if the four layer points are distinct."""
    pts = six_points(hint)[2:]
    rs = [math.hypot(p[0] - pith[0], p[1] - pith[1]) for p in pts]
    bark_r = max(rs[3], 1.0)
    if bark_r < 40:
        return None
    if rs[3] - rs[0] < 24:
        return None
    if rs[1] - rs[0] < 8 or rs[2] - rs[1] < 3 or rs[3] - rs[2] < 3:
        return None
    return rs[0], rs[1], rs[2], rs[3]


def _detect_layer_radii(
    img: np.ndarray,
    stem_mask: np.ndarray,
    pith: tuple[float, float],
    seg_ang: float,
    ring_r: float,
    bark_r: float,
) -> tuple[float, float]:
    from calibration.inference import _find_layer_radii, _stem_radial_profile

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    profile = _stem_radial_profile(gray, stem_mask, pith, seg_ang)
    if len(profile) < 40:
        xylem_r = ring_r + (bark_r - ring_r) * 0.88
        phloem_r = ring_r + (bark_r - ring_r) * 0.95
        return xylem_r, phloem_r
    _pe, xylem_r, phloem_r, _be = _find_layer_radii(profile)
    span = max(bark_r - ring_r, 8.0)
    xylem_r = float(np.clip(xylem_r, ring_r + 0.55 * span, bark_r - 10.0))
    phloem_r = float(np.clip(phloem_r, xylem_r + max(6.0, 0.02 * bark_r), bark_r - 4.0))
    return xylem_r, phloem_r


def canonicalize(
    img: np.ndarray,
    stem_mask: np.ndarray | None = None,
    hint: LineGeometry | None = None,
    mag: int = 2,
    ring_frac: float | None = None,
    trust_hint_pith: bool = False,
) -> LineGeometry | None:
    """Snap a noisy parse (or YOLO keypoints) onto the 6-point two-ray method."""
    if stem_mask is None:
        info = detect_stem_mask(img)
        if info is None:
            return None
        stem_mask, _ = info

    pith = detect_pith_center(img, stem_mask, mag=mag)
    if hint is not None:
        hx, hy = hint.center
        h, w = stem_mask.shape
        ix, iy = int(round(hx)), int(round(hy))
        if 0 <= ix < w and 0 <= iy < h and stem_mask[iy, ix] > 0:
            if trust_hint_pith:
                pith = (float(hx), float(hy))
            else:
                dist = math.hypot(hx - pith[0], hy - pith[1])
                dt = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
                stem_r = float(dt[int(round(pith[1])), int(round(pith[0]))]) or float(dt.max())
                max_pull = 0.10 * stem_r if mag <= 2 else 0.06 * stem_r
                if dist < max_pull:
                    pith = (0.70 * pith[0] + 0.30 * hx, 0.70 * pith[1] + 0.30 * hy)

    if hint is not None:
        green_ang = float(hint.green_angle_deg)
        seg_ang = float(hint.seg_angle_deg)
        if not valid_angle_pair(green_ang, seg_ang):
            seg_ang = _nudge_seg_angle(green_ang, seg_ang)
    else:
        green_ang = estimate_green_angle(img, stem_mask, pith)
        from calibration.inference import _estimate_seg_angle

        seg_ang = _estimate_seg_angle(img, stem_mask, pith, green_ang)
        seg_ang = _pick_seg_angle(green_ang, seg_ang)

    cx, cy = pith
    green_end = _ray_to_bark(img, stem_mask, cx, cy, green_ang)
    bark_end = _ray_to_bark(img, stem_mask, cx, cy, seg_ang)
    bark_r = math.hypot(bark_end[0] - cx, bark_end[1] - cy)
    if bark_r < 30:
        return None

    dt = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    stem_r = float(dt.max()) or bark_r
    prior = ring_frac if ring_frac is not None else (0.21 if mag <= 2 else 0.24)
    ring_r = detect_inner_ring_radius(img, stem_mask, pith)
    ring_r = 0.40 * ring_r + 0.60 * (prior * stem_r)
    min_ring = max(36.0, 0.12 * stem_r)
    max_ring = 0.32 * stem_r
    ring_r = float(np.clip(ring_r, min_ring, max_ring))

    if trust_hint_pith and hint is not None:
        hint_rs = _hint_radii(hint, pith)
    else:
        hint_rs = _hint_radii(hint, pith) if hint is not None else None
    if hint_rs is not None:
        h_ring, h_xy, h_ph, h_bk = hint_rs
        scale = bark_r / max(h_bk, 1.0)
        hint_ring = float(h_ring * scale)
        if trust_hint_pith:
            ring_r = float(np.clip(hint_ring, min_ring, max(max_ring, 0.36 * bark_r)))
            xylem_r = float(np.clip(h_xy * scale, ring_r + 8.0, bark_r - 8.0))
            phloem_r = float(np.clip(h_ph * scale, xylem_r + 4.0, bark_r - 4.0))
        else:
            if hint_ring >= min_ring:
                ring_r = float(np.clip(0.35 * ring_r + 0.65 * hint_ring, min_ring, max_ring))
            xylem_r = float(np.clip(h_xy * scale, ring_r + 8.0, bark_r - 8.0))
            phloem_r = float(np.clip(h_ph * scale, xylem_r + 4.0, bark_r - 4.0))
    else:
        xylem_r, phloem_r = _detect_layer_radii(img, stem_mask, pith, seg_ang, ring_r, bark_r)

    p0 = point_on_ray(pith, seg_ang, ring_r)
    p1 = point_on_ray(pith, seg_ang, xylem_r)
    p2 = point_on_ray(pith, seg_ang, phloem_r)
    return geom_from_six_points(pith, green_end, p0, p1, p2, bark_end)


def geom_from_keypoints(
    kpts: np.ndarray,
    img: np.ndarray,
    stem_mask: np.ndarray | None = None,
    mag: int = 2,
) -> LineGeometry | None:
    """YOLO 6 keypoints → snapped LineGeometry."""
    if kpts is None or kpts.shape[0] < 2:
        return None
    pith = (float(kpts[0, 0]), float(kpts[0, 1]))
    green_end = (float(kpts[1, 0]), float(kpts[1, 1]))
    if kpts.shape[0] >= 6:
        p0 = (float(kpts[2, 0]), float(kpts[2, 1]))
        p1 = (float(kpts[3, 0]), float(kpts[3, 1]))
        p2 = (float(kpts[4, 0]), float(kpts[4, 1]))
        p3 = (float(kpts[5, 0]), float(kpts[5, 1]))
        hint = geom_from_six_points(pith, green_end, p0, p1, p2, p3)
    else:
        hint = LineGeometry(
            center=pith,
            green_start=pith,
            green_end=green_end,
            seg_y=pith[1],
            seg_x0=pith[0],
            xylem_end=green_end[0],
            phloem_end=green_end[0],
            bark_end=green_end[0],
            green_angle_deg=math.degrees(math.atan2(-(green_end[1] - pith[1]), green_end[0] - pith[0])),
            seg_angle_deg=math.degrees(math.atan2(-(green_end[1] - pith[1]), green_end[0] - pith[0])) + 40.0,
            seg_y0=pith[1],
            xylem_end_y=green_end[1],
            phloem_end_y=green_end[1],
            bark_end_y=green_end[1],
        )
    return canonicalize(img, stem_mask, hint, mag=mag, trust_hint_pith=True)


def relabel_from_ink(base_dir: Path, json_path: Path | None = None) -> dict[str, dict]:
    """JSON labels = 人工墨线六点（端点不改射到树皮），供训练和对照。"""
    orig_dir = base_dir / "原图" / "第一区域"
    manual_dir = base_dir / "人工标定" / "第一区域"
    json_path = json_path or (base_dir / "推理结果" / "region1_sixpoint.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    skipped: list[str] = []
    files = sorted(manual_dir.glob("*.jpg"))
    print(f"[墨线JSON] 解析 {len(files)} 张人工标定...")
    sims: list[float] = []
    for i, mp in enumerate(files):
        orig = imread(orig_dir / mp.name)
        manual = imread(mp)
        if orig is None or manual is None:
            skipped.append(mp.name)
            continue
        parse = parse_manual_geometry(manual, orig)
        if parse is None:
            skipped.append(mp.name)
            continue
        info = detect_stem_mask(orig)
        stem_r = float(info[1]) if info is not None else 1500.0
        geom = faithful_from_parse(parse, stem_r)
        rec = geom_to_record(geom)
        rec["filename"] = mp.name
        records[mp.name] = rec
        h, w = orig.shape[:2]
        sims.append(region1_geometry_similarity_local(geom, parse, math.hypot(h, w)))
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i + 1}/{len(files)}: {mp.name}")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if sims:
        arr = np.array(sims, dtype=np.float64)
        print(
            f"[墨线JSON] 成功 {len(records)} 跳过 {len(skipped)} -> {json_path}\n"
            f"          标签相对人工墨线 mean={arr.mean():.1%}  >=95%={float((arr >= 0.95).mean()):.0%}"
        )
    return records


def region1_geometry_similarity_local(pred, gt, img_diag: float) -> float:
    from ml.metrics import region1_geometry_similarity

    return region1_geometry_similarity(pred, gt, img_diag)


def relabel_manuals(base_dir: Path, json_path: Path | None = None) -> dict[str, dict]:
    """Parse JPG ink, snap to 6 points, write JSON. Does not overwrite 人工标定."""
    orig_dir = base_dir / "原图" / "第一区域"
    manual_dir = base_dir / "人工标定" / "第一区域"
    json_path = json_path or (base_dir / "推理结果" / "region1_sixpoint.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    skipped: list[str] = []
    files = sorted(manual_dir.glob("*.jpg"))
    print(f"[六点改标] 解析 {len(files)} 张人工标定...")
    for i, mp in enumerate(files):
        orig = imread(orig_dir / mp.name)
        manual = imread(mp)
        if orig is None or manual is None:
            skipped.append(mp.name)
            continue
        hint = parse_manual_geometry(manual, orig)
        geom = canonicalize(orig, hint=hint)
        if geom is None:
            skipped.append(mp.name)
            continue
        rec = geom_to_record(geom)
        rec["filename"] = mp.name
        if hint is not None:
            rec["hint_sep_deg"] = angle_sep(hint.green_angle_deg, hint.seg_angle_deg)
        records[mp.name] = rec
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i + 1}/{len(files)}: {mp.name}")

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if records:
        ring = np.array([r["ring_frac"] for r in records.values()])
        xy = np.array([r["xylem_frac"] for r in records.values()])
        ph = np.array([r["phloem_frac"] for r in records.values()])
        print(
            f"[六点改标] 成功 {len(records)}  跳过 {len(skipped)}  -> {json_path}\n"
            f"          中位比例 内环={float(np.median(ring)):.2f}  "
            f"木质部={float(np.median(xy)):.2f}  韧皮部={float(np.median(ph)):.2f}"
        )
    return records


def load_relabel_records(base_dir: Path) -> dict[str, dict]:
    path = base_dir / "推理结果" / "region1_sixpoint.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_break_fracs(geom: LineGeometry) -> tuple[float, float]:
    """Return (xylem_t, phloem_t) in [0,1] along layer start→bark."""
    p0 = (float(geom.seg_x0), float(geom.seg_y0 or geom.seg_y))
    p1 = (float(geom.xylem_end), float(geom.xylem_end_y or geom.seg_y))
    p2 = (float(geom.phloem_end), float(geom.phloem_end_y or geom.seg_y))
    p3 = (float(geom.bark_end), float(geom.bark_end_y or geom.seg_y))
    span = math.hypot(p3[0] - p0[0], p3[1] - p0[1]) or 1.0
    t1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / span
    t2 = math.hypot(p2[0] - p0[0], p2[1] - p0[1]) / span
    t1 = float(np.clip(t1, 0.05, 0.95))
    t2 = float(np.clip(max(t2, t1 + 0.02), t1 + 0.02, 0.98))
    return t1, t2


def _json_layer_break_fracs(json_rec: dict, p0: tuple[float, float]) -> tuple[float, float]:
    """JSON 4 / 5 positions as fractions along JSON 3→6."""
    p1 = (float(json_rec["xylem"][0]), float(json_rec["xylem"][1]))
    p2 = (float(json_rec["phloem"][0]), float(json_rec["phloem"][1]))
    p3 = (float(json_rec["bark"][0]), float(json_rec["bark"][1]))
    span = math.hypot(p3[0] - p0[0], p3[1] - p0[1]) or 1.0
    t1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / span
    t2 = math.hypot(p2[0] - p0[0], p2[1] - p0[1]) / span
    t1 = float(np.clip(t1, 0.05, 0.95))
    t2 = float(np.clip(max(t2, t1 + 0.02), t1 + 0.02, 0.98))
    return t1, t2


def _rescue_green_end(
    img: np.ndarray,
    stem_mask: np.ndarray,
    pith: tuple[float, float],
    green_end: tuple[float, float],
    json_rec: dict,
    min_frac: float = 0.10,
) -> tuple[float, float]:
    """If the predicted 1→2 ray collapsed (pith at the crop edge), use JSON direction."""
    dt = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    stem_r = float(dt.max()) or 1.0
    min_len = max(80.0, min_frac * stem_r)
    if math.hypot(green_end[0] - pith[0], green_end[1] - pith[1]) >= min_len:
        return green_end
    json_ang = json_rec.get("green_angle_deg")
    if json_ang is not None:
        rescued = _ray_to_bark(img, stem_mask, pith[0], pith[1], float(json_ang))
        if math.hypot(rescued[0] - pith[0], rescued[1] - pith[1]) >= min_len:
            return rescued
    je = json_rec.get("green_end")
    if je is not None:
        return (float(je[0]), float(je[1]))
    return green_end


def _ray_until_slide(
    img: np.ndarray,
    origin: tuple[float, float],
    toward: tuple[float, float],
    inset: float = 6.0,
) -> tuple[float, float]:
    """Walk origin→toward until the white slide; keeps dark outer bark.

    Interior vessel lumens and cracks are ignored until past the JSON bark hint.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape[:2]
    vx, vy = toward[0] - origin[0], toward[1] - origin[1]
    n = math.hypot(vx, vy) or 1.0
    ux, uy = vx / n, vy / n
    last = (float(origin[0]), float(origin[1]))
    min_t = max(0.0, n - 40.0)
    max_t = n + 70.0
    for t in range(0, int(max(h, w) * 1.2)):
        x = origin[0] + t * ux
        y = origin[1] + t * uy
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= w or iy >= h or t > max_t:
            extra = max(inset, 22.0)
            return _inset_end(origin, last, extra)
        g = int(gray[iy, ix])
        s = int(hsv[iy, ix, 1])
        slide = (g >= 218 and s < 48) or g >= 242
        if slide and t >= min_t:
            break
        if not slide:
            last = (float(x), float(y))
        elif t < min_t:
            last = (float(x), float(y))
    extra = inset
    ix, iy = int(round(last[0])), int(round(last[1]))
    if ix < 10 or iy < 10 or ix >= w - 10 or iy >= h - 10:
        extra = max(extra, 22.0)
    return _inset_end(origin, last, extra)


def _rescue_layer_end(
    img: np.ndarray,
    stem_mask: np.ndarray,
    p0: tuple[float, float],
    p3: tuple[float, float],
    json_rec: dict,
    green_end: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], bool]:
    """If 3→6 stopped inside xylem (crop + 16° nudge), reuse JSON layer to bark."""
    span = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
    jb = json_rec.get("bark")
    json_span = 0.0
    if jb is not None:
        json_span = math.hypot(float(jb[0]) - p0[0], float(jb[1]) - p0[1])
    pith = json_rec.get("pith")
    green_len = 0.0
    if green_end is not None and pith is not None:
        green_len = math.hypot(green_end[0] - float(pith[0]), green_end[1] - float(pith[1]))
    dt = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    stem_r = float(dt.max()) or 1.0
    min_len = max(120.0, 0.35 * stem_r)
    if json_span > 0:
        min_len = max(min_len, 0.80 * json_span)
    if green_len > 0:
        min_len = max(min_len, 0.55 * green_len)
    ix, iy = int(round(p3[0])), int(round(p3[1]))
    h, w = stem_mask.shape[:2]
    inside = float(dt[iy, ix]) if 0 <= iy < h and 0 <= ix < w else 0.0
    at_bark = inside <= max(36.0, 0.04 * stem_r)
    if span >= min_len and at_bark:
        return p3, False
    toward = None
    json_ang = json_rec.get("seg_angle_deg")
    if jb is not None:
        toward = (float(jb[0]), float(jb[1]))
    elif json_ang is not None:
        toward = point_on_ray(p0, float(json_ang), max(min_len, 2.0 * stem_r))
    if toward is not None:
        rescued = _ray_until_slide(img, p0, toward, inset=6.0)
        if math.hypot(rescued[0] - p0[0], rescued[1] - p0[1]) >= min_len * 0.85:
            return rescued, True
        if jb is not None:
            return (float(jb[0]), float(jb[1])), True
    return p3, False


def geom_json_starts_predicted_rays(
    img: np.ndarray,
    json_rec: dict,
    pred: LineGeometry | None,
    stem_mask: np.ndarray | None = None,
    mag: int = 2,
) -> LineGeometry | None:
    """Fix green/layer starts from LabelMe JSON; shoot rays by predicted angles.

    - Point 1 (pith) and point 3 (seg_inner) come from JSON.
    - Green / layer directions and bark ends come from YOLO (or CV fallback).
    - If the predicted 1→2 ray is too short (crop-edge pith), reuse JSON green.
    - If 3→6 stops inside the xylem (16° nudge off a cropped bark), reuse JSON layer.
    - Segment 4→5 length comes from JSON (absolute px); point 4 position follows
      the predicted fraction along the new layer ray (clamped to fit 4→5 + tip).
    """
    if stem_mask is None:
        info = detect_stem_mask(img)
        if info is None:
            return None
        stem_mask, _ = info

    g0 = (float(json_rec["pith"][0]), float(json_rec["pith"][1]))
    p0 = (float(json_rec["seg_inner"][0]), float(json_rec["seg_inner"][1]))
    j4 = (float(json_rec["xylem"][0]), float(json_rec["xylem"][1]))
    j5 = (float(json_rec["phloem"][0]), float(json_rec["phloem"][1]))
    len_45 = math.hypot(j5[0] - j4[0], j5[1] - j4[1])
    len_45 = max(float(len_45), 4.0)

    if pred is None:
        pred = canonicalize(img, stem_mask, hint=None, mag=mag)
        if pred is None:
            return None

    green_ang = float(pred.green_angle_deg)
    seg_ang = float(pred.seg_angle_deg)
    if not valid_angle_pair(green_ang, seg_ang):
        seg_ang = _nudge_seg_angle(green_ang, seg_ang)

    g1 = _ray_to_bark(img, stem_mask, g0[0], g0[1], green_ang)
    g1 = _rescue_green_end(img, stem_mask, g0, g1, json_rec)
    p3 = _ray_to_bark(img, stem_mask, p0[0], p0[1], seg_ang)
    p3, layer_from_json = _rescue_layer_end(img, stem_mask, p0, p3, json_rec, g1)
    span = math.hypot(p3[0] - p0[0], p3[1] - p0[1]) or 1.0
    ux, uy = (p3[0] - p0[0]) / span, (p3[1] - p0[1]) / span

    if layer_from_json:
        t1, t2 = _json_layer_break_fracs(json_rec, p0)
    else:
        t1, t2 = _layer_break_fracs(pred)
    # Leave room for JSON 4→5 and a short 5→6 tip before bark.
    tip = max(4.0, min(0.05 * span, span * 0.08))
    max_d1 = max(4.0, span - len_45 - tip)
    d1 = float(np.clip(t1 * span, 4.0, max_d1))
    d2 = d1 + len_45
    if layer_from_json:
        d2 = float(np.clip(t2 * span, d1 + 2.0, span - 2.0))
        d1 = float(np.clip(t1 * span, 4.0, d2 - 2.0))
    elif d2 > span - tip:
        d2 = max(d1 + 2.0, span - tip)
        d1 = max(4.0, d2 - len_45)
    p1 = (p0[0] + ux * d1, p0[1] + uy * d1)
    p2 = (p0[0] + ux * d2, p0[1] + uy * d2)
    d45 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    d56 = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
    if d56 > 1.0 and d45 < 3.0 * d56:
        p2 = (p1[0] + 0.75 * (p3[0] - p1[0]), p1[1] + 0.75 * (p3[1] - p1[1]))
    return geom_from_six_points(g0, g1, p0, p1, p2, p3, green_start=g0)


_LABELME_POINT_MAP = {
    "1": "pith",
    "2": "green_end",
    "3": "seg_inner",
    "4": "xylem",
    "5": "phloem",
    "6": "bark",
}


def parse_labelme_sixpoint(data: dict) -> dict | None:
    """Parse one LabelMe JSON (labels 1–6 points, optional line 10) into a six-point record."""
    by_label: dict[str, list] = {}
    for shape in data.get("shapes", []):
        lab = str(shape.get("label", "")).strip()
        pts = shape.get("points") or []
        st = shape.get("shape_type")
        if lab in _LABELME_POINT_MAP and st == "point" and len(pts) >= 1:
            # keep first occurrence if duplicates slipped through
            by_label.setdefault(lab, pts[0])
        elif lab == "10" and st in ("line", "linestrip") and len(pts) >= 2:
            by_label["10"] = pts[:2]
    if any(k not in by_label for k in _LABELME_POINT_MAP):
        return None
    pith = (float(by_label["1"][0]), float(by_label["1"][1]))
    green_end = (float(by_label["2"][0]), float(by_label["2"][1]))
    p0 = (float(by_label["3"][0]), float(by_label["3"][1]))
    p1 = (float(by_label["4"][0]), float(by_label["4"][1]))
    p2 = (float(by_label["5"][0]), float(by_label["5"][1]))
    p3 = (float(by_label["6"][0]), float(by_label["6"][1]))
    geom = geom_from_six_points(pith, green_end, p0, p1, p2, p3, green_start=pith)
    return geom_to_record(geom)


def load_labelme_records(
    base_dir: Path,
    json_out: Path | None = None,
    region: str = "第一区域",
) -> dict[str, dict]:
    """Load six-point labels from LabelMe JSONs next to originals under 原图/<region>."""
    orig_dir = base_dir / "原图" / region
    json_out = json_out or (base_dir / "推理结果" / "region1_sixpoint.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    skipped: list[str] = []
    for jp in sorted(orig_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            skipped.append(jp.name)
            continue
        image_name = data.get("imagePath") or (jp.stem + ".jpg")
        img_path = orig_dir / image_name
        if not img_path.exists():
            # fall back to same stem jpg
            alt = orig_dir / f"{jp.stem}.jpg"
            if alt.exists():
                image_name = alt.name
                img_path = alt
            else:
                skipped.append(jp.name)
                continue
        rec = parse_labelme_sixpoint(data)
        if rec is None:
            skipped.append(jp.name)
            continue
        rec["filename"] = image_name
        rec["source"] = "labelme"
        records[image_name] = rec

    json_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LabelMe] 载入 {len(records)} 张，跳过 {len(skipped)} -> {json_out}")
    if skipped:
        print(f"  跳过: {skipped[:8]}{'...' if len(skipped) > 8 else ''}")
    return records
