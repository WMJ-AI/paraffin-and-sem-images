"""Unified pixel coordinate system for region-1 start points.

Origin: image top-left. +x right, +y down. Unit: pixel.
Same axes for 2X and 4X; values differ because the crop/scale differs.

Also stored:
  nx, ny = x / width, y / height          (image-normalized)
  ox, oy = (x - stem_cx) / stem_r         (stem-normalized, comparable across mag)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import parse_manual_geometry
from calibration.io_util import imread, parse_name
from calibration.stem import detect_pith, detect_stem_mask


COORD_META = {
    "origin": "top-left",
    "x": "rightward",
    "y": "downward",
    "unit": "pixel",
    "normalized": "nx=x/width, ny=y/height",
    "stem_frame": "ox=(x-stem_cx)/stem_r, oy=(y-stem_cy)/stem_r",
}


def _pt(x: float, y: float, w: int, h: int, cx: float, cy: float, stem_r: float) -> dict:
    sr = max(stem_r, 1.0)
    return {
        "x": float(x),
        "y": float(y),
        "nx": float(x / max(w, 1)),
        "ny": float(y / max(h, 1)),
        "ox": float((x - cx) / sr),
        "oy": float((y - cy) / sr),
    }


def pith_from_layer_ray(
    layer_start: tuple[float, float],
    layer_end: tuple[float, float],
    ring_frac: float = 0.21,
) -> tuple[float, float]:
    """Deprecated helper: extrapolates inward on the layer ray.

    Do not use for green_start — it lands on orange ink, not the green stroke.
    Kept only for any external callers / notebooks.
    """
    f = float(np.clip(ring_frac, 0.08, 0.35))
    p0x, p0y = layer_start
    p3x, p3y = layer_end
    return ((p0x - f * p3x) / (1.0 - f), (p0y - f * p3y) / (1.0 - f))


def _snap_to_mask(
    mask: np.ndarray,
    pt: tuple[float, float],
) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pt
    d2 = (xs.astype(np.float64) - pt[0]) ** 2 + (ys.astype(np.float64) - pt[1]) ** 2
    i = int(np.argmin(d2))
    return (float(xs[i]), float(ys[i]))


def _layer_inner_on_orange(
    orange_m: np.ndarray,
    pith: tuple[float, float],
    layer_end: tuple[float, float],
    stem_r: float,
    min_radius: float | None = None,
) -> tuple[float, float]:
    """Orange pixel on the outer-ray side at/outside the pith–xylem ring."""
    ys, xs = np.where(orange_m > 0)
    if len(xs) < 20:
        return layer_end
    px, py = pith
    d2 = (xs.astype(np.float64) - px) ** 2 + (ys.astype(np.float64) - py) ** 2
    vx, vy = layer_end[0] - px, layer_end[1] - py
    dots = (xs.astype(np.float64) - px) * vx + (ys.astype(np.float64) - py) * vy
    same = dots > 0
    # Prefer the detected inner-ring radius so起点不落在髓心内
    min_r = float(min_radius) if min_radius is not None else max(36.0, 0.12 * stem_r)
    min_r = max(min_r, 36.0, 0.12 * stem_r)
    min_r2 = min_r * min_r
    if np.any(same):
        cand = d2[same]
        ok = cand >= min_r2
        if np.any(ok):
            # nearest to the ring radius (not nearest to pith)
            target = min_r2
            ii = int(np.argmin(np.where(ok, np.abs(cand - target), 1e18)))
            return (float(xs[same][ii]), float(ys[same][ii]))
        ii = int(np.argmin(cand))
        return (float(xs[same][ii]), float(ys[same][ii]))
    ok = d2 >= min_r2
    if np.any(ok):
        ii = int(np.argmin(np.where(ok, np.abs(d2 - min_r2), 1e18)))
        return (float(xs[ii]), float(ys[ii]))
    ii = int(np.argmin(d2))
    return (float(xs[ii]), float(ys[ii]))


def extract_ink_starts(
    manual: np.ndarray,
    orig: np.ndarray,
) -> dict | None:
    """Ink geometry = parse_manual_geometry (忠实人工墨线).

    Do not re-place layer start with detect_inner_ring_radius — that pushes
    the orange start outward past the ink and shortens xylem / blue / red.
    """
    info = detect_stem_mask(orig)
    if info is None:
        return None
    stem, _ = info
    dist = cv2.distanceTransform(stem, cv2.DIST_L2, 5)
    yx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    stem_c = (float(yx[1]), float(yx[0]))
    stem_r = float(dist.max())
    dark = detect_pith(orig, stem)

    parsed = parse_manual_geometry(manual, orig)
    if parsed is None:
        return None

    g0 = (float(parsed.green_start[0]), float(parsed.green_start[1]))
    g1 = (float(parsed.green_end[0]), float(parsed.green_end[1]))
    p0 = (float(parsed.seg_x0), float(parsed.seg_y0 or parsed.seg_y))
    p1 = (float(parsed.xylem_end), float(parsed.xylem_end_y or parsed.seg_y))
    p2 = (float(parsed.phloem_end), float(parsed.phloem_end_y or parsed.seg_y))
    p3 = (float(parsed.bark_end), float(parsed.bark_end_y or parsed.seg_y))
    h, w = manual.shape[:2]

    return {
        "stem_c": stem_c,
        "stem_r": stem_r,
        "dark_pith": dark,
        "green_start": g0,
        "green_end": g1,
        "layer_start": p0,
        "xylem_end": p1,
        "phloem_end": p2,
        "layer_end": p3,
        "width": w,
        "height": h,
    }


def first_n_manuals(base_dir: Path, n: int | None = 40) -> list[Path]:
    from calibration.io_util import parse_name as pn

    files = list((base_dir / "人工标定" / "第一区域").glob("*.jpg"))

    def key(p: Path) -> tuple[int, int, str]:
        m = pn(p.name)
        if m is None:
            return (10**9, 9, p.name)
        return (m.sample_id, m.view_id, p.name)

    ordered = sorted(files, key=key)
    return ordered if n is None else ordered[:n]


def build_coord_json(base_dir: Path, n: int | None = 40) -> dict:
    orig_dir = base_dir / "原图" / "第一区域"
    files = first_n_manuals(base_dir, n)
    images: dict[str, dict] = {}
    print(f"[坐标] 解析 {len(files)} 张人工图...")
    for i, mp in enumerate(files):
        orig = imread(orig_dir / mp.name)
        manual = imread(mp)
        if orig is None or manual is None:
            print(f"  跳过(读图失败): {mp.name}")
            continue
        raw = extract_ink_starts(manual, orig)
        if raw is None:
            print(f"  跳过(抽线失败): {mp.name}")
            continue
        meta = parse_name(mp.name)
        mag = meta.magnification if meta else 0
        w, h = raw["width"], raw["height"]
        cx, cy = raw["stem_c"]
        sr = raw["stem_r"]
        g0, p0 = raw["green_start"], raw["layer_start"]
        p1 = raw.get("xylem_end") or (
            p0[0] + (raw["layer_end"][0] - p0[0]) * 0.88,
            p0[1] + (raw["layer_end"][1] - p0[1]) * 0.88,
        )
        p2 = raw.get("phloem_end") or (
            p0[0] + (raw["layer_end"][0] - p0[0]) * 0.95,
            p0[1] + (raw["layer_end"][1] - p0[1]) * 0.95,
        )
        start_sep = math.hypot(g0[0] - p0[0], g0[1] - p0[1])
        rec = {
            "filename": mp.name,
            "sample_id": meta.sample_id if meta else None,
            "view_id": meta.view_id if meta else None,
            "magnification": mag,
            "width": w,
            "height": h,
            "stem_center": _pt(cx, cy, w, h, cx, cy, sr),
            "stem_r_px": sr,
            "dark_pith": _pt(raw["dark_pith"][0], raw["dark_pith"][1], w, h, cx, cy, sr),
            "green_start": _pt(g0[0], g0[1], w, h, cx, cy, sr),
            "green_end": _pt(raw["green_end"][0], raw["green_end"][1], w, h, cx, cy, sr),
            "layer_start": _pt(p0[0], p0[1], w, h, cx, cy, sr),
            "xylem_end": _pt(p1[0], p1[1], w, h, cx, cy, sr),
            "phloem_end": _pt(p2[0], p2[1], w, h, cx, cy, sr),
            "layer_end": _pt(raw["layer_end"][0], raw["layer_end"][1], w, h, cx, cy, sr),
            "green_start_to_stem_c_frac": math.hypot(g0[0] - cx, g0[1] - cy) / sr,
            "layer_start_to_stem_c_frac": math.hypot(p0[0] - cx, p0[1] - cy) / sr,
            "layer_start_to_green_start_frac": start_sep / sr,
            "starts_distinct": start_sep > max(28.0, 0.06 * sr),
        }
        g_ang = math.degrees(math.atan2(-(raw["green_end"][1] - g0[1]), raw["green_end"][0] - g0[0]))
        s_ang = math.degrees(math.atan2(-(raw["layer_end"][1] - p0[1]), raw["layer_end"][0] - p0[0]))
        rec["green_angle_deg"] = g_ang
        rec["layer_angle_deg"] = s_ang
        images[mp.name] = rec
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{len(files)}: {mp.name}")
    summary = summarize(images)
    payload = {"coord_system": COORD_META, "n": len(images), "images": images, "summary": summary}
    return payload


def summarize(images: dict[str, dict]) -> dict:
    by_mag: dict[int, list[dict]] = {}
    for rec in images.values():
        by_mag.setdefault(int(rec["magnification"]), []).append(rec)

    def stats(vals: list[float]) -> dict:
        arr = np.array(vals, dtype=np.float64)
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None,
            "p25": float(np.percentile(arr, 25)) if len(arr) else None,
            "p75": float(np.percentile(arr, 75)) if len(arr) else None,
        }

    out: dict = {}
    for mag, rows in sorted(by_mag.items()):
        out[str(mag)] = {
            "count": len(rows),
            "green_start_to_stem_c_frac": stats([r["green_start_to_stem_c_frac"] for r in rows]),
            "layer_start_to_stem_c_frac": stats([r["layer_start_to_stem_c_frac"] for r in rows]),
            "layer_minus_green_frac": stats(
                [r["layer_start_to_stem_c_frac"] - r["green_start_to_stem_c_frac"] for r in rows]
            ),
            "starts_distinct_rate": float(np.mean([1.0 if r["starts_distinct"] else 0.0 for r in rows])),
            "green_start_nx": stats([r["green_start"]["nx"] for r in rows]),
            "green_start_ny": stats([r["green_start"]["ny"] for r in rows]),
            "stem_r_px": stats([r["stem_r_px"] for r in rows]),
        }
    all_rows = list(images.values())
    out["all"] = {
        "count": len(all_rows),
        "green_start_to_stem_c_frac": stats([r["green_start_to_stem_c_frac"] for r in all_rows]),
        "layer_start_to_stem_c_frac": stats([r["layer_start_to_stem_c_frac"] for r in all_rows]),
        "layer_minus_green_frac": stats(
            [r["layer_start_to_stem_c_frac"] - r["green_start_to_stem_c_frac"] for r in all_rows]
        ),
        "starts_distinct_rate": float(np.mean([1.0 if r["starts_distinct"] else 0.0 for r in all_rows])),
    }
    return out


def save_coord_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_coord_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mag_priors(summary: dict) -> dict[int, dict]:
    """Per-magnification start radii (fraction of stem_r) learned from manuals."""
    priors: dict[int, dict] = {}
    for key, block in summary.items():
        if key == "all":
            continue
        mag = int(key)
        g = block["green_start_to_stem_c_frac"]["median"] or 0.04
        ring = block["layer_start_to_stem_c_frac"]["median"] or 0.18
        # layer start must sit outside green start
        ring = max(ring, g + 0.08)
        priors[mag] = {
            "green_start_frac": float(np.clip(g, 0.0, 0.12)),
            "layer_start_frac": float(np.clip(ring, 0.10, 0.32)),
        }
    if not priors:
        priors = {2: {"green_start_frac": 0.04, "layer_start_frac": 0.18},
                  4: {"green_start_frac": 0.04, "layer_start_frac": 0.20}}
    return priors


def coords_to_sixpoint_records(base_dir: Path, payload: dict) -> dict[str, dict]:
    """Write YOLO-ready 6-point records from ink JSON (orange/blue/red breaks)."""
    from calibration.region1_sixpoint import (
        faithful_from_parse,
        geom_to_record,
        geom_from_six_points,
    )

    json_dir = base_dir / "推理结果" / "coords" / "第一区域"
    json_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    images = payload.get("images", {})
    print(f"[坐标] 把 {len(images)} 份 JSON 收成六点训练标签（忠实墨线）...")
    for i, (name, rec) in enumerate(images.items()):
        mag = int(rec.get("magnification") or 2)
        stem = rec.get("stem_center") or rec["green_start"]
        g0 = (float(stem["x"]), float(stem["y"]))
        g1 = (float(rec["green_end"]["x"]), float(rec["green_end"]["y"]))
        p0 = (float(rec["layer_start"]["x"]), float(rec["layer_start"]["y"]))
        p3 = (float(rec["layer_end"]["x"]), float(rec["layer_end"]["y"]))
        if "xylem_end" in rec and "phloem_end" in rec:
            p1 = (float(rec["xylem_end"]["x"]), float(rec["xylem_end"]["y"]))
            p2 = (float(rec["phloem_end"]["x"]), float(rec["phloem_end"]["y"]))
        else:
            p1 = (p0[0] + (p3[0] - p0[0]) * 0.88, p0[1] + (p3[1] - p0[1]) * 0.88)
            p2 = (p0[0] + (p3[0] - p0[0]) * 0.95, p0[1] + (p3[1] - p0[1]) * 0.95)
        geom = geom_from_six_points(g0, g1, p0, p1, p2, p3, green_start=g0)
        if geom is None:
            continue
        geom = faithful_from_parse(geom, float(rec["stem_r_px"]))
        six = geom_to_record(geom)
        six["filename"] = name
        six["magnification"] = mag
        records[name] = six
        one = {
            "coord_system": COORD_META,
            "filename": name,
            "magnification": mag,
            "width": rec["width"],
            "height": rec["height"],
            "stem_r_px": rec["stem_r_px"],
            "stem_center": rec.get("stem_center"),
            "manual": {
                "green_start": rec["green_start"],
                "layer_start": rec["layer_start"],
                "green_end": rec["green_end"],
                "xylem_end": rec.get("xylem_end"),
                "phloem_end": rec.get("phloem_end"),
                "layer_end": rec["layer_end"],
            },
            "train": six,
        }
        (json_dir / f"{Path(name).stem}.json").write_text(
            json.dumps(one, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i + 1}/{len(images)}: {name}")
    return records


def relabel_all_to_json(base_dir: Path) -> dict[str, dict]:
    """Parse every manual, save coords + six-point JSON for training."""
    payload = build_coord_json(base_dir, n=None)
    save_coord_json(payload, base_dir / "推理结果" / "region1_coords.json")
    records = coords_to_sixpoint_records(base_dir, payload)
    six_path = base_dir / "推理结果" / "region1_sixpoint.json"
    six_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[坐标] 六点标签 {len(records)} -> {six_path}")
    if payload.get("summary"):
        s = payload["summary"]
        for mag in ("2", "4", "all"):
            if mag not in s:
                continue
            b = s[mag]
            print(
                f"        {mag}X n={b['count']}  "
                f"两起点分离={b['starts_distinct_rate']:.0%}  "
                f"内环-髓心中位={b['layer_minus_green_frac']['median']:.3f}"
            )
    return records
