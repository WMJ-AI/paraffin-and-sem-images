"""Diagnose LabelMe vs 人工标定 alignment."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import parse_manual_geometry
from calibration.io_util import imread
from calibration.region1_coords import first_n_manuals

BASE = Path(__file__).resolve().parents[2]
LM_PATH = BASE / "推理结果" / "labelme_first30_raw.txt"


def split_json(text: str) -> list[dict]:
    objs, buf, depth = [], "", 0
    for ch in text:
        if ch == "{":
            depth += 1
        if depth > 0:
            buf += ch
        if ch == "}":
            depth -= 1
            if depth == 0 and buf.strip():
                objs.append(json.loads(buf))
                buf = ""
    return objs


def lm_points(lm: dict) -> dict[str, tuple[float, float, float, float]]:
    out = {}
    for s in lm["shapes"]:
        p0, p1 = s["points"]
        out[s["label"]] = (p0[0], p0[1], p1[0], p1[1])
    return out


def ang(x0, y0, x1, y1) -> float:
    return math.degrees(math.atan2(-(y1 - y0), x1 - x0))


def score_pair(lm: dict, man_path: Path) -> float:
    """Higher = better match between labelme lines and manual parse."""
    man = imread(man_path)
    orig = imread(BASE / "原图" / "第一区域" / man_path.name)
    if man is None or orig is None:
        return -1e9
    h, w = man.shape[:2]
    sx, sy = w / lm["imageWidth"], h / lm["imageHeight"]
    p = parse_manual_geometry(man, orig)
    if p is None:
        return -1e9
    pts = lm_points(lm)
    if "半径" not in pts:
        return -1e9
    g0 = (pts["半径"][0] * sx, pts["半径"][1] * sy)
    g1 = (pts["半径"][2] * sx, pts["半径"][3] * sy)
    ga_lm = ang(g0[0], g0[1], g1[0], g1[1])
    ga_p = p.green_angle_deg
    # layer direction from 木质部
    if "木质部" in pts:
        la_lm = ang(pts["木质部"][0] * sx, pts["木质部"][1] * sy,
                    pts["木质部"][2] * sx, pts["木质部"][3] * sy)
        la_p = p.seg_angle_deg
    else:
        la_lm, la_p = 0, 0
    def ad(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)
    # green start should be near parse pith
    d0 = math.hypot(g0[0] - p.green_start[0], g0[1] - p.green_start[1])
    d0n = d0 / max(w, h)
    ang_pen = ad(ga_lm, ga_p) + ad(la_lm, la_p)
    return -d0n * 500 - ang_pen


def main() -> None:
    objs = split_json(LM_PATH.read_text(encoding="utf-8"))
    manuals = first_n_manuals(BASE, 30)

    print("=== assumed order (img01 -> manuals[0]) ===")
    for i in range(3):
        lm, mp = objs[i], manuals[i]
        man = imread(mp)
        orig = imread(BASE / "原图" / "第一区域" / mp.name)
        h, w = man.shape[:2]
        sx, sy = w / lm["imageWidth"], h / lm["imageHeight"]
        p = parse_manual_geometry(man, orig)
        pts = lm_points(lm)
        g0 = (pts["半径"][0] * sx, pts["半径"][1] * sy)
        g1 = (pts["半径"][2] * sx, pts["半径"][3] * sy)
        print(f"{lm['imagePath']} -> {mp.name}")
        print(f"  lm green ang={ang(g0[0],g0[1],g1[0],g1[1]):.1f}  parse={p.green_angle_deg:.1f}")
        print(f"  lm layer ang={ang(pts['木质部'][0]*sx,pts['木质部'][1]*sy,pts['木质部'][2]*sx,pts['木质部'][3]*sy):.1f}  parse={p.seg_angle_deg:.1f}")
        print(f"  g0 dist lm->parse={math.hypot(g0[0]-p.green_start[0],g0[1]-p.green_start[1]):.0f}px")
        print(f"  parse g0={p.green_start}  lm g0={g0}")

    print("\n=== best manual match per labelme (first 5) ===")
    for lm in objs[:5]:
        scores = [(score_pair(lm, mp), mp.name) for mp in manuals]
        scores.sort(reverse=True)
        print(f"{lm['imagePath']}: best={scores[0][1]} ({scores[0][0]:.1f})  assumed={manuals[objs.index(lm)].name} ({score_pair(lm, manuals[objs.index(lm)]):.1f})")

    # Test: labelme coords on manual resized to 1432x960 (no full-res scale)
    print("\n=== coords on manual resized to labelme canvas ===")
    for i in range(3):
        lm, mp = objs[i], manuals[i]
        man = imread(mp)
        small = cv2.resize(man, (lm["imageWidth"], lm["imageHeight"]))
        pts = lm_points(lm)
        g0 = pts["半径"][:2]
        # draw check - distance from labelme point to nearest saturated ink on small manual
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        ink = (s > 80) & (v > 70)
        ys, xs = np.where(ink)
        x0, y0 = g0
        if len(xs):
            d = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2).min()
            print(f"{lm['imagePath']} -> {mp.name}: lm g0 to ink on resized manual: {d:.1f}px")


if __name__ == "__main__":
    main()
