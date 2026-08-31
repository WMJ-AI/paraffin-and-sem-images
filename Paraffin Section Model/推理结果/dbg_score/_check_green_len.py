"""Compare JSON green length vs parse / ink."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import _best_stroke_mask, _color_layers, parse_manual_geometry
from calibration.io_util import imread
from calibration.region1_coords import first_n_manuals
from calibration.stem import detect_stem_mask

base = Path(__file__).resolve().parents[2]
files = first_n_manuals(base, 10)
json_dir = base / "推理结果" / "coords" / "第一区域"

print(
    f"{'name':20s} {'json':>8} {'parse':>8} {'ink':>8} "
    f"{'g0->pith':>9} {'dt_g1':>7} {'dt_pg1':>7}"
)
for mp in files:
    name = mp.name
    j = json.loads((json_dir / f"{Path(name).stem}.json").read_text(encoding="utf-8"))
    m = j["manual"]
    g0 = (m["green_start"]["x"], m["green_start"]["y"])
    g1 = (m["green_end"]["x"], m["green_end"]["y"])
    pith = (j["stem_center"]["x"], j["stem_center"]["y"])
    json_len = math.hypot(g1[0] - g0[0], g1[1] - g0[1])
    g0_pith = math.hypot(g0[0] - pith[0], g0[1] - pith[1])

    orig = imread(base / "原图" / "第一区域" / name)
    man = imread(base / "人工标定" / "第一区域" / name)
    parse = parse_manual_geometry(man, orig)
    parse_len = math.hypot(
        parse.green_end[0] - parse.green_start[0],
        parse.green_end[1] - parse.green_start[1],
    )

    info = detect_stem_mask(orig)
    stem, _ = info
    dist = cv2.distanceTransform(stem, cv2.DIST_L2, 5)
    green, _, _, _ = _color_layers(man)
    stem_d = cv2.dilate(stem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), 1)
    green = cv2.bitwise_and(green, stem_d)
    gm = _best_stroke_mask(green, dist, pith)
    ys, xs = np.where(gm > 0)
    d2 = (xs.astype(float) - pith[0]) ** 2 + (ys.astype(float) - pith[1]) ** 2
    i0 = int(np.argmin(d2))
    i1 = int(np.argmax(d2))
    ink0 = (float(xs[i0]), float(ys[i0]))
    ink1 = (float(xs[i1]), float(ys[i1]))
    ink_len = math.hypot(ink1[0] - ink0[0], ink1[1] - ink0[1])

    def dt_at(pt):
        yi, xi = int(round(pt[1])), int(round(pt[0]))
        return float(dist[yi, xi])

    print(
        f"{name:20s} {json_len:8.1f} {parse_len:8.1f} {ink_len:8.1f} "
        f"{g0_pith:9.1f} {dt_at(g1):7.1f} {dt_at(parse.green_end):7.1f}"
    )
