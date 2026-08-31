"""Redraw first 10 compares from JSON (full green/orange/blue/red)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import draw_calibration_lines
from calibration.io_util import imread, imwrite
from calibration.region1_coords import first_n_manuals
from calibration.region1_sixpoint import geom_from_record, geom_from_six_points

base = Path(__file__).resolve().parents[2]
json_dir = base / "推理结果" / "coords" / "第一区域"
out_dir = base / "推理结果" / "dbg_score" / "redraw_first10"
out_dir.mkdir(parents=True, exist_ok=True)
files = first_n_manuals(base, 10)


def stitch(a, b):
    h = max(a.shape[0], b.shape[0])

    def pad(im):
        if im.shape[0] == h:
            return im
        out = np.zeros((h, im.shape[1], 3), np.uint8)
        out[: im.shape[0]] = im
        return out

    return np.hstack([pad(a), pad(b)])


def geom_from_manual_json(data):
    m = data["manual"]
    g0 = (m["green_start"]["x"], m["green_start"]["y"])
    g1 = (m["green_end"]["x"], m["green_end"]["y"])
    p0 = (m["layer_start"]["x"], m["layer_start"]["y"])
    p3 = (m["layer_end"]["x"], m["layer_end"]["y"])
    if m.get("xylem_end") and m.get("phloem_end"):
        p1 = (m["xylem_end"]["x"], m["xylem_end"]["y"])
        p2 = (m["phloem_end"]["x"], m["phloem_end"]["y"])
    else:
        p1 = (p0[0] + (p3[0] - p0[0]) * 0.88, p0[1] + (p3[1] - p0[1]) * 0.88)
        p2 = (p0[0] + (p3[0] - p0[0]) * 0.95, p0[1] + (p3[1] - p0[1]) * 0.95)
    return geom_from_six_points(g0, g1, p0, p1, p2, p3, green_start=g0)


for i, mp in enumerate(files, 1):
    name = mp.name
    stem = Path(name).stem
    data = json.loads((json_dir / f"{stem}.json").read_text(encoding="utf-8"))
    orig = imread(base / "原图" / "第一区域" / name)
    man = imread(base / "人工标定" / "第一区域" / name)
    auto_m = draw_calibration_lines(orig, geom_from_manual_json(data), numbered=True)
    auto_t = draw_calibration_lines(orig, geom_from_record(data["train"]), numbered=True)
    imwrite(out_dir / f"{i:02d}_{stem}_manualJSON.jpg", auto_m)
    imwrite(out_dir / f"{i:02d}_{stem}_trainJSON.jpg", auto_t)
    cmp = stitch(man, auto_m)
    scale = min(1.0, 2400 / cmp.shape[1])
    if scale < 1:
        cmp = cv2.resize(cmp, (int(cmp.shape[1] * scale), int(cmp.shape[0] * scale)))
    imwrite(out_dir / f"{i:02d}_{stem}_compare.jpg", cmp)
    print(f"{i:02d} {name} redrawn")

compares = sorted(out_dir.glob("*_compare.jpg"))
imgs = [imread(p) for p in compares]
tw = 1200
rows, row = [], []
for im in imgs:
    h, w = im.shape[:2]
    im2 = cv2.resize(im, (tw, int(h * tw / w)))
    row.append(im2)
    if len(row) == 2:
        mh = max(x.shape[0] for x in row)

        def padh(x):
            if x.shape[0] == mh:
                return x
            o = np.zeros((mh, x.shape[1], 3), np.uint8)
            o[: x.shape[0]] = x
            return o

        rows.append(np.hstack([padh(row[0]), padh(row[1])]))
        row = []
if row:
    rows.append(np.hstack([row[0], np.zeros_like(row[0])]))
imwrite(out_dir / "_all_compare_grid.jpg", np.vstack(rows))
print("grid updated")
