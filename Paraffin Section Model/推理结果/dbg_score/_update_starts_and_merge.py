"""Rebuild coords + update 自动/合并 for first N from JSON."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import draw_calibration_lines
from calibration.io_util import imread, imwrite
from calibration.merge import stitch_horizontal
from calibration.region1_coords import (
    build_coord_json,
    first_n_manuals,
    relabel_all_to_json,
    save_coord_json,
)
from calibration.region1_sixpoint import geom_from_record, geom_from_six_points
from calibration.stem import detect_stem_mask

base = Path(__file__).resolve().parents[2]


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


print("rebuild coords...")
p40 = build_coord_json(base, 40)
save_coord_json(p40, base / "推理结果" / "region1_coords_first40.json")
recs = relabel_all_to_json(base)
print("records", len(recs))

json_dir = base / "推理结果" / "coords" / "第一区域"
auto_dir = base / "自动标定" / "第一区域"
merge_dir = base / "合并标定" / "第一区域"
orig_dir = base / "原图" / "第一区域"
manual_dir = base / "人工标定" / "第一区域"
auto_dir.mkdir(parents=True, exist_ok=True)
merge_dir.mkdir(parents=True, exist_ok=True)
out_dir = base / "推理结果" / "dbg_score" / "redraw_first10"
out_dir.mkdir(parents=True, exist_ok=True)

files = first_n_manuals(base, 10)
for i, mp in enumerate(files, 1):
    name = mp.name
    stem = Path(name).stem
    data = json.loads((json_dir / f"{stem}.json").read_text(encoding="utf-8"))
    orig = imread(orig_dir / name)
    man = imread(manual_dir / name)
    info = detect_stem_mask(orig)
    stem_mask = info[0] if info else None
    geom = geom_from_manual_json(data)
    auto = draw_calibration_lines(orig, geom, stem_mask, numbered=True)
    imwrite(auto_dir / name, auto)
    if man is not None:
        imwrite(merge_dir / name, stitch_horizontal(man, auto))
    # also refresh dbg compare
    cmp = stitch_horizontal(man, auto) if man is not None else auto
    scale = min(1.0, 2400 / cmp.shape[1])
    if scale < 1:
        cmp = cv2.resize(cmp, (int(cmp.shape[1] * scale), int(cmp.shape[0] * scale)))
    imwrite(out_dir / f"{i:02d}_{stem}_compare.jpg", cmp)
    m = data["manual"]
    g0 = (m["green_start"]["x"], m["green_start"]["y"])
    p0 = (m["layer_start"]["x"], m["layer_start"]["y"])
    sr = float(data["stem_r_px"])
    sep = ((p0[0] - g0[0]) ** 2 + (p0[1] - g0[1]) ** 2) ** 0.5 / sr
    print(f"{i:02d} {name}: layer_sep={sep:.3f} ring_frac_train={data['train'].get('ring_frac')}")

print("updated 自动/合并 + redraw_first10")
print("1-2 JSON layer_start:", json.loads((json_dir / "1-2 4X.json").read_text(encoding="utf-8"))["manual"]["layer_start"])
