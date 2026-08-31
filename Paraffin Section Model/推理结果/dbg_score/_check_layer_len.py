"""Compare JSON layer outer vs parse bark (blue/red short cause)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from calibration.geometry import (
    _color_layers,
    _layer_runs_along_segment,
    parse_manual_geometry,
)
from calibration.io_util import imread
from calibration.region1_coords import first_n_manuals

base = Path(__file__).resolve().parents[2]
json_dir = base / "推理结果" / "coords" / "第一区域"

print(f"{'name':18s} {'json_L':>8} {'parse_L':>8} {'delta':>8} {'runs_L':>8}")
for mp in first_n_manuals(base, 10):
    name = mp.name
    j = json.loads((json_dir / f"{Path(name).stem}.json").read_text(encoding="utf-8"))
    m = j["manual"]
    p0 = (m["layer_start"]["x"], m["layer_start"]["y"])
    p3 = (m["layer_end"]["x"], m["layer_end"]["y"])
    jl = math.hypot(p3[0] - p0[0], p3[1] - p0[1])

    man = imread(base / "人工标定" / "第一区域" / name)
    orig = imread(base / "原图" / "第一区域" / name)
    p = parse_manual_geometry(man, orig)
    pp0 = (p.seg_x0, p.seg_y0)
    pb = (p.bark_end, p.bark_end_y)
    pl = math.hypot(pb[0] - pp0[0], pb[1] - pp0[1])

    g, o, b, r = _color_layers(man)
    runs = _layer_runs_along_segment(man, p0, p3, extra=1.35, orange=o, blue=b, red=r)
    rl = 0.0
    if runs:
        pe = runs["p3"]
        rl = math.hypot(pe[0] - p0[0], pe[1] - p0[1])
    print(f"{name:18s} {jl:8.1f} {pl:8.1f} {pl - jl:8.1f} {rl:8.1f}")
    print(
        f"  parse p1-p2-p3 lens from p0: "
        f"{math.hypot(p.xylem_end-pp0[0], p.xylem_end_y-pp0[1]):.0f} / "
        f"{math.hypot(p.phloem_end-pp0[0], p.phloem_end_y-pp0[1]):.0f} / "
        f"{pl:.0f}"
    )
