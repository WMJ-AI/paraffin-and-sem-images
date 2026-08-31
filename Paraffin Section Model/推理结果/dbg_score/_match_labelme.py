"""Match labelme img01..30 to manual images by line angles."""
from __future__ import annotations

import json
import math
from pathlib import Path

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


def ang(a: list[float], b: list[float]) -> float:
    return math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))


def ad(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def main() -> None:
    objs = split_json(LM_PATH.read_text(encoding="utf-8"))
    manuals = first_n_manuals(BASE, 30)
    parsed: dict[str, object] = {}
    for mp in manuals:
        man = imread(BASE / "人工标定" / "第一区域" / mp.name)
        orig = imread(BASE / "原图" / "第一区域" / mp.name)
        parsed[mp.name] = parse_manual_geometry(man, orig)

    print("labelme -> best_match (angle) | assumed_order")
    mismatches = 0
    for i, lm in enumerate(objs):
        pts = {s["label"]: s["points"] for s in lm["shapes"]}
        ga = ang(pts["半径"][0], pts["半径"][1])
        la = ang(pts["木质部"][0], pts["木质部"][1])
        best_name = None
        best_s = -1e9
        for name, p in parsed.items():
            s = -(ad(ga, p.green_angle_deg) + ad(la, p.seg_angle_deg))
            if s > best_s:
                best_s = s
                best_name = name
        assumed = manuals[i].name
        flag = "OK" if best_name == assumed else "MISMATCH"
        if flag == "MISMATCH":
            mismatches += 1
        print(f"  {lm['imagePath']:10s} -> {best_name:16s} ({best_s:6.1f})  assumed={assumed:16s} {flag}")
    print(f"mismatches {mismatches}/{len(objs)}")


if __name__ == "__main__":
    main()
