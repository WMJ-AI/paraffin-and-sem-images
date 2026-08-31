"""Rebuild first-30 coords from 人工墨线 (parse_manual_geometry), not LabelMe."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import draw_calibration_lines
from calibration.io_util import imread, imwrite
from calibration.merge import stitch_horizontal
from calibration.region1_coords import (
    build_coord_json,
    coords_to_sixpoint_records,
    first_n_manuals,
    save_coord_json,
)
from calibration.region1_sixpoint import geom_from_record, geom_from_six_points
from calibration.stem import detect_stem_mask

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "推理结果" / "coords" / "第一区域"
QC_DIR = BASE / "推理结果" / "dbg_score" / "labelme30_qc"
N = 30


def geom_from_manual_json(data: dict):
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


def main() -> None:
    print(f"[coords] rebuild first {N} from 人工墨线...")
    payload = build_coord_json(BASE, N)
    save_coord_json(payload, BASE / "推理结果" / f"region1_coords_first{N}.json")
    records = coords_to_sixpoint_records(BASE, payload)

    six_path = BASE / "推理结果" / "region1_sixpoint.json"
    six_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[coords] {len(records)} -> {six_path}")

    # archive bad labelme import
    lm_backup = BASE / "推理结果" / "region1_sixpoint_labelme30_misaligned.json"
    old = BASE / "推理结果" / "region1_sixpoint_labelme30.json"
    if old.exists() and not lm_backup.exists():
        shutil.copy2(old, lm_backup)

    auto_dir = BASE / "自动标定" / "第一区域"
    merge_dir = BASE / "合并标定" / "第一区域"
    orig_dir = BASE / "原图" / "第一区域"
    manual_dir = BASE / "人工标定" / "第一区域"
    QC_DIR.mkdir(parents=True, exist_ok=True)

    for i, mp in enumerate(first_n_manuals(BASE, N), 1):
        name = mp.name
        stem = Path(name).stem
        data = json.loads((OUT_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        orig = imread(orig_dir / name)
        man = imread(manual_dir / name)
        info = detect_stem_mask(orig)
        geom = geom_from_manual_json(data)
        auto = draw_calibration_lines(orig, geom, info[0] if info else None, numbered=True)
        imwrite(auto_dir / name, auto)
        if man is not None:
            cmp = stitch_horizontal(man, auto)
            scale = min(1.0, 2400 / cmp.shape[1])
            if scale < 1:
                cmp = cv2.resize(cmp, (int(cmp.shape[1] * scale), int(cmp.shape[0] * scale)))
            imwrite(QC_DIR / f"{i:02d}_{stem}_compare.jpg", cmp)
        print(f"  {i:02d}/{N} {name}")

    print(f"per-image JSON: {OUT_DIR}")
    print(f"QC: {QC_DIR}")


if __name__ == "__main__":
    main()
