"""Convert concatenated LabelMe JSON (img01..img30 @ 1432x960) to
full-res six-point training labels for the first 30 人工标定 images.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from calibration.io_util import imread, parse_name
from calibration.region1_coords import COORD_META, first_n_manuals, _pt
from calibration.region1_sixpoint import geom_from_six_points, geom_to_record

BASE = Path(__file__).resolve().parents[2]
LABELME_TXT = BASE / "推理结果" / "新建 文本文档.txt"
OUT_DIR = BASE / "推理结果" / "coords" / "第一区域"
SIX_PATH = BASE / "推理结果" / "region1_sixpoint_labelme30.json"
INDEX_PATH = BASE / "推理结果" / "region1_labelme30_index.json"

LM_W, LM_H = 1432, 960
LABEL_GREEN = "半径"
LABEL_XYLEM = "木质部"
LABEL_PHLOEM = "韧皮部"
LABEL_BARK = "树皮皮层"


def split_concat_json(text: str) -> list[dict]:
    objs: list[dict] = []
    buf = ""
    depth = 0
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


def shapes_by_label(shapes: list[dict]) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for s in shapes:
        out[s["label"]] = s["points"]
    return out


def scale_pt(pt: list[float], sx: float, sy: float) -> tuple[float, float]:
    return (float(pt[0]) * sx, float(pt[1]) * sy)


def main() -> None:
    manuals = first_n_manuals(BASE, 30)
    if len(manuals) != 30:
        raise SystemExit(f"expected 30 manuals, got {len(manuals)}")

    text = LABELME_TXT.read_text(encoding="utf-8")
    objs = split_concat_json(text)
    if len(objs) != 30:
        raise SystemExit(f"expected 30 labelme objects, got {len(objs)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.json"):
        old.unlink()

    records: dict[str, dict] = {}
    index: list[dict] = []

    for i, (mp, lm) in enumerate(zip(manuals, objs), 1):
        name = mp.name
        meta = parse_name(name)
        orig = imread(BASE / "原图" / "第一区域" / name)
        if orig is None:
            raise SystemExit(f"missing orig: {name}")
        h, w = orig.shape[:2]
        sx, sy = w / float(lm["imageWidth"]), h / float(lm["imageHeight"])

        by = shapes_by_label(lm["shapes"])
        for need in (LABEL_GREEN, LABEL_XYLEM, LABEL_PHLOEM, LABEL_BARK):
            if need not in by:
                raise SystemExit(f"{lm['imagePath']} missing label {need}")

        g0 = scale_pt(by[LABEL_GREEN][0], sx, sy)
        g1 = scale_pt(by[LABEL_GREEN][1], sx, sy)
        p0 = scale_pt(by[LABEL_XYLEM][0], sx, sy)
        # joints: 木质部 end ≈ 韧皮部 start; 韧皮部 end ≈ 树皮 start
        p1_a = scale_pt(by[LABEL_XYLEM][1], sx, sy)
        p1_b = scale_pt(by[LABEL_PHLOEM][0], sx, sy)
        p2_a = scale_pt(by[LABEL_PHLOEM][1], sx, sy)
        p2_b = scale_pt(by[LABEL_BARK][0], sx, sy)
        p3 = scale_pt(by[LABEL_BARK][1], sx, sy)
        p1 = ((p1_a[0] + p1_b[0]) / 2.0, (p1_a[1] + p1_b[1]) / 2.0)
        p2 = ((p2_a[0] + p2_b[0]) / 2.0, (p2_a[1] + p2_b[1]) / 2.0)

        from calibration.stem import detect_stem_mask

        info = detect_stem_mask(orig)
        if info is not None:
            import cv2

            dist = cv2.distanceTransform(info[0], cv2.DIST_L2, 5)
            sr = float(dist.max()) or 1.0
            yx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            stem_c = (float(yx[1]), float(yx[0]))
        else:
            sr = math.hypot(g1[0] - g0[0], g1[1] - g0[1]) or 1.0
            stem_c = g0
        cx, cy = g0  # ox/oy frame relative to labeled pith (green start)

        geom = geom_from_six_points(g0, g1, p0, p1, p2, p3, green_start=g0)
        six = geom_to_record(geom)
        mag = meta.magnification if meta else 0
        six["filename"] = name
        six["magnification"] = mag
        six["source"] = "labelme"
        six["labelme_image"] = lm["imagePath"]
        records[name] = six

        one = {
            "coord_system": COORD_META,
            "filename": name,
            "magnification": mag,
            "width": w,
            "height": h,
            "stem_r_px": sr,
            "source": "labelme",
            "labelme": {
                "imagePath": lm["imagePath"],
                "imageWidth": lm["imageWidth"],
                "imageHeight": lm["imageHeight"],
                "scale_x": sx,
                "scale_y": sy,
            },
            "manual": {
                "green_start": _pt(g0[0], g0[1], w, h, cx, cy, sr),
                "layer_start": _pt(p0[0], p0[1], w, h, cx, cy, sr),
                "green_end": _pt(g1[0], g1[1], w, h, cx, cy, sr),
                "xylem_end": _pt(p1[0], p1[1], w, h, cx, cy, sr),
                "phloem_end": _pt(p2[0], p2[1], w, h, cx, cy, sr),
                "layer_end": _pt(p3[0], p3[1], w, h, cx, cy, sr),
            },
            "train": six,
        }
        (OUT_DIR / f"{Path(name).stem}.json").write_text(
            json.dumps(one, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        index.append(
            {
                "i": i,
                "labelme": lm["imagePath"],
                "filename": name,
                "magnification": mag,
                "width": w,
                "height": h,
            }
        )
        print(f"{i:02d}/{len(manuals)} {lm['imagePath']} -> {name}")

    SIX_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    INDEX_PATH.write_text(
        json.dumps(
            {
                "n": len(records),
                "source_file": str(LABELME_TXT.name),
                "labelme_size": [LM_W, LM_H],
                "mapping": index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"per-image JSON: {OUT_DIR} ({len(records)})")
    print(f"train sixpoint: {SIX_PATH}")
    print(f"index: {INDEX_PATH}")


if __name__ == "__main__":
    main()
