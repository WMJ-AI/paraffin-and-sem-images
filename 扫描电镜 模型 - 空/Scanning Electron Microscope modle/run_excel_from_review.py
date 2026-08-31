# -*- coding: utf-8 -*-
"""
从复核对照图右侧标注解析绿腔 / 橙框，按复核结果重算指标并导出 Excel。

用法:
  python run_excel_from_review.py --review 复核_1-40
  python run_excel_from_review.py --review 复核_1-40 --lo 1 --hi 40
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
_pylibs = ROOT / ".pylibs"
if _pylibs.is_dir():
    sys.path.insert(0, str(_pylibs))

from export_from_review_pngs import (  # noqa: E402
    _lumen_contours_from_green,
    _orange_box_contours,
    _right_body,
)
from learn_from_review import (  # noqa: E402
    OrientedBox,
    _blue_mask,
    _imread_unicode,
    _lines_from_yellow,
    apply_human_tvw_guides,
    image_key_from_stem,
    pair_from_two_largest_in_boxes,
)
from run_batch_excel63 import write_workbook  # noqa: E402
from run_from_human import _find_tif, natural_key_name  # noqa: E402
from scale_bar import detect_scale_um_per_px, load_rgb_gray  # noqa: E402
from segment import Vessel, attach_metrics, tissue_height  # noqa: E402

OUT_DIR = ROOT / "output" / "excel_samples"


def _resolve_review_dir(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p.resolve()
    cand = ROOT / "output" / name
    if cand.is_dir():
        return cand
    outp = ROOT / "output"
    hit = next(
        (d for d in outp.iterdir() if d.is_dir() and "复核" in d.name and "1-40" in d.name),
        None,
    )
    if hit is not None and ("1-40" in name or name == "复核_1-40"):
        return hit
    raise SystemExit(f"找不到复核目录: {name}")


def _sample_id(key: str) -> int | None:
    m = re.match(r"(\d+)\s*\(", key)
    return int(m.group(1)) if m else None


def _prefix(image_name: str) -> str:
    m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
    if m:
        return f"{m.group(1)}({m.group(2)})"
    return Path(image_name).stem


def _scale_contour(c: np.ndarray, sx: float, sy: float) -> np.ndarray:
    cc = c.astype(np.float64).copy()
    cc[:, 0, 0] *= sx
    cc[:, 0, 1] *= sy
    return cc.astype(np.int32)


def _mask_scale_bar_region(body_bgr: np.ndarray) -> np.ndarray:
    """复制右侧面板，抹掉左下角绿色标尺，避免当成导管腔。"""
    out = body_bgr.copy()
    h, w = out.shape[:2]
    y0, y1 = int(h * 0.82), h
    x0, x1 = 0, int(w * 0.45)
    out[y0:y1, x0:x1] = 0
    return out


def _is_scale_bar_like(cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int) -> bool:
    """标尺多为扁长条，落在左下角。"""
    if cy < img_h * 0.78:
        return False
    if cx > img_w * 0.50:
        return False
    if bw < 40:
        return False
    return (bw / max(bh, 1.0)) >= 3.5 or bh <= 12


def _vessels_from_green(
    lumens: list[np.ndarray],
    image_name: str,
    um: float,
    sx: float,
    sy: float,
    img_w: int,
    img_h: int,
) -> list[Vessel]:
    vessels: list[Vessel] = []
    prefix = _prefix(image_name)
    idx = 0
    for c in lumens:
        cc = _scale_contour(c, sx, sy)
        area = float(cv2.contourArea(cc))
        if area < 50:
            continue
        peri = float(cv2.arcLength(cc, True))
        m = cv2.moments(cc)
        if m["m00"] < 1:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        x, y, bw, bh = cv2.boundingRect(cc)
        if _is_scale_bar_like(cx, cy, float(bw), float(bh), img_w, img_h):
            continue
        idx += 1
        vessels.append(
            Vessel(
                vessel_id=f"{prefix}-V{idx:02d}",
                image_name=image_name,
                contour=cc,
                area_px=area,
                peri_px=peri,
                cx=cx,
                cy=cy,
                bbox=(int(x), int(y), int(bw), int(bh)),
                status="primary",
                reason="review_png_green",
            )
        )
    return attach_metrics(vessels, um)


def _tvw_guide_lines(body_bgr: np.ndarray, sx: float, sy: float):
    mask = _blue_mask(body_bgr)
    # 去掉橙框/绿腔残留
    hsv = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (40, 70, 70), (95, 255, 255))
    rgb = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    orange = (
        (rgb[:, :, 0] > 180)
        & (rgb[:, :, 1] > 60)
        & (rgb[:, :, 1] < 160)
        & (rgb[:, :, 2] < 100)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(green))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(orange))
    lines = _lines_from_yellow(mask)
    scaled = []
    for p0, p1 in lines:
        a = np.asarray(p0, dtype=float) * np.array([sx, sy], dtype=float)
        b = np.asarray(p1, dtype=float) * np.array([sx, sy], dtype=float)
        scaled.append((a, b))
    return scaled


def process_review_png(png: Path) -> tuple[list[Vessel], list, dict]:
    key = image_key_from_stem(png.stem)
    tif = _find_tif(key)
    if tif is None:
        raise FileNotFoundError(f"无原图 TIF: {key}")

    compare = _imread_unicode(png)
    if compare is None:
        raise RuntimeError(f"无法读取复核图: {png.name}")
    body = _right_body(compare)
    body_clean = _mask_scale_bar_region(body)
    lumens = _lumen_contours_from_green(body_clean)
    orange = _orange_box_contours(body)

    rgb, gray = load_rgb_gray(tif)
    scale = detect_scale_um_per_px(rgb)
    um = float(scale["um_per_px"])
    h, w = gray.shape[:2]
    bh, bw = body.shape[:2]
    sx, sy = w / bw, h / bh

    vessels = _vessels_from_green(lumens, tif.name, um, sx, sy, w, h)
    boxes: list[OrientedBox] = []
    for c in orange:
        cc = _scale_contour(c, sx, sy).reshape(-1, 2).astype(float)
        x0, x1 = float(cc[:, 0].min()), float(cc[:, 0].max())
        y0, y1 = float(cc[:, 1].min()), float(cc[:, 1].max())
        w, h = x1 - x0, y1 - y0
        if w < 20 or h < 20:
            continue
        boxes.append(
            OrientedBox(
                cx=(x0 + x1) / 2,
                cy=(y0 + y1) / 2,
                w=max(1.0, w),
                h=max(1.0, h),
                angle=0.0,
            )
        )

    guide_lines = _tvw_guide_lines(body, sx, sy)
    pairs = pair_from_two_largest_in_boxes(
        vessels,
        boxes,
        um,
        tif.name,
        pad=55.0,
        guide_lines=guide_lines,
    )
    if pairs and guide_lines:
        pairs = apply_human_tvw_guides(pairs, guide_lines, um)

    meta = {
        "image": tif.name,
        "sample": _sample_id(key),
        "um_per_px": um,
        "tissue_h": tissue_height(h),
        "n_vessel": len(vessels),
        "n_pair": len(pairs),
        "scale_method": scale.get("method"),
    }
    return vessels, pairs, meta


def main():
    ap = argparse.ArgumentParser(description="从复核对照图导出 Excel")
    ap.add_argument("--review", type=str, default="复核_1-40")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=40)
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 xlsx（默认 output/excel_samples/样品{lo}-{hi}_扫描电镜指标_自动分析结果.xlsx）",
    )
    args = ap.parse_args()
    lo, hi = int(args.lo), int(args.hi)
    review_dir = _resolve_review_dir(args.review)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = OUT_DIR / out_path
    else:
        out_path = OUT_DIR / f"样品{lo}-{hi}_扫描电镜指标_自动分析结果.xlsx"

    pngs = sorted(
        review_dir.glob("*_左原图_右自动.png"),
        key=lambda p: natural_key_name(image_key_from_stem(p.stem)),
    )
    pngs = [
        p
        for p in pngs
        if (sid := _sample_id(image_key_from_stem(p.stem))) is not None and lo <= sid <= hi
    ]
    if not pngs:
        raise SystemExit(f"复核目录无对照图: {review_dir}")

    print(f"复核目录: {review_dir}")
    print(f"样品 {lo}-{hi}: {len(pngs)} 张 | 从复核标注出表")
    print(f"Excel -> {out_path}")
    t0 = time.time()

    by_sample: dict[int, dict] = defaultdict(lambda: {"vessels": [], "pairs": []})
    images: list[dict] = []
    rows: list[dict] = []
    errors: list[tuple[str, str]] = []

    for i, png in enumerate(pngs, 1):
        key = image_key_from_stem(png.stem)
        sid = _sample_id(key) or 0
        try:
            vessels, pairs, meta = process_review_png(png)
            by_sample[sid]["vessels"].extend(vessels)
            by_sample[sid]["pairs"].extend(pairs)
            images.append(
                {
                    "image": meta["image"],
                    "sample": sid,
                    "um_per_px": meta["um_per_px"],
                    "tissue_h": meta["tissue_h"],
                    "n_vessel": meta["n_vessel"],
                    "n_pair": meta["n_pair"],
                }
            )
            rows.append(
                {
                    "image": meta["image"],
                    "sample": sid,
                    "n_vessel": meta["n_vessel"],
                    "n_pair": meta["n_pair"],
                    "um_per_px": round(float(meta["um_per_px"]), 6),
                    "scale_method": meta.get("scale_method"),
                    "status": "ok",
                    "error": "",
                    "review_png": png.name,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "image": key,
                    "sample": sid,
                    "n_vessel": "",
                    "n_pair": "",
                    "um_per_px": "",
                    "scale_method": "",
                    "status": "error",
                    "error": str(e),
                    "review_png": png.name,
                }
            )
            errors.append((png.name, traceback.format_exc()))

        if i % 40 == 0 or i == len(pngs) or i <= 3:
            print(f"[{i}/{len(pngs)}] {png.name} | {time.time() - t0:.0f}s")

    csv_path = OUT_DIR / f"样品{lo}-{hi}_batch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "sample",
                "n_vessel",
                "n_pair",
                "um_per_px",
                "scale_method",
                "status",
                "error",
                "review_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Writing Excel...")
    write_workbook(
        by_sample,
        images,
        out_path=out_path,
        sample_lo=lo,
        sample_hi=hi,
    )
    n_v = sum(len(by_sample[s]["vessels"]) for s in by_sample)
    n_p = sum(len(by_sample[s]["pairs"]) for s in by_sample)
    print(f"Done in {time.time() - t0:.1f}s | vessels={n_v} pairs={n_p}")
    print(f"CSV: {csv_path}")
    print(f"XLSX: {out_path}")
    if errors:
        err_path = OUT_DIR / f"样品{lo}-{hi}_errors.txt"
        err_path.write_text(
            "\n\n".join(f"== {n} ==\n{tb}" for n, tb in errors), encoding="utf-8"
        )
        print(f"Errors: {len(errors)} -> {err_path}")


if __name__ == "__main__":
    main()
