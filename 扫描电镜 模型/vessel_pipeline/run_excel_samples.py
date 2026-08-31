# -*- coding: utf-8 -*-
"""
按通用规则处理样品 lo–hi，导出与「样品1_扫描电镜指标_Wand精细优化版.xlsx」
同结构的 Excel（汇总 / 导管母表 / 审计 / 视野 / CWR对 / TVW线 / Sheet1…）。

用法:
  python run_excel_samples.py 1 40
  python run_excel_samples.py 1 40 --out 样品1-40_扫描电镜指标_自动分析结果.xlsx
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from batch_io import BATCH, list_unique_images, sample_id  # noqa: E402
from human_rules import find_pairs_auto, load_learned_rules, select_single_vessels  # noqa: E402
from run_batch_excel63 import write_workbook  # noqa: E402
from run_sample1 import process_image  # noqa: E402
from segment import tissue_height  # noqa: E402

OUT_DIR = ROOT / "output" / "excel_samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=int, nargs="?", default=1)
    ap.add_argument("hi", type=int, nargs="?", default=40)
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 xlsx 文件名或路径（默认 output/excel_samples/样品{lo}-{hi}_扫描电镜指标_自动分析结果.xlsx）",
    )
    args = ap.parse_args()
    lo, hi = int(args.lo), int(args.hi)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = OUT_DIR / out_path
    else:
        out_path = OUT_DIR / f"样品{lo}-{hi}_扫描电镜指标_自动分析结果.xlsx"

    paths = [
        p
        for p in list_unique_images()
        if str(sample_id(p)).isdigit() and lo <= int(sample_id(p)) <= hi
    ]
    rules = load_learned_rules()
    print(f"Batch: {BATCH}")
    print(f"样品 {lo}-{hi}: {len(paths)} 张图 | 通用规则")
    print(f"Excel -> {out_path}")
    t0 = time.time()

    by_sample: dict[int, dict] = defaultdict(lambda: {"vessels": [], "pairs": []})
    images: list[dict] = []
    rows: list[dict] = []
    errors: list[tuple[str, str]] = []

    for i, path in enumerate(paths, 1):
        sid = int(sample_id(path))
        try:
            rgb, gray, vessels, _, scale = process_image(path, force_um_per_px=None)
            for v in vessels:
                v.pair_id = None
            h, w = gray.shape[:2]
            pairs = find_pairs_auto(
                vessels,
                scale["um_per_px"],
                path.name,
                img_w=w,
                img_h=h,
                rules=rules,
            )
            # A/D/Dh = 双导管成员 + 框外单导管（与复核绿线一致）
            singles = select_single_vessels(vessels, pairs, w, h)
            counted: list = []
            seen: set[str] = set()
            for p in pairs:
                for v in (p.v1, p.v2):
                    if v.vessel_id not in seen:
                        counted.append(v)
                        seen.add(v.vessel_id)
            for v in singles:
                if v.vessel_id not in seen:
                    counted.append(v)
                    seen.add(v.vessel_id)

            by_sample[sid]["vessels"].extend(counted)
            by_sample[sid]["pairs"].extend(pairs)
            images.append(
                {
                    "image": path.name,
                    "sample": sid,
                    "um_per_px": float(scale["um_per_px"]),
                    "tissue_h": tissue_height(gray.shape[0]),
                    "n_vessel": len(counted),
                    "n_pair": len(pairs),
                }
            )
            rows.append(
                {
                    "image": path.name,
                    "sample": sid,
                    "n_vessel": len(counted),
                    "n_pair": len(pairs),
                    "um_per_px": round(float(scale["um_per_px"]), 6),
                    "scale_method": scale.get("method"),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "image": path.name,
                    "sample": sid,
                    "n_vessel": "",
                    "n_pair": "",
                    "um_per_px": "",
                    "scale_method": "",
                    "status": "error",
                    "error": str(e),
                }
            )
            errors.append((path.name, traceback.format_exc()))

        if i % 40 == 0 or i == len(paths) or i <= 3:
            print(f"[{i}/{len(paths)}] {path.name} | {time.time() - t0:.0f}s")

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
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"CSV: {csv_path}")
    if errors:
        err_path = OUT_DIR / f"样品{lo}-{hi}_errors.txt"
        err_path.write_text(
            "\n\n".join(f"== {n} ==\n{tb}" for n, tb in errors), encoding="utf-8"
        )
        print(f"Errors: {len(errors)} -> {err_path}")


if __name__ == "__main__":
    main()
