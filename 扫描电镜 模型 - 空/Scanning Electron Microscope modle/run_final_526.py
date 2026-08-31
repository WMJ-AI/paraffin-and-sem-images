"""
可选：526 张统计写 Excel（纯通用规则，不读逐图指南）。
"""
from __future__ import annotations

import csv
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from batch_io import BATCH, list_unique_images, sample_id  # noqa: E402
from human_rules import find_pairs_auto, select_single_vessels  # noqa: E402
from run_batch_excel63 import XLSX_OUT, write_workbook  # noqa: E402
from run_sample1 import process_image  # noqa: E402
from segment import tissue_height  # noqa: E402

OUT = ROOT / "output" / "batch_all_526"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    paths = list_unique_images()
    print(f"Batch: {BATCH}")
    print(f"Images: {len(paths)} (纯通用规则自动)")
    print(f"Excel -> {XLSX_OUT}")
    t0 = time.time()

    by_sample: dict[int, dict] = defaultdict(lambda: {"vessels": [], "pairs": []})
    images: list[dict] = []
    rows, errors = [], []

    for i, path in enumerate(paths, 1):
        sid_s = sample_id(path)
        sid = int(sid_s) if str(sid_s).isdigit() else 0
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
            )
            # A/D/Dh = 双导管腔 + 其外单导管（与绿线一致；不把碎腔全计入）
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
                    "sample": sid_s,
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
                    "sample": sid_s,
                    "n_vessel": "",
                    "n_pair": "",
                    "um_per_px": "",
                    "scale_method": "",
                    "status": "error",
                    "error": str(e),
                }
            )
            errors.append((path.name, traceback.format_exc()))

        if i % 50 == 0 or i == len(paths) or i <= 2:
            print(f"[{i}/{len(paths)}] {path.name}")

    csv_path = OUT / "batch_summary.csv"
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
    write_workbook(by_sample, images)
    print(f"Done in {time.time()-t0:.1f}s -> {XLSX_OUT}")
    if errors:
        (OUT / "batch_errors.txt").write_text(
            "\n\n".join(f"== {n} ==\n{tb}" for n, tb in errors), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
