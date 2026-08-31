"""Pith-only error vs first-40 manual green starts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from calibration.io_util import imread
from calibration.region1_coords import load_coord_json
from calibration.stem import detect_pith_center, detect_stem_mask


def main():
    payload = load_coord_json(ROOT / "推理结果" / "region1_coords_first40.json")
    orig_dir = ROOT / "原图" / "第一区域"
    buckets = {2: [], 4: []}
    for fname, rec in payload["images"].items():
        orig = imread(orig_dir / fname)
        if orig is None:
            continue
        info = detect_stem_mask(orig)
        if info is None:
            continue
        mag = int(rec["magnification"])
        pith = detect_pith_center(orig, info[0], mag=mag)
        gx, gy = rec["green_start"]["x"], rec["green_start"]["y"]
        sr = rec["stem_r_px"]
        err = float(np.hypot(pith[0] - gx, pith[1] - gy) / sr)
        buckets[mag].append((fname, err))
        print(f"  {fname:16s} {mag}X  {err:.3f}")
    for mag, rows in buckets.items():
        arr = np.array([e for _, e in rows])
        print(
            f"{mag}X n={len(arr)} med={np.median(arr):.3f} mean={arr.mean():.3f} "
            f"<=0.08={float((arr<=0.08).mean()):.0%} <=0.12={float((arr<=0.12).mean()):.0%}"
        )


if __name__ == "__main__":
    main()
