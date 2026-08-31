"""Compare YOLO+rule starts vs JSON six-point labels."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from calibration.io_util import imread, parse_name
from calibration.region1_sixpoint import geom_from_record, load_relabel_records, six_points
from calibration.stem import detect_stem_mask
from ml.yolo_infer import YoloRegion1Model


def main():
    base = ROOT
    recs = load_relabel_records(base)
    orig_dir = base / "原图" / "第一区域"
    w = base / "models" / "yolo" / "region1_pose_json_best.pt"
    model = YoloRegion1Model(w)
    pith_e, ring_e, dist_auto, dist_gt = [], [], [], []
    by_mag = {2: [], 4: []}
    distinct = 0
    n = 0
    items = list(recs.items())[:40]
    for i, (name, rec) in enumerate(items):
        orig = imread(orig_dir / name)
        if orig is None:
            continue
        meta = parse_name(name)
        mag = meta.magnification if meta else 2
        geom = model.predict(orig, mag=mag)
        if geom is None:
            continue
        gt = geom_from_record(rec)
        sr = rec.get("stem_r_px")
        if not sr:
            info = detect_stem_mask(orig)
            sr = float(info[1]) if info else 1500.0
        g_a, l_a = six_points(geom)[0], six_points(geom)[2]
        g_t, l_t = six_points(gt)[0], six_points(gt)[2]
        pe = math.hypot(g_a[0] - g_t[0], g_a[1] - g_t[1]) / sr
        ra = math.hypot(l_a[0] - g_a[0], l_a[1] - g_a[1]) / sr
        rg = math.hypot(l_t[0] - g_t[0], l_t[1] - g_t[1]) / sr
        pith_e.append(pe)
        ring_e.append(abs(ra - rg))
        dist_auto.append(ra)
        dist_gt.append(rg)
        by_mag[mag].append(pe)
        if ra >= 0.10:
            distinct += 1
        n += 1
        if (i + 1) % 10 == 0:
            print(f"  eval {i + 1}/{len(items)}")
    arr = np.array(pith_e)
    print(f"n={n}  髓心误差 med={np.median(arr):.3f}  <=0.08={float((arr<=0.08).mean()):.0%}")
    print(f"      内环半径差 med={np.median(ring_e):.3f}  两起点分离={distinct}/{n}")
    print(f"      自动两起点距 med={np.median(dist_auto):.3f}  标签两起点距 med={np.median(dist_gt):.3f}")
    for mag, vals in by_mag.items():
        if not vals:
            continue
        a = np.array(vals)
        print(f"      {mag}X n={len(a)} 髓心med={np.median(a):.3f} <=0.08={float((a<=0.08).mean()):.0%}")


if __name__ == "__main__":
    main()
