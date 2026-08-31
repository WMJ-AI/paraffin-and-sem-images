"""How close is 6-point snap / current YOLO to actual 人工标定 ink."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from calibration.geometry import parse_manual_geometry
from calibration.io_util import imread, parse_name
from calibration.region1_sixpoint import canonicalize, geom_from_record, load_relabel_records
from calibration.stem import detect_stem_mask
from ml.metrics import region1_geometry_similarity


def main():
    base = ROOT
    orig_dir = base / "原图" / "第一区域"
    man_dir = base / "人工标定" / "第一区域"
    six = load_relabel_records(base)
    files = sorted(man_dir.glob("*.jpg"))[:40]
    snap_s, json_s = [], []
    print(f"对照人工墨线 {len(files)} 张...")
    for i, mp in enumerate(files):
        orig = imread(orig_dir / mp.name)
        man = imread(mp)
        if orig is None or man is None:
            continue
        parse = parse_manual_geometry(man, orig)
        if parse is None:
            continue
        meta = parse_name(mp.name)
        mag = meta.magnification if meta else 2
        info = detect_stem_mask(orig)
        stem = info[0] if info else None
        h, w = orig.shape[:2]
        diag = math.hypot(h, w)
        snapped = canonicalize(orig, stem, hint=parse, mag=mag, trust_hint_pith=True)
        if snapped is not None:
            snap_s.append(region1_geometry_similarity(snapped, parse, diag))
        if mp.name in six:
            g = geom_from_record(six[mp.name])
            json_s.append(region1_geometry_similarity(g, parse, diag))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(files)}")
    for name, arr in ("扣齐人工", snap_s), ("现有JSON", json_s):
        a = np.array(arr, dtype=np.float64)
        print(
            f"{name} n={len(a)} mean={a.mean():.1%} med={np.median(a):.1%} "
            f">=95%={float((a >= 0.95).mean()):.0%} >=85%={float((a >= 0.85).mean()):.0%}"
        )


if __name__ == "__main__":
    main()
