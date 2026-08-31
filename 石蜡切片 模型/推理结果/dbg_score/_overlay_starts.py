"""Draw manual starts vs pith detectors on a few first-40 images."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from calibration.io_util import imread, imwrite
from calibration.region1_coords import load_coord_json
from calibration.stem import detect_pith, detect_stem_mask


def mark(img, xy, color, label):
    p = (int(xy[0]), int(xy[1]))
    cv2.circle(img, p, 18, color, 4, cv2.LINE_AA)
    cv2.putText(img, label, (p[0] + 22, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)


def main():
    payload = load_coord_json(ROOT / "推理结果" / "region1_coords_first40.json")
    orig_dir = ROOT / "原图" / "第一区域"
    out = ROOT / "推理结果" / "dbg_starts"
    out.mkdir(parents=True, exist_ok=True)
    names = [
        "1-1 2X.jpg",
        "4-1 2X.jpg",
        "7-1 2X.jpg",
        "1-2 4X.jpg",
        "2-2  4X.jpg",
        "5-2 4X.jpg",
        "10-2  4X.jpg",
        "13-2  4X.jpg",
    ]
    for name in names:
        rec = payload["images"].get(name)
        orig = imread(orig_dir / name)
        if rec is None or orig is None:
            print("skip", name)
            continue
        info = detect_stem_mask(orig)
        vis = orig.copy()
        g = (rec["green_start"]["x"], rec["green_start"]["y"])
        p0 = (rec["layer_start"]["x"], rec["layer_start"]["y"])
        mark(vis, g, (0, 180, 0), "G0")
        mark(vis, p0, (0, 140, 255), "L0")
        if info is not None:
            stem, _ = info
            dist = cv2.distanceTransform(stem, cv2.DIST_L2, 5)
            yx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            mark(vis, (float(yx[1]), float(yx[0])), (0, 0, 255), "DT")
            mark(vis, detect_pith(orig, stem), (255, 0, 255), "DK")
        h, w = vis.shape[:2]
        scale = 900 / max(h, w)
        small = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        imwrite(out / name.replace(" ", "_"), small)
        print("saved", name)


if __name__ == "__main__":
    main()
