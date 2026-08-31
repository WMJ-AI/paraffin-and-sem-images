"""Inspect manual ink vs filled masks for a few region-2 images."""
from pathlib import Path

import cv2
import numpy as np

from calibration.io_util import imread
from calibration.region2.manual_areas import (
    parse_manual_area_boxes,
    parse_manual_areas,
    parse_overlay_rects,
)

base = Path(r"d:\BaiduNetdiskDownload\shibie")
orig_dir = base / "原图" / "第二区域"
man_dir = base / "人工标定" / "第二区域"
out = base / "_tmp_area_vis"
out.mkdir(exist_ok=True)


def inner_profile(mask, from_left):
    h, w = mask.shape
    xs = []
    for y in range(h):
        cols = np.where(mask[y] > 0)[0]
        if cols.size == 0:
            xs.append(-1)
        elif from_left:
            xs.append(int(cols.max()) + 1)
        else:
            xs.append(int(cols.min()))
    return np.array(xs)


names = ["15-5 10X.jpg", "9-4 10X.jpg", "5-5 10X.jpg", "2-4 10X.jpg", "10-5 10X.jpg"]
for name in names:
    orig = imread(orig_dir / name)
    man = imread(man_dir / name)
    h, w = orig.shape[:2]
    ph1, pi1 = parse_manual_areas(orig, man)
    ph2, pi2, pb, ib = parse_manual_area_boxes(orig, man)
    n_ph = len(parse_overlay_rects(orig, man, 100, 116, 80, 255))
    n_pi = len(parse_overlay_rects(orig, man, 118, 155, 40, 255))
    print(name)
    print(
        f"  overlay rects ph={n_ph} pith={n_pi}  "
        f"rowfill ph={cv2.countNonZero(ph1)} pith={cv2.countNonZero(pi1)}  "
        f"boxfill ph={cv2.countNonZero(ph2)} pith={cv2.countNonZero(pi2)}"
    )
    vis = orig.copy()
    vis[ph1 > 0] = (vis[ph1 > 0] * 0.5 + np.array([255, 220, 0]) * 0.5).astype(np.uint8)
    vis[pi1 > 0] = (vis[pi1 > 0] * 0.5 + np.array([200, 40, 180]) * 0.5).astype(np.uint8)
    # draw inner polyline of row-fill
    for mask, color, left_guess in (
        (ph1, (255, 220, 0), True),
        (pi1, (200, 40, 180), False),
    ):
        if cv2.countNonZero(mask) == 0:
            continue
        xs = np.where(mask > 0)[1]
        from_left = float(xs.mean()) < 0.5 * w
        prof = inner_profile(mask, from_left)
        pts = [(int(prof[y]), y) for y in range(h) if prof[y] >= 0]
        for i in range(0, len(pts) - 1, 8):
            cv2.line(vis, pts[i], pts[min(i + 8, len(pts) - 1)], color, 6)
        widths = []
        for y in range(h):
            cols = np.where(mask[y] > 0)[0]
            if cols.size:
                widths.append(int(cols.max() - cols.min() + 1))
        if widths:
            print(
                f"  rowfill {'ph' if color==(255,220,0) else 'pi'} "
                f"side={'L' if from_left else 'R'} "
                f"W p10/50/90={np.percentile(widths,[10,50,90]).astype(int).tolist()} "
                f"inner std={float(np.std([p for p in prof if p>=0])):.1f}"
            )
    # left and right crops
    left = vis[:, : int(0.32 * w)]
    right = vis[:, int(0.68 * w) :]
    cv2.imencode(".jpg", left, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(
        str((out / f"{name}_L.jpg").resolve())
    )
    cv2.imencode(".jpg", right, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(
        str((out / f"{name}_R.jpg").resolve())
    )

# how many manuals actually have cyan/purple ink?
print("\n=== ink presence ===")
have = 0
for mp in sorted(man_dir.glob("*.jpg")):
    orig = imread(orig_dir / mp.name)
    man = imread(mp)
    ph, pi = parse_manual_areas(orig, man)
    if cv2.countNonZero(ph) or cv2.countNonZero(pi):
        have += 1
        w = orig.shape[1]
        for tag, mask in (("ph", ph), ("pi", pi)):
            if cv2.countNonZero(mask) == 0:
                continue
            xs = np.where(mask > 0)[1]
            from_left = float(xs.mean()) < 0.5 * w
            widths = []
            for y in range(mask.shape[0]):
                cols = np.where(mask[y] > 0)[0]
                if cols.size:
                    widths.append(int(cols.max() - cols.min() + 1))
            print(
                f"{mp.name} {tag} {'L' if from_left else 'R'} "
                f"W50={int(np.median(widths))} frac={np.median(widths)/w:.3f}"
            )
print("manuals with area ink", have, "/", len(list(man_dir.glob('*.jpg'))))
