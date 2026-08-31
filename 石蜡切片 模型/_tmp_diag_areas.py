from pathlib import Path

import cv2
import numpy as np

from calibration.io_util import imread
from calibration.region2.areas import detect_phloem_pith, load_area_model, stack_band_boxes
from calibration.region2.manual_areas import parse_manual_area_boxes

base = Path(r"d:\BaiduNetdiskDownload\shibie")
orig_dir = base / "原图" / "第二区域"
man_dir = base / "人工标定" / "第二区域"
clf = load_area_model(base / "models" / "region2_area_clf.joblib")

names = [
    "15-5 10X.jpg",
    "9-4 10X.jpg",
    "1-5 10X.jpg",
    "8-5 10X.jpg",
    "5-5 10X.jpg",
    "11-5  10X.jpg",
    "2-4 10X.jpg",
    "7-6 10X.jpg",
]


def stats(mask, boxes, tag, w):
    nz = cv2.countNonZero(mask)
    if nz == 0:
        print(f"  {tag}: empty boxes={len(boxes)}")
        return
    ys, xs = np.where(mask > 0)
    from_left = float(xs.mean()) < 0.5 * w
    widths = []
    inners = []
    for y in range(int(ys.min()), int(ys.max()) + 1):
        cols = np.where(mask[y] > 0)[0]
        if cols.size:
            widths.append(int(cols.max() - cols.min() + 1))
            inners.append(int(cols.max()) if from_left else int(cols.min()))
    bw = [b[2] for b in boxes]
    bh = [b[3] for b in boxes]
    side = "L" if from_left else "R"
    wp = np.percentile(widths, [10, 50, 90]).astype(int).tolist()
    inn = np.percentile(inners, [10, 50, 90]).astype(int).tolist()
    print(
        f"  {tag}: boxes={len(boxes)} area={nz} side={side} "
        f"maskW p10/50/90={wp} innerX p10/50/90={inn} "
        f"boxW med={int(np.median(bw)) if bw else 0} "
        f"boxH med={int(np.median(bh)) if bh else 0} "
        f"yspan={int(ys.max() - ys.min())} fracW={float(np.median(widths)) / w:.3f}"
    )


for name in names:
    orig = imread(orig_dir / name)
    man = imread(man_dir / name)
    if orig is None or man is None:
        print(name, "MISSING")
        continue
    h, w = orig.shape[:2]
    ph, pi, pb, ib = parse_manual_area_boxes(orig, man)
    print(f"=== {name} {w}x{h} ===")
    print(" MANUAL")
    stats(ph, pb, "phloem", w)
    stats(pi, ib, "pith", w)
    aph, api = detect_phloem_pith(orig, clf)
    print(" AUTO")
    stats(aph, stack_band_boxes(aph), "phloem", w)
    stats(api, stack_band_boxes(api), "pith", w)

# also summarize all manuals
print("\n=== ALL MANUALS box counts ===")
n_ph, n_pi = [], []
w_ph, w_pi = [], []
for mp in sorted(man_dir.glob("*.jpg")):
    orig = imread(orig_dir / mp.name)
    man = imread(mp)
    if orig is None or man is None:
        continue
    ph, pi, pb, ib = parse_manual_area_boxes(orig, man)
    n_ph.append(len(pb))
    n_pi.append(len(ib))
    w_ph.extend([b[2] for b in pb])
    w_pi.extend([b[2] for b in ib])
print(
    "phloem boxes/img p10/50/90",
    np.percentile(n_ph, [10, 50, 90]).astype(int).tolist(),
    "n=",
    len(n_ph),
)
print(
    "pith boxes/img p10/50/90",
    np.percentile(n_pi, [10, 50, 90]).astype(int).tolist(),
)
if w_ph:
    print("phloem boxW p10/50/90", np.percentile(w_ph, [10, 50, 90]).astype(int).tolist())
if w_pi:
    print("pith boxW p10/50/90", np.percentile(w_pi, [10, 50, 90]).astype(int).tolist())
