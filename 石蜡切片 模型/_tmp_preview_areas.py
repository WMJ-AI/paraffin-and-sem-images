"""Preview line+inner-rect drawing on a few region-2 images."""
from pathlib import Path

from calibration.io_util import imread, imwrite
from calibration.region2.areas import load_area_model, stack_band_boxes
from calibration.region2.draw import draw_region2
from calibration.region2.vessels import analyze_region2

base = Path(r"d:\BaiduNetdiskDownload\shibie")
orig_dir = base / "原图" / "第二区域"
man_dir = base / "人工标定" / "第二区域"
out = base / "_tmp_area_preview"
out.mkdir(exist_ok=True)

clf = load_area_model(base / "models" / "region2_area_clf.joblib")
names = ["15-5 10X.jpg", "9-4 10X.jpg", "5-5 10X.jpg", "2-4 10X.jpg"]
for name in names:
    img = imread(orig_dir / name)
    man = imread(man_dir / name)
    result = analyze_region2(img, area_clf=clf, manual=man, image_name=name)
    annotated = draw_region2(img, result)
    imwrite(out / name, annotated)
    h, w = img.shape[:2]
    imwrite(out / f"{name}_L.jpg", annotated[:, : int(0.28 * w)])
    imwrite(out / f"{name}_R.jpg", annotated[:, int(0.72 * w) :])
    print(
        name,
        "ph_boxes",
        len(result.phloem_boxes),
        "pi_boxes",
        len(result.pith_boxes),
        "ph_um2",
        round(result.phloem_area_um2),
        "pi_um2",
        round(result.pith_area_um2),
        "vessels",
        result.count,
    )
    if result.phloem_boxes:
        print("  ph W", [b[2] for b in result.phloem_boxes])
    if result.pith_boxes:
        print("  pi W", [b[2] for b in result.pith_boxes])
