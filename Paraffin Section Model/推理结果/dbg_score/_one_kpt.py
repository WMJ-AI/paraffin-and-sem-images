import sys
from pathlib import Path
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8")

from calibration.geometry import parse_manual_geometry
from calibration.io_util import imread
from calibration.region1_sixpoint import light_from_keypoints, six_points
from calibration.stem import detect_stem_mask
from ml.metrics import region1_geometry_similarity
from ml.yolo_infer import YoloRegion1Model

base = Path(__file__).resolve().parents[2]
name = "1-1 2X.jpg"
orig = imread(base / "原图" / "第一区域" / name)
man = imread(base / "人工标定" / "第一区域" / name)
parse = parse_manual_geometry(man, orig)
model = YoloRegion1Model(base / "models" / "yolo" / "region1_pose_ink_best.pt")
kpts = model.keypoints(orig)
print("kpts", None if kpts is None else kpts.tolist())
print("parse pith", parse.center, "green_end", parse.green_end)
print("parse p0", parse.seg_x0, parse.seg_y0, "bark", parse.bark_end, parse.bark_end_y)
if kpts is not None:
    stem = detect_stem_mask(orig)
    geom = light_from_keypoints(kpts, stem[0] if stem else None)
    h, w = orig.shape[:2]
    print("sim", region1_geometry_similarity(geom, parse, math.hypot(h, w)))
    print("pred pts", six_points(geom))
    for i, (p, g) in enumerate(zip(six_points(geom), [
        parse.center, parse.green_end,
        (parse.seg_x0, parse.seg_y0 or parse.seg_y),
        (parse.xylem_end, parse.xylem_end_y or parse.seg_y),
        (parse.phloem_end, parse.phloem_end_y or parse.seg_y),
        (parse.bark_end, parse.bark_end_y or parse.seg_y),
    ])):
        print(f"  pt{i+1} err={math.hypot(p[0]-g[0], p[1]-g[1]):.1f}px")
    print("pred ang", geom.green_angle_deg, geom.seg_angle_deg)
    print("gt ang", parse.green_angle_deg, parse.seg_angle_deg)
