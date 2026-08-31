from pathlib import Path
import numpy as np
from calibration.io_util import imread, parse_name, list_region_images
from calibration.region3.cambium import load_box_params, locate_cambium

base = Path(r"d:\BaiduNetdiskDownload\shibie")
params = load_box_params(base / "models" / "region3_box_params.json")
hs, ws = [], []
for i, p in enumerate(list_region_images(base, "第三区域", (7,8,9))):
    img = imread(p)
    box = locate_cambium(img, params)
    hs.append(box.y1-box.y0)
    ws.append(box.x1-box.x0)
    if i >= 20:
        break
print("h", min(hs), np.median(hs), max(hs))
print("w", min(ws), np.median(ws), max(ws))
print("aspect w/h", np.median(np.array(ws)/np.array(hs)))
print("img", img.shape)
