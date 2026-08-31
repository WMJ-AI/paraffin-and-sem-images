"""Remove manual annotation lines from calibrated images and save as originals."""

import cv2
import numpy as np
from pathlib import Path


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(path))


def _annotation_color_mask(img: np.ndarray) -> np.ndarray:
    """Detect bright saturated annotation line colors (green/orange/blue/red)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img)

    green = cv2.inRange(hsv, (40, 150, 150), (80, 255, 255))
    blue = cv2.inRange(hsv, (100, 200, 200), (130, 255, 255))
    orange = cv2.inRange(hsv, (5, 200, 200), (20, 255, 255))
    red = cv2.inRange(hsv, (0, 180, 180), (10, 255, 255)) | cv2.inRange(
        hsv, (170, 180, 180), (180, 255, 255)
    )
    red |= ((r > 170) & (g < 90) & (b < 90) & (r > g + 80) & (r > b + 80)).astype(np.uint8) * 255

    return green | blue | orange | red


def remove_annotation_lines(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    combined = _annotation_color_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=2)
    combined[: int(h * 0.15), int(w * 0.75) :] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    line_mask = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / (min(bh, bw) + 1)
        if area > 2000 or (area > 200 and aspect > 4):
            line_mask[labels == i] = 255

    if line_mask.sum() == 0:
        return img

    line_mask = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    return cv2.inpaint(img, line_mask, 7, cv2.INPAINT_TELEA)


def process_folder(src_dir: Path, dst_dir: Path) -> None:
    files = sorted(src_dir.glob("*.jpg"))
    print(f"Processing {len(files)} images from {src_dir}")
    for i, src_path in enumerate(files):
        img = imread_unicode(src_path)
        if img is None:
            continue
        result = remove_annotation_lines(img)
        imwrite_unicode(dst_dir / src_path.name, result)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i + 1}/{len(files)}: {src_path.name}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    process_folder(base / "人工标定" / "第一区域", base / "原图" / "第一区域")
