"""Only image product: side-by-side 原图 | 自动标注."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def imwrite_unicode(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    path.write_bytes(buf.tobytes())


def _cn_font(size: int = 22):
    for name in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if Path(name).exists():
            try:
                return ImageFont.truetype(name, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


LABEL_H = 36


def panel_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """在图像上方追加标题条，不覆盖内容（避免顶栏挡住橙框/绿轮廓）。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    bar = Image.new("RGB", (im.width, LABEL_H), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    draw.text((10, 6), text, fill=(240, 240, 240), font=_cn_font(22))
    out = Image.new("RGB", (im.width, im.height + LABEL_H))
    out.paste(bar, (0, 0))
    out.paste(im, (0, LABEL_H))
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def make_compare_bgr(rgb: np.ndarray, auto_rgb: np.ndarray, image_name: str) -> np.ndarray:
    """Left = original, right = auto annotation."""
    left = panel_label(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), f"{image_name} | 原图（请在此侧对照修改）"
    )
    right = panel_label(
        cv2.cvtColor(auto_rgb, cv2.COLOR_RGB2BGR), f"{image_name} | 自动标注（参考）"
    )
    return cv2.hconcat([left, right])
