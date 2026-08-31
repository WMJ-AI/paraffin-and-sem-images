"""Unicode-safe image I/O and filename parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageMeta:
    sample_id: int
    view_id: int
    magnification: int
    filename: str

    @property
    def region1(self) -> bool:
        return self.view_id in (1, 2, 3)

    @property
    def region2(self) -> bool:
        return self.view_id in (4, 5, 6)

    @property
    def region3(self) -> bool:
        return self.view_id in (7, 8, 9)


def imread(path: Path | str) -> np.ndarray | None:
    path = Path(path)
    data = np.fromfile(str(path.resolve()), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path | str, img: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tofile(str(path.resolve()))


def parse_name(filename: str) -> ImageMeta | None:
    name = re.sub(r"\s+", " ", filename.strip())
    match = re.match(r"(\d+)-(\d+)\s+(\d+)X\.jpg$", name, flags=re.IGNORECASE)
    if not match:
        return None
    return ImageMeta(
        sample_id=int(match.group(1)),
        view_id=int(match.group(2)),
        magnification=int(match.group(3)),
        filename=filename,
    )


def list_region_images(base_dir: Path, region: str, view_ids: tuple[int, ...] | None = None) -> list[Path]:
    folder = base_dir / "原图" / region
    files = sorted(folder.glob("*.jpg"))
    if view_ids is None:
        return files
    out: list[Path] = []
    for path in files:
        meta = parse_name(path.name)
        if meta and meta.view_id in view_ids:
            out.append(path)
    return out
