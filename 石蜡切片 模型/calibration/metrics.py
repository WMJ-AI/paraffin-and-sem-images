"""Length measurements from region-1 geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from calibration.geometry import LineGeometry
from calibration.scale import ScaleInfo, pixels_to_microns


@dataclass
class Region1Measurements:
    radius_um: float
    xylem_um: float
    phloem_um: float
    bark_um: float


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def measure_region1(geom: LineGeometry, scale: ScaleInfo) -> Region1Measurements:
    center = geom.center
    green_end = geom.green_end
    xylem = (geom.xylem_end, geom.xylem_end_y or geom.seg_y)
    phloem = (geom.phloem_end, geom.phloem_end_y or geom.seg_y)
    bark = (geom.bark_end, geom.bark_end_y or geom.seg_y)
    seg0 = (geom.seg_x0, geom.seg_y0 or geom.seg_y)

    radius_px = _dist(center, green_end)
    xylem_px = _dist(seg0, xylem)
    phloem_px = _dist(xylem, phloem)
    bark_px = _dist(phloem, bark)

    return Region1Measurements(
        radius_um=pixels_to_microns(radius_px, scale),
        xylem_um=pixels_to_microns(xylem_px, scale),
        phloem_um=pixels_to_microns(phloem_px, scale),
        bark_um=pixels_to_microns(bark_px, scale),
    )
