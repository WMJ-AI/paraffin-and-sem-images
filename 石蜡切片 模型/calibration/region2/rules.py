"""Region-2 rules learned from 人工标定/第二区域 (45 张 10X).

Manual JPGs mix several overlays. Yellow 导管 boxes are *examples*
(median 54 boxes/image, typically 1–3 lumens per box), not a complete
inventory. Auto must find every similar lumen in 木质部.

Stats from the largest round cavities inside those yellow boxes
(45 manuals, 4581 lumens in 2378 boxes):
  area p10 198 µm², p25 291, median 493, p75 738
  eq. diameter p10 16 µm, p25 19, median 25, p75 31
  yellow-box short side median ~50 µm (box is larger than the lumen)
  64% of labeled lumens are below the old 28 µm / 616 µm² floor
"""

from __future__ import annotations

# Printed scale on 10X xylem views is "100 µm" (black bar, ~412 px).
SCALE_MICRONS = 100.0
FALLBACK_BAR_PX = 412.0

# Floor from labeled lumens (p10), not the yellow-box outline.
MIN_DIAM_UM = 16.0
MIN_AREA_UM2 = 200.0  # ≈ π * (8 µm)²
MAX_DIAM_UM = 70.0

# Soft floor while proposing candidates; final keep uses MIN_AREA_UM2 / MIN_DIAM_UM.
DETECT_MIN_AREA_UM2 = 150.0
DETECT_MIN_DIAM_UM = 14.0

MIN_CIRCULARITY = 0.45
SMALL_MIN_CIRCULARITY = 0.55  # extra roundness below ~250 µm²
MAX_ASPECT = 2.2
RING_DELTA = 4.0

# Drawing (BGR): fill the lumen itself — not a detection rectangle.
VESSEL_FILL_BGR = (0, 210, 255)
VESSEL_OUTLINE_BGR = (0, 140, 230)
VESSEL_FILL_ALPHA = 0.50
VESSEL_LINE_THICKNESS = 3
