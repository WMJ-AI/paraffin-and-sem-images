"""Region-3 (cambium) rules and scoring constants.

Scoring rubric (from 需求.docx):
  1  - 上下衔接的小区域已经出现断层
  5  - 上下衔接的小区域排列散乱，即将断开
  10 - 上下衔接的小区域紧凑衔接
"""

from __future__ import annotations

# BGR outline color (manual-like magenta); no fill
CAMBIUM_COLOR = (179, 0, 122)
CAMBIUM_FILL_ALPHA = 0.0
CAMBIUM_LINE_THICKNESS = 4

# Default box rules: center-first; height ~1/4, hard max 1.5*(H/4)=0.375
DEFAULT_BOX_PARAMS = {
    "center_frac": 0.50,
    "height_frac": 0.25,
    "width_frac": 0.95,
    "x_center_frac": 0.50,
    "search_lo": 0.35,
    "search_hi": 0.65,
    "min_height_frac": 0.18,
    "min_width_frac": 0.88,
    "max_height_frac": 0.375,
}

# Rule thresholds used before calibration mapping
SCORE_RULES = {
    "gap_ratio_high": 0.35,
    "gap_ratio_mid": 0.18,
    "segment_count_high": 4,
    "alignment_std_high": 18.0,
    "continuity_low": 0.55,
}

DEFAULT_SCORE_WEIGHTS = {
    "continuity": 4.2,
    "alignment": -0.08,
    "gap_penalty": -8.5,
    "segment_penalty": -0.9,
    "base": 7.5,
}
