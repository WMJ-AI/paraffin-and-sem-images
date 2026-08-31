"""Find labelme->manual mapping by ink hit on 1432x960 resized manual."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import _color_layers
from calibration.io_util import imread
from calibration.region1_coords import first_n_manuals

BASE = Path(__file__).resolve().parents[2]
LM_PATH = BASE / "推理结果" / "labelme_first30_raw.txt"
LABELS = ["半径", "木质部", "韧皮部", "树皮皮层"]
COLORS = ["green", "orange", "blue", "red"]


def split_json(text: str) -> list[dict]:
    objs, buf, depth = [], "", 0
    for ch in text:
        if ch == "{":
            depth += 1
        if depth > 0:
            buf += ch
        if ch == "}":
            depth -= 1
            if depth == 0 and buf.strip():
                objs.append(json.loads(buf))
                buf = ""
    return objs


def ink_score(lm: dict, man_path: Path) -> float:
    man = imread(man_path)
    if man is None:
        return -1e9
    small = cv2.resize(man, (lm["imageWidth"], lm["imageHeight"]))
    green, orange, blue, red = _color_layers(small)
    masks = {"半径": green, "木质部": orange, "韧皮部": blue, "树皮皮层": red}
    score = 0.0
    for s in lm["shapes"]:
        m = masks.get(s["label"])
        if m is None:
            return -1e9
        for pt in s["points"]:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= x < m.shape[1] and 0 <= y < m.shape[0]):
                return -1e9
            r = 6
            patch = m[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
            hit = float(np.mean(patch > 0))
            score += hit
    return score


def main() -> None:
    objs = split_json(LM_PATH.read_text(encoding="utf-8"))
    manuals = first_n_manuals(BASE, 30)
    paths = [BASE / "人工标定" / "第一区域" / mp.name for mp in manuals]

    print("labelme -> best ink match on resized manual")
    used: set[str] = set()
    mapping: list[tuple[str, str, float]] = []
    for lm in objs:
        scores = [(ink_score(lm, p), p.name) for p in paths]
        scores.sort(reverse=True)
        best_s, best = scores[0]
        mapping.append((lm["imagePath"], best, best_s))
        print(f"  {lm['imagePath']:10s} -> {best:16s} score={best_s:.3f}")

    # check uniqueness
    names = [m[1] for m in mapping]
    from collections import Counter
    dup = [k for k, v in Counter(names).items() if v > 1]
    print("duplicate targets:", len(dup), dup[:10] if dup else "none")

    # first 3 assumed vs best
    print("\nfirst3 compare:")
    for i in range(3):
        lm = objs[i]
        assumed = manuals[i].name
        best = mapping[i][1]
        s_assumed = ink_score(lm, paths[i])
        s_best = mapping[i][2]
        print(f"  {lm['imagePath']}: assumed {assumed} score={s_assumed:.3f}  best {best} score={s_best:.3f}")


if __name__ == "__main__":
    main()
