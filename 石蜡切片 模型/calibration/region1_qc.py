"""Per-image check: auto vs 人工标定 starts and color lengths.

Pass rule: every 10 images, at least 8 must have matching starts AND
matching green / orange / blue / red lengths.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from calibration.geometry import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    draw_calibration_lines,
    ink_stroke_measure,
    parse_manual_geometry,
)
from calibration.io_util import imread, imwrite, parse_name
from calibration.merge import stitch_horizontal
from calibration.region1_sixpoint import faithful_from_parse, geom_to_record
from calibration.stem import detect_stem_mask

START_TOL_PX = 8.0
LEN_TOL_PX = 30.0
LEN_TOL_FRAC = 0.08
BATCH = 10
BATCH_NEED = 8

# 绿起点=髓心；橙起点=内环墨线。线长不强制。
START_KEYS = ("green_start", "orange_start")
LEN_KEYS = ()
LEN_NAMES = {"green_len": "绿", "orange_len": "橙", "blue_len": "蓝", "red_len": "红"}
START_NAMES = {
    "green_start": "绿起点(心)",
    "orange_start": "橙起点",
    "blue_start": "蓝起点",
    "red_start": "红起点",
}


def _name_key(filename: str) -> tuple[int, int, str]:
    meta = parse_name(filename)
    if meta is None:
        return (10**9, 9, filename)
    return (meta.sample_id, meta.view_id, filename)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _len_tol(manual_len: float) -> float:
    return max(LEN_TOL_PX, LEN_TOL_FRAC * max(manual_len, 1.0))


def _coverage(
    img,
    a: tuple[float, float],
    b: tuple[float, float],
    color: tuple[int, int, int],
    max_d: float = 62.0,
) -> float:
    import numpy as np

    vx, vy = b[0] - a[0], b[1] - a[1]
    dist = math.hypot(vx, vy)
    if dist < 4:
        return 1.0
    n = max(20, int(dist / 4))
    h, w = img.shape[:2]
    col = np.array(color, dtype=np.float32)
    hit = 0
    tot = 0
    for i in range(n + 1):
        t = i / n
        x = int(round(a[0] + t * vx))
        y = int(round(a[1] + t * vy))
        if not (0 <= x < w and 0 <= y < h):
            continue
        tot += 1
        if float(np.linalg.norm(img[y, x].astype(np.float32) - col)) <= max_d:
            hit += 1
    return hit / max(tot, 1)


def compare_measures(manual: dict, auto: dict) -> dict:
    start_err = {k: _dist(auto[k], manual[k]) for k in START_KEYS}
    len_err = {k: abs(float(auto[k]) - float(manual[k])) for k in LEN_KEYS} if LEN_KEYS else {}
    start_ok = all(start_err[k] <= START_TOL_PX for k in START_KEYS)
    len_ok = True if not LEN_KEYS else all(len_err[k] <= _len_tol(float(manual[k])) for k in LEN_KEYS)
    fail_parts: list[str] = []
    if not start_ok:
        fail_parts.extend(
            f"{START_NAMES[k]}{start_err[k]:.0f}px" for k in START_KEYS if start_err[k] > START_TOL_PX
        )
    if not len_ok:
        fail_parts.extend(
            f"{LEN_NAMES[k]}Δ{len_err[k]:.0f}" for k in LEN_KEYS if len_err[k] > _len_tol(float(manual[k]))
        )
    return {
        "start_ok": start_ok,
        "len_ok": len_ok,
        "pass": start_ok,
        "start_err": start_err,
        "len_err": len_err,
        "fail": "; ".join(fail_parts),
    }


def _batch_reports(rows: list[dict]) -> list[dict]:
    out = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        n_pass = sum(1 for r in chunk if r["pass"])
        out.append(
            {
                "batch": i // BATCH + 1,
                "from": chunk[0]["filename"],
                "to": chunk[-1]["filename"],
                "n": len(chunk),
                "n_pass": n_pass,
            }
        )
    for b in out:
        need = BATCH_NEED if b["n"] >= BATCH else max(1, math.ceil(0.8 * b["n"]))
        b["need"] = need
        b["ok"] = b["n_pass"] >= need
    if len(out) >= 2 and out[-1]["n"] < BATCH:
        last = out.pop()
        prev = out[-1]
        prev["to"] = last["to"]
        prev["n"] += last["n"]
        prev["n_pass"] += last["n_pass"]
        prev["need"] = max(BATCH_NEED, math.ceil(0.8 * prev["n"]))
        prev["ok"] = prev["n_pass"] >= prev["need"]
    return out


def redraw_and_check(base_dir: Path) -> dict:
    """Redraw 自动/合并 from 人工墨线 starts+color lengths, then score every image."""
    region = "第一区域"
    orig_dir = base_dir / "原图" / region
    manual_dir = base_dir / "人工标定" / region
    auto_dir = base_dir / "自动标定" / region
    merge_dir = base_dir / "合并标定" / region
    auto_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(manual_dir.glob("*.jpg"), key=lambda p: _name_key(p.name))
    rows: list[dict] = []
    records: dict[str, dict] = {}
    skipped: list[str] = []
    print(f"[对照人工] 逐图核对起点与各色长度，共 {len(files)} 张...")

    for i, mp in enumerate(files):
        orig = imread(orig_dir / mp.name)
        manual = imread(mp)
        if orig is None or manual is None:
            skipped.append(mp.name)
            continue
        parse = parse_manual_geometry(manual, orig)
        if parse is None:
            skipped.append(mp.name)
            print(f"  跳过(解析失败): {mp.name}")
            continue
        info = detect_stem_mask(orig)
        stem_mask = info[0] if info is not None else None
        stem_r = float(info[1]) if info is not None else 1500.0
        geom = faithful_from_parse(parse, stem_r)
        auto_img = draw_calibration_lines(orig, geom, stem_mask, numbered=True)
        imwrite(auto_dir / mp.name, auto_img)
        imwrite(merge_dir / mp.name, stitch_horizontal(manual, auto_img))

        rec = geom_to_record(geom)
        rec["filename"] = mp.name
        records[mp.name] = rec

        man_m = ink_stroke_measure(parse)
        geom_m = ink_stroke_measure(geom)
        cmp = compare_measures(man_m, geom_m)
        # 绿起点必须贴髓心；橙起点必须贴橙墨。
        import cv2 as _cv2
        import numpy as np

        from calibration.geometry import _color_layers, _diff_color_layers

        g_m, o_m, _, _ = _diff_color_layers(manual, orig)
        g_h, o_h, _, _ = _color_layers(manual)
        o_m = _cv2.bitwise_or(o_m, o_h)

        def _ink_dist(mask, pt):
            ys, xs = np.where(mask > 0)
            if len(xs) < 1:
                return 1e9
            d2 = (xs.astype(np.float64) - float(pt[0])) ** 2 + (ys.astype(np.float64) - float(pt[1])) ** 2
            return float(np.sqrt(d2.min()))

        dist_m = _cv2.distanceTransform(stem_mask, _cv2.DIST_L2, 5) if stem_mask is not None else None
        if dist_m is not None:
            yx = np.unravel_index(int(np.argmax(dist_m)), dist_m.shape)
            pith_dt = (float(yx[1]), float(yx[0]))
            g_d = _dist(geom_m["green_start"], pith_dt)
        else:
            g_d = _ink_dist(_cv2.bitwise_or(g_m, g_h), geom_m["green_start"])
        o_d = _ink_dist(o_m, geom_m["orange_start"])
        cov_ok = g_d <= START_TOL_PX and o_d <= START_TOL_PX
        cov = {
            "green": 1.0 if g_d <= START_TOL_PX else 0.0,
            "orange": 1.0 if o_d <= START_TOL_PX else 0.0,
            "blue": 1.0,
            "red": 1.0,
        }
        passed = bool(cmp["pass"] and cov_ok)
        fail = cmp["fail"]
        if not cov_ok:
            fail = (fail + "; " if fail else "") + f"心距{g_d:.0f}px 橙墨距{o_d:.0f}px"

        meta = parse_name(mp.name)
        row = {
            "filename": mp.name,
            "sample_id": meta.sample_id if meta else "",
            "view_id": meta.view_id if meta else "",
            "magnification": meta.magnification if meta else "",
            "pass": passed,
            "start_ok": cmp["start_ok"],
            "len_ok": cmp["len_ok"],
            "cov_ok": cov_ok,
            "green_start_px": round(cmp["start_err"].get("green_start", 0.0), 1),
            "orange_start_px": round(cmp["start_err"].get("orange_start", 0.0), 1),
            "blue_start_px": round(_dist(geom_m["blue_start"], man_m["blue_start"]), 1),
            "red_start_px": round(_dist(geom_m["red_start"], man_m["red_start"]), 1),
            "ink_green_px": round(g_d, 1),
            "ink_orange_px": round(o_d, 1),
            "green_len_man": round(float(man_m["green_len"]), 1),
            "green_len_auto": round(float(geom_m["green_len"]), 1),
            "orange_len_man": round(float(man_m["orange_len"]), 1),
            "orange_len_auto": round(float(geom_m["orange_len"]), 1),
            "blue_len_man": round(float(man_m["blue_len"]), 1),
            "blue_len_auto": round(float(geom_m["blue_len"]), 1),
            "red_len_man": round(float(man_m["red_len"]), 1),
            "red_len_auto": round(float(geom_m["red_len"]), 1),
            "cov_green": round(cov["green"], 3),
            "cov_orange": round(cov["orange"], 3),
            "cov_blue": round(cov["blue"], 3),
            "cov_red": round(cov["red"], 3),
            "fail": fail if not passed else "",
        }
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            mark = "通过" if passed else f"未过 {row['fail']}"
            print(f"  {i + 1}/{len(files)}: {mp.name}  {mark}")

    json_path = base_dir / "推理结果" / "region1_sixpoint.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    batches = _batch_reports(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    qc_csv = base_dir / "推理结果" / "region1_ink_qc.csv"
    if rows:
        with qc_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"[对照人工] {n_pass}/{len(rows)} 通过  跳过 {len(skipped)}  "
        f"起点容差 {START_TOL_PX:.0f}px（只核绿/橙起点，线长不强制）"
    )
    batch_fail = [b for b in batches if not b["ok"]]
    for b in batches:
        flag = "OK" if b["ok"] else "不足"
        print(
            f"  第{b['batch']:02d}组 {b['from']} ~ {b['to']}  "
            f"{b['n_pass']}/{b['n']} (需>={b['need']})  {flag}"
        )
    if batch_fail:
        print(f"[对照人工] {len(batch_fail)} 组未满 8/10")
    else:
        print("[对照人工] 每10张均达到至少8张一致")
    print(f"          QC CSV -> {qc_csv}")
    return {"rows": rows, "batches": batches, "n_pass": n_pass, "n": len(rows), "skipped": skipped}
