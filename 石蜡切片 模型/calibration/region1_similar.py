"""Before annotating, look up similar JSON labels and reuse their method.

Similarity uses: same magnification, same sample, downscaled thumbnail.
Method copied from the neighbor: ring / xylem / phloem fractions and the
signed angle between the green ray and the layer ray.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import LineGeometry
from calibration.io_util import imread, parse_name
from calibration.region1_geom import _nudge_seg_angle, angle_sep, valid_angle_pair
from calibration.region1_sixpoint import canonicalize, geom_from_six_points, load_relabel_records
from calibration.stem import detect_pith_center, estimate_green_angle, point_on_ray

THUMB = 16
INDEX_NAME = "region1_similar_index.json"
MIN_SCORE = 0.42


@dataclass
class SimilarHit:
    filename: str
    score: float
    magnification: int
    sample_id: int | None
    ring_frac: float
    xylem_frac: float
    phloem_frac: float
    green_angle_deg: float
    seg_angle_deg: float
    pith_nx: float
    pith_ny: float
    same_mag: bool
    same_sample: bool
    record: dict


def _thumb(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).reshape(-1)


def _angle_delta(green_deg: float, seg_deg: float) -> float:
    return (float(seg_deg) - float(green_deg) + 180.0) % 360.0 - 180.0


def index_path(base_dir: Path) -> Path:
    return base_dir / "推理结果" / INDEX_NAME


def build_index(base_dir: Path) -> list[dict]:
    """Build/save a lookup index from existing six-point JSON + 原图 thumbs."""
    records = load_relabel_records(base_dir)
    orig_dir = base_dir / "原图" / "第一区域"
    entries: list[dict] = []
    print(f"[相似检索] 为 {len(records)} 份 JSON 建索引...")
    for i, (name, rec) in enumerate(records.items()):
        orig = imread(orig_dir / name)
        if orig is None:
            continue
        meta = parse_name(name)
        h, w = orig.shape[:2]
        pith = rec["pith"]
        green = float(rec["green_angle_deg"])
        seg = float(rec["seg_angle_deg"])
        entries.append(
            {
                "filename": name,
                "sample_id": meta.sample_id if meta else None,
                "view_id": meta.view_id if meta else None,
                "magnification": int(meta.magnification) if meta else int(rec.get("magnification") or 0),
                "pith_nx": float(pith[0]) / max(w, 1),
                "pith_ny": float(pith[1]) / max(h, 1),
                "ring_frac": float(rec.get("ring_frac") or 0.21),
                "xylem_frac": float(rec.get("xylem_frac") or 0.90),
                "phloem_frac": float(rec.get("phloem_frac") or 0.96),
                "green_angle_deg": green,
                "seg_angle_deg": seg,
                "angle_sep_deg": angle_sep(green, seg),
                "thumb": [round(float(x), 4) for x in _thumb(orig)],
            }
        )
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(records)}")
    path = index_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_source": len(records), "entries": entries}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[相似检索] {len(entries)} 条 -> {path}")
    return entries


def load_index(base_dir: Path) -> list[dict]:
    path = index_path(base_dir)
    records = load_relabel_records(base_dir)
    entries: list[dict] | None = None
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if int(raw.get("n_source") or 0) == len(records):
                entries = list(raw.get("entries") or [])
        elif isinstance(raw, list) and len(raw) == len(records):
            entries = raw
    if entries is None:
        entries = build_index(base_dir)
    for e in entries:
        e["record"] = records.get(e["filename"], {})
    return entries


def _hit_from_entry(e: dict, score: float, meta) -> SimilarHit:
    mag = int(e.get("magnification") or 0)
    sid = e.get("sample_id")
    return SimilarHit(
        filename=e["filename"],
        score=float(score),
        magnification=mag,
        sample_id=sid,
        ring_frac=float(e.get("ring_frac") or 0.21),
        xylem_frac=float(e.get("xylem_frac") or 0.90),
        phloem_frac=float(e.get("phloem_frac") or 0.96),
        green_angle_deg=float(e.get("green_angle_deg") or 0.0),
        seg_angle_deg=float(e.get("seg_angle_deg") or 40.0),
        pith_nx=float(e.get("pith_nx") or 0.5),
        pith_ny=float(e.get("pith_ny") or 0.5),
        same_mag=bool(meta and mag == meta.magnification),
        same_sample=bool(meta and sid == meta.sample_id),
        record=e.get("record") or {},
    )


def find_similar(
    base_dir: Path,
    filename: str,
    img: np.ndarray,
    k: int = 3,
    index: list[dict] | None = None,
) -> list[SimilarHit]:
    """Return similar labeled images from JSON, best first."""
    if index is None:
        index = load_index(base_dir)
    if not index:
        return []
    meta = parse_name(filename)
    thumb = _thumb(img)
    scored: list[tuple[float, dict]] = []
    for e in index:
        if e.get("filename") == filename:
            continue
        score = 0.0
        emag = int(e.get("magnification") or 0)
        if meta is not None and emag == meta.magnification:
            score += 0.38
        elif meta is not None and emag:
            score -= 0.25
        if meta is not None and e.get("sample_id") == meta.sample_id:
            score += 0.28
        arr = np.array(e.get("thumb") or [], dtype=np.float32)
        if arr.size == thumb.size:
            mse = float(np.mean((arr - thumb) ** 2))
            score += 0.34 * max(0.0, 1.0 - mse / 0.07)
        scored.append((score, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [_hit_from_entry(e, s, meta) for s, e in scored[:k] if s >= MIN_SCORE]


def apply_similar_method(
    img: np.ndarray,
    stem_mask: np.ndarray,
    mag: int,
    hit: SimilarHit,
    yolo_geom: LineGeometry | None = None,
) -> LineGeometry | None:
    """Draw this image using the neighbor's ring fractions and ray-angle gap."""
    h, w = img.shape[:2]
    if yolo_geom is not None:
        pith = yolo_geom.center
        green_ang = float(yolo_geom.green_angle_deg)
        trust = True
    else:
        pith = detect_pith_center(img, stem_mask, mag=mag)
        blend = 0.55 if hit.same_sample and hit.same_mag else (0.25 if hit.same_mag else 0.0)
        if blend > 0:
            npith = (hit.pith_nx * w, hit.pith_ny * h)
            cand = ((1.0 - blend) * pith[0] + blend * npith[0], (1.0 - blend) * pith[1] + blend * npith[1])
            ix, iy = int(round(cand[0])), int(round(cand[1]))
            if 0 <= ix < w and 0 <= iy < h and stem_mask[iy, ix] > 0:
                pith = cand
        green_ang = estimate_green_angle(img, stem_mask, pith)
        trust = False

    seg_ang = green_ang + _angle_delta(hit.green_angle_deg, hit.seg_angle_deg)
    if not valid_angle_pair(green_ang, seg_ang):
        seg_ang = _nudge_seg_angle(green_ang, seg_ang)

    dummy = 1000.0
    hint = geom_from_six_points(
        pith,
        point_on_ray(pith, green_ang, dummy),
        point_on_ray(pith, seg_ang, dummy * hit.ring_frac),
        point_on_ray(pith, seg_ang, dummy * hit.xylem_frac),
        point_on_ray(pith, seg_ang, dummy * hit.phloem_frac),
        point_on_ray(pith, seg_ang, dummy),
    )
    return canonicalize(
        img,
        stem_mask,
        hint=hint,
        mag=mag,
        ring_frac=hit.ring_frac,
        trust_hint_pith=trust,
    )
