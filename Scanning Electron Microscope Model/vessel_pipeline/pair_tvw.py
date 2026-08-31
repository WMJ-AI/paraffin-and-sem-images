"""Adjacent vessel pairing and multi-normal TVW / CWR."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from segment import Vessel


@dataclass
class TVWLine:
    pair_id: str
    image_name: str
    line_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    length_px: float
    um_per_px: float
    tvw_um: float
    quantile: float


@dataclass
class VesselPair:
    pair_id: str
    image_name: str
    v1: Vessel
    v2: Vessel
    lines: list[TVWLine]
    tvw_median_um: float
    tvw_mean_um: float
    tvw_sd_um: float
    cwr1: float
    cwr2: float
    cwr_pair: float

    @property
    def tvw_primary_um(self) -> float:
        """Main TVW for CWR: mean of wall normals (prefer 3–5, allow 1–2)."""
        return self.tvw_mean_um


def _contour_points(cnt: np.ndarray) -> np.ndarray:
    return cnt.reshape(-1, 2).astype(np.float64)


def min_contour_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    if len(a) > 400:
        a = a[:: max(1, len(a) // 400)]
    if len(b) > 400:
        b = b[:: max(1, len(b) // 400)]
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    ia, ib = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return float(np.sqrt(d2[ia, ib])), int(ia), int(ib)


def shared_wall_interface(
    pts1: np.ndarray,
    pts2: np.ndarray,
    max_gap_px: float,
) -> tuple[np.ndarray, float, float] | None:
    """Return (ordered midpoints, frac_pts1_close, frac_pts2_close)."""
    s1 = pts1[:: max(1, len(pts1) // 300)]
    s2 = pts2[:: max(1, len(pts2) // 300)]
    d2 = ((s1[:, None, :] - s2[None, :, :]) ** 2).sum(axis=2)
    nn1 = d2.min(axis=1)
    nn2 = d2.min(axis=0)
    keep = nn1 <= max_gap_px**2
    if keep.sum() < 8:
        return None
    frac1 = float(keep.mean())
    frac2 = float((nn2 <= max_gap_px**2).mean())
    close1 = s1[keep]
    j = d2[keep].argmin(axis=1)
    close2 = s2[j]
    mids = 0.5 * (close1 + close2)
    mids_c = mids - mids.mean(axis=0)
    _, _, vt = np.linalg.svd(mids_c, full_matrices=False)
    axis = vt[0]
    order = np.argsort(mids_c @ axis)
    return mids[order], frac1, frac2


def _nearest_point(pts: np.ndarray, p: np.ndarray) -> np.ndarray:
    d2 = ((pts - p) ** 2).sum(axis=1)
    return pts[int(np.argmin(d2))]


def measure_tvw_lines(
    pts1: np.ndarray,
    pts2: np.ndarray,
    interface: np.ndarray,
    n_lines: int = 5,
    search_px: float = 40.0,
    min_len_px: float = 8.0,
) -> list[tuple[float, float, float, float, float, float]]:
    if len(interface) < 2:
        return []
    if np.linalg.norm(interface[-1] - interface[0]) < 1e-6:
        return []
    qs = np.linspace(0.25, 0.75, n_lines)
    lines = []
    for q in qs:
        idx = int(np.clip(round(q * (len(interface) - 1)), 0, len(interface) - 1))
        p = interface[idx]
        a = _nearest_point(pts1, p)
        b = _nearest_point(pts2, p)
        length = float(np.linalg.norm(a - b))
        if length < min_len_px or length > search_px:
            continue
        if idx == 0:
            tangent = interface[min(3, len(interface) - 1)] - interface[0]
        elif idx >= len(interface) - 1:
            tangent = interface[-1] - interface[max(0, len(interface) - 4)]
        else:
            tangent = interface[min(idx + 2, len(interface) - 1)] - interface[max(0, idx - 2)]
        tn = np.linalg.norm(tangent)
        if tn > 1e-6:
            tangent = tangent / tn
            normal = np.array([-tangent[1], tangent[0]])
            ab = (b - a) / (length + 1e-9)
            if abs(float(np.dot(ab, normal))) < 0.35:
                continue
        lines.append((float(a[0]), float(a[1]), float(b[0]), float(b[1]), length, float(q)))
    return lines


def measure_nearest_distance_lines(
    pts1: np.ndarray,
    pts2: np.ndarray,
    n_lines: int = 5,
    min_len_px: float = 3.0,
    max_len_px: float = 80.0,
    min_sep_px: float = 14.0,
    max_vs_min_gap: float = 1.2,
) -> list[tuple[float, float, float, float, float, float]]:
    """
    配对导管之间的 TVW 线（优先 3–5 条，不够则 1–2 条）：
    在两腔轮廓上找彼此距离最近的点对，取最短的若干条（沿共同壁拉开间距）。
    每条线长不得超过两导管最小间距的 max_vs_min_gap 倍（默认 120%）。
    返回 (x0,y0,x1,y1,length_px,quantile)。
    """
    if len(pts1) < 4 or len(pts2) < 4:
        return []

    step1 = max(1, len(pts1) // 500)
    step2 = max(1, len(pts2) // 500)
    s1 = pts1[::step1]
    s2 = pts2[::step2]
    d2 = ((s1[:, None, :] - s2[None, :, :]) ** 2).sum(axis=2)

    # 两导管真实最小间距（采样分辨率）
    min_gap_px = float(np.sqrt(d2.min()))
    if not np.isfinite(min_gap_px) or min_gap_px < 1e-6:
        return []
    # 硬上限：不得超过最小间距的 120%（可与调用方 max_len_px 取更严）
    cap_px = min(float(max_len_px), min_gap_px * float(max_vs_min_gap))
    lo_px = min(float(min_len_px), min_gap_px * 0.5)

    # 每个采样点到对侧轮廓的最近邻 → 候选最短弦
    cands: list[tuple[float, np.ndarray, np.ndarray]] = []
    jb = np.argmin(d2, axis=1)
    for i in range(len(s1)):
        j = int(jb[i])
        length = float(np.sqrt(d2[i, j]))
        if lo_px <= length <= cap_px:
            cands.append((length, s1[i].copy(), s2[j].copy()))
    ia = np.argmin(d2, axis=0)
    for j in range(len(s2)):
        i = int(ia[j])
        length = float(np.sqrt(d2[i, j]))
        if lo_px <= length <= cap_px:
            cands.append((length, s1[i].copy(), s2[j].copy()))

    if not cands:
        # 至少保留最短的一根（仍受 120% 约束）
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        length = float(np.sqrt(d2[i, j]))
        if length > cap_px * 1.01:
            return []
        cands = [(length, s1[int(i)].copy(), s2[int(j)].copy())]

    # 按距离升序：优先真正的最近壁间点对
    cands.sort(key=lambda c: c[0])

    # 去重（端点/中点过近）
    uniq: list[tuple[float, np.ndarray, np.ndarray]] = []
    for length, a, b in cands:
        mid = 0.5 * (a + b)
        if any(float(np.linalg.norm(mid - 0.5 * (u[1] + u[2]))) < 3.5 for u in uniq):
            continue
        if any(
            float(np.linalg.norm(a - u[1])) < 3.0 and float(np.linalg.norm(b - u[2])) < 3.0
            for u in uniq
        ):
            continue
        uniq.append((length, a, b))
    cands = uniq

    # 共同壁方向：用最短一批中点估计，便于沿壁拉开
    mids0 = np.array([0.5 * (a + b) for _, a, b in cands[: min(50, len(cands))]])
    tdir = np.array([1.0, 0.0])
    if len(mids0) >= 2:
        mc = mids0 - mids0.mean(axis=0)
        _, _, vt = np.linalg.svd(mc, full_matrices=False)
        tdir = vt[0]

    # 优先 3–5；约束下不够则接受 1–2
    want = int(np.clip(n_lines, 1, 5))

    def _pick(sep: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
        chosen: list[tuple[float, np.ndarray, np.ndarray]] = []
        for length, a, b in cands:  # already shortest-first
            mid = 0.5 * (a + b)
            if any(abs(float(np.dot(mid - 0.5 * (x[1] + x[2]), tdir))) < sep for x in chosen):
                continue
            if any(float(np.linalg.norm(mid - 0.5 * (x[1] + x[2]))) < sep * 0.6 for x in chosen):
                continue
            chosen.append((length, a, b))
            if len(chosen) >= want:
                break
        return chosen

    selected = _pick(min_sep_px)
    if len(selected) < min(3, want):
        selected = _pick(max(5.0, min_sep_px * 0.5))
    if len(selected) < min(3, want):
        selected = _pick(max(3.0, min_sep_px * 0.35))
    if len(selected) < 1:
        selected = cands[: min(want, len(cands))]
    if len(selected) > 5:
        selected = selected[:5]

    # 端点精修：在采样点邻域求局部壁间弦，避免互为全局最近点把多线收成同一根
    refined: list[tuple[float, np.ndarray, np.ndarray]] = []
    for _length, a, b in selected:
        # 局部：从各自端点找对侧最近点，取较短的那根弦（仍落在共同壁附近）
        b_from_a = _nearest_point(pts2, a)
        a_from_b = _nearest_point(pts1, b)
        L1 = float(np.linalg.norm(a - b_from_a))
        L2 = float(np.linalg.norm(a_from_b - b))
        if L1 <= L2:
            aa, bb, L = a, b_from_a, L1
        else:
            aa, bb, L = a_from_b, b, L2
        # 若局部弦超 120%，回退到原采样对；仍超则丢弃
        if L > cap_px:
            aa, bb = a, b
            L = float(np.linalg.norm(aa - bb))
        if L > cap_px or L < lo_px * 0.4:
            continue
        refined.append((L, aa, bb))
    # 中点过近去重（精修后可能粘连）
    dedup: list[tuple[float, np.ndarray, np.ndarray]] = []
    for L, aa, bb in refined:
        mid = 0.5 * (aa + bb)
        if any(float(np.linalg.norm(mid - 0.5 * (x[1] + x[2]))) < 4.0 for x in dedup):
            continue
        dedup.append((L, aa, bb))
    refined = dedup
    if not refined:
        # 最后保底：全分辨率最小间距那一根
        gmin, ia0, ib0 = min_contour_distance(pts1, pts2)
        if gmin <= cap_px * 1.01:
            refined = [(gmin, pts1[ia0], pts2[ib0])]
        else:
            return []

    # 沿共同壁排序，赋 quantile
    refined.sort(key=lambda c: float(np.dot(0.5 * (c[1] + c[2]), tdir)))
    n = len(refined)
    out = []
    for i, (length, a, b) in enumerate(refined):
        q = 0.0 if n == 1 else float(i / (n - 1))
        out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1]), float(length), q))
    return out


def lines_to_pair_fields(
    raw_lines: list[tuple[float, float, float, float, float, float]],
    v1: Vessel,
    v2: Vessel,
    um_per_px: float,
    pair_id: str,
    image_name: str,
) -> tuple[list[TVWLine], float, float, float, float, float, float]:
    """Build TVWLine list + stats/CWR from raw (x0,y0,x1,y1,len_px,q)."""
    lines: list[TVWLine] = []
    tvws = []
    for k, (x0, y0, x1, y1, length_px, q) in enumerate(raw_lines, start=1):
        tvw_um = float(length_px * um_per_px)
        tvws.append(tvw_um)
        lines.append(
            TVWLine(
                pair_id=pair_id,
                image_name=image_name,
                line_id=f"L{k}",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                length_px=length_px,
                um_per_px=um_per_px,
                tvw_um=tvw_um,
                quantile=q,
            )
        )
    tvws_arr = np.array(tvws, dtype=float)
    tvw_med = float(np.median(tvws_arr))
    tvw_mean = float(np.mean(tvws_arr))
    tvw_sd = float(np.std(tvws_arr, ddof=1)) if len(tvws_arr) > 1 else 0.0
    d1, d2 = v1.di_um or 1.0, v2.di_um or 1.0
    cwr1 = (tvw_mean / d1) ** 2
    cwr2 = (tvw_mean / d2) ** 2
    return lines, tvw_med, tvw_mean, tvw_sd, cwr1, cwr2, 0.5 * (cwr1 + cwr2)


def apply_nearest_tvw_lines(
    pairs: list[VesselPair],
    um_per_px: float,
    n_lines: int = 5,
    min_len_um: float = 0.5,
    max_len_um: float = 45.0,
    max_vs_min_gap: float = 1.2,
) -> list[VesselPair]:
    """用两导管轮廓间最近距离重写每对的 TVW 线（优先 3–5，不足可 1–2）。
    每条线长 ≤ 该对最小间距 × max_vs_min_gap（默认 120%）。
    """
    min_len_px = min_len_um / max(um_per_px, 1e-9)
    max_len_px = max_len_um / max(um_per_px, 1e-9)
    for pair in pairs:
        pts1 = _contour_points(pair.v1.contour)
        pts2 = _contour_points(pair.v2.contour)
        gmin, _, _ = min_contour_distance(pts1, pts2)
        # 对该对：全局 max 与「最小间距×120%」取更严
        pair_cap = min(max_len_px, gmin * max_vs_min_gap)
        raw = measure_nearest_distance_lines(
            pts1,
            pts2,
            n_lines=n_lines,
            min_len_px=min_len_px,
            max_len_px=pair_cap,
            min_sep_px=14.0,
            max_vs_min_gap=max_vs_min_gap,
        )
        if len(raw) < 3:
            raw = measure_nearest_distance_lines(
                pts1,
                pts2,
                n_lines=n_lines,
                min_len_px=min_len_px * 0.6,
                max_len_px=pair_cap,
                min_sep_px=8.0,
                max_vs_min_gap=max_vs_min_gap,
            )
        if len(raw) < 1:
            raw = measure_nearest_distance_lines(
                pts1,
                pts2,
                n_lines=max(1, min(2, n_lines)),
                min_len_px=min_len_px * 0.4,
                max_len_px=pair_cap,
                min_sep_px=5.0,
                max_vs_min_gap=max_vs_min_gap,
            )
        if not raw:
            # 放弃异常长跨度伪线（误配远处两腔时常见）
            if pair.lines and (
                len(pair.lines) == 1 and pair.lines[0].length_px > pair_cap * 1.5
            ):
                pair.lines = []
            continue
        # 再滤一次：杜绝精修后仍超 120% 的线
        raw = [ln for ln in raw if ln[4] <= pair_cap * 1.001]
        if not raw:
            continue
        if len(raw) > 5:
            idx = np.linspace(0, len(raw) - 1, 5).astype(int)
            raw = [raw[i] for i in idx]
        lines, med, mean, sd, c1, c2, cp = lines_to_pair_fields(
            raw, pair.v1, pair.v2, um_per_px, pair.pair_id, pair.image_name
        )
        pair.lines = lines
        pair.tvw_median_um = med
        pair.tvw_mean_um = mean
        pair.tvw_sd_um = sd
        pair.cwr1 = c1
        pair.cwr2 = c2
        pair.cwr_pair = cp
    return pairs


def find_pairs(
    vessels: list[Vessel],
    um_per_px: float,
    image_name: str,
    min_gap_um: float = 1.2,
    max_gap_um: float = 20.0,
    min_interface_points: int = 5,
    min_pair_ai_um2: float = 230.0,
    max_ai_ratio: float = 3.2,
    min_span_px: float = 40.0,
    min_tvw_um: float = 3.5,
    max_tvw_um: float = 22.0,
) -> list[VesselPair]:
    min_gap = min_gap_um / um_per_px
    max_gap = max_gap_um / um_per_px
    pairs: list[VesselPair] = []
    used: set[int] = set()
    gap_mat = np.full((len(vessels), len(vessels)), np.inf)
    inter_mat: dict[tuple[int, int], np.ndarray] = {}
    for i, v1 in enumerate(vessels):
        pts1 = _contour_points(v1.contour)
        for j, v2 in enumerate(vessels):
            if j <= i:
                continue
            pts2 = _contour_points(v2.contour)
            dist, _, _ = min_contour_distance(pts1, pts2)
            if dist < min_gap or dist > max_gap:
                continue
            got = shared_wall_interface(pts1, pts2, max_gap_px=max_gap * 1.15)
            if got is None:
                continue
            inter, frac1, frac2 = got
            if len(inter) < min_interface_points:
                continue
            if min(frac1, frac2) < 0.08:
                continue
            span = float(np.linalg.norm(inter[-1] - inter[0]))
            # 极近贴壁时共同壁弧天然偏短（如 41(5)），勿用固定 40px 一刀切
            span_need = min_span_px if dist > max(min_gap * 2.0, 10.0) else min(min_span_px, max(12.0, dist * 6.0))
            if span < span_need:
                continue
            assert v1.di_um is not None and v2.di_um is not None
            if min(v1.ai_um2 or 0, v2.ai_um2 or 0) < min_pair_ai_um2:
                continue
            if max(v1.ai_um2 or 0, v2.ai_um2 or 0) / max(min(v1.ai_um2 or 1, v2.ai_um2 or 1), 1e-6) > max_ai_ratio:
                continue
            expect = (v1.di_um + v2.di_um) / (2 * um_per_px) + dist
            center_dist = float(np.hypot(v1.cx - v2.cx, v1.cy - v2.cy))
            if center_dist > expect * 1.45 + 30:
                continue
            gap_mat[i, j] = dist
            gap_mat[j, i] = dist
            inter_mat[(i, j)] = inter
            inter_mat[(j, i)] = inter

    # 候选按「壁间距升序」贪心：三管邻近时最近两腔成双，另一腔为单
    cands = []
    for i in range(len(vessels)):
        for j in range(i + 1, len(vessels)):
            if not np.isfinite(gap_mat[i, j]):
                continue
            dist = float(gap_mat[i, j])
            inter = inter_mat[(i, j)]
            gap_um = dist * um_per_px
            score = len(inter) / (1.0 + abs(gap_um - 7.5) + dist * 0.05)
            cands.append((score, i, j, inter, dist))
    cands.sort(key=lambda x: (x[4], -x[0]))

    pair_idx = 1
    for _score, i, j, inter, dist in cands:
        if i in used or j in used:
            continue
        v1, v2 = vessels[i], vessels[j]
        pts1 = _contour_points(v1.contour)
        pts2 = _contour_points(v2.contour)
        # 主策略：两腔之间最短壁间距离；线长 ≤ 该对最小间距 × 120%
        pair_cap = dist * 1.2
        raw_lines = measure_nearest_distance_lines(
            pts1,
            pts2,
            n_lines=5,
            min_len_px=min_gap * 0.8,
            max_len_px=pair_cap,
            min_sep_px=14.0,
            max_vs_min_gap=1.2,
        )
        if len(raw_lines) < 3:
            raw_lines = measure_nearest_distance_lines(
                pts1,
                pts2,
                n_lines=5,
                min_len_px=min_gap * 0.5,
                max_len_px=pair_cap,
                min_sep_px=8.0,
                max_vs_min_gap=1.2,
            )
        if len(raw_lines) < 1:
            raw_lines = measure_tvw_lines(
                pts1, pts2, inter, n_lines=5, search_px=pair_cap, min_len_px=min_gap * 0.8
            )
            raw_lines = [ln for ln in raw_lines if ln[4] <= pair_cap * 1.001]
        if len(raw_lines) < 1:
            continue
        if len(raw_lines) > 5:
            idx = np.linspace(0, len(raw_lines) - 1, 5).astype(int)
            raw_lines = [raw_lines[i] for i in idx]

        tvws = [ln[4] * um_per_px for ln in raw_lines]
        tvw_med = float(np.median(tvws))
        tvw_mean = float(np.mean(tvws))
        if tvw_mean < min_tvw_um or tvw_mean > max_tvw_um:
            continue

        m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
        prefix = f"{m.group(1)}({m.group(2)})" if m else image_name
        pid = f"{prefix}-P{pair_idx:02d}"
        pair_idx += 1

        lines, tvw_med, tvw_mean, tvw_sd, cwr1, cwr2, cwr_pair = lines_to_pair_fields(
            raw_lines, v1, v2, um_per_px, pid, image_name
        )
        v1.pair_id = pid
        v2.pair_id = pid
        used.add(i)
        used.add(j)
        pairs.append(
            VesselPair(
                pair_id=pid,
                image_name=image_name,
                v1=v1,
                v2=v2,
                lines=lines,
                tvw_median_um=tvw_med,
                tvw_mean_um=tvw_mean,
                tvw_sd_um=tvw_sd,
                cwr1=cwr1,
                cwr2=cwr2,
                cwr_pair=cwr_pair,
            )
        )
    return pairs
