"""Draw a client-facing example of the recommended region-1 annotation."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
from matplotlib.lines import Line2D

OUT = Path(__file__).with_name("region1_annotation_example.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False


def polar(cx, cy, r, deg):
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def draw_stem(ax, cx, cy):
    rings = [
        (1.00, "#c4a882", "皮层 / 树皮"),
        (0.86, "#8b5a3c", "韧皮部"),
        (0.78, "#d8b48a", "木质部"),
        (0.18, "#f3e6c8", "髓心"),
    ]
    for r, color, _ in rings:
        ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="#5a4632", lw=1.2, zorder=1))
    ax.add_patch(Circle((cx, cy), 0.06, facecolor="#efe4c4", edgecolor="#8a7048", lw=0.8, zorder=2))
    ax.text(cx, cy - 1.18, "横切面（示意）", ha="center", va="top", fontsize=11, color="#444")


def numbered_dot(ax, x, y, n, color, z=8):
    ax.add_patch(Circle((x, y), 0.055, facecolor="white", edgecolor=color, lw=2.0, zorder=z))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=9, fontweight="bold", color=color, zorder=z + 1)


def main():
    fig = plt.figure(figsize=(12.2, 6.6), facecolor="white")
    ax = fig.add_axes([0.02, 0.06, 0.52, 0.86])
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.38, 1.42)
    ax.axis("off")
    ax.set_title("第一区域推荐标注：点 6 个点（两条直线）", fontsize=14, pad=8, fontweight="bold")

    cx, cy = 0.0, 0.08
    draw_stem(ax, cx, cy)

    pith = (cx, cy)
    green_deg = 28
    g_end = polar(cx, cy, 1.00, green_deg)

    layer_deg = -42
    p0 = polar(cx, cy, 0.22, layer_deg)  # inner ring, not pith
    p_xy = polar(cx, cy, 0.78, layer_deg)
    p_ph = polar(cx, cy, 0.86, layer_deg)
    p_bk = polar(cx, cy, 1.00, layer_deg)

    ax.plot([pith[0], g_end[0]], [pith[1], g_end[1]], color="#0aa00a", lw=3.2, solid_capstyle="round", zorder=4)
    ax.plot([p0[0], p_xy[0]], [p0[1], p_xy[1]], color="#f47a12", lw=3.2, solid_capstyle="round", zorder=4)
    ax.plot([p_xy[0], p_ph[0]], [p_xy[1], p_ph[1]], color="#1e5ad8", lw=3.4, solid_capstyle="round", zorder=4)
    ax.plot([p_ph[0], p_bk[0]], [p_ph[1], p_bk[1]], color="#d21e1e", lw=3.4, solid_capstyle="round", zorder=4)

    numbered_dot(ax, *pith, 1, "#0aa00a")
    numbered_dot(ax, *g_end, 2, "#0aa00a")
    numbered_dot(ax, *p0, 3, "#f47a12")
    numbered_dot(ax, *p_xy, 4, "#1e5ad8")
    numbered_dot(ax, *p_ph, 5, "#d21e1e")
    numbered_dot(ax, *p_bk, 6, "#d21e1e")

    ax.annotate("绿线：只量半径\n从①髓心 到 ②树皮", xy=(0.55, 0.72), xytext=(0.72, 1.18),
                fontsize=9, color="#0a7a0a",
                arrowprops=dict(arrowstyle="->", color="#0a7a0a", lw=1.1),
                ha="left", va="bottom")
    ax.annotate("层线：③④⑤⑥必须共线\n橙=木质部  蓝=韧皮部  红=皮层",
                xy=(-0.15, -0.55), xytext=(-1.38, -1.22),
                fontsize=9, color="#8a3d00",
                arrowprops=dict(arrowstyle="->", color="#8a3d00", lw=1.1),
                ha="left", va="center")

    ax2 = fig.add_axes([0.55, 0.08, 0.43, 0.84])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.018",
                                 facecolor="#f7f7f4", edgecolor="#d0d0c8", lw=1.0))

    ax2.text(0.07, 0.93, "人怎么点（每张图一次）", fontsize=13, fontweight="bold", va="top")
    steps = [
        ("①", "#0aa00a", "点髓心", "切面最中心，不是最暗的洞"),
        ("②", "#0aa00a", "点绿线外端", "沿半径点到树皮外侧，软件自动连到①"),
        ("③", "#f47a12", "点层线内端", "点在木质部内侧 / 髓心环外，不要再从髓心画"),
        ("④", "#1e5ad8", "木质部 → 韧皮部", "只能在③→⑥这条线上滑动"),
        ("⑤", "#d21e1e", "韧皮部 → 皮层", "同样锁在这条直线上"),
        ("⑥", "#d21e1e", "点层线外端", "树皮外侧；③④⑤⑥必须成一条直尺"),
    ]
    y = 0.84
    for tag, color, title, note in steps:
        ax2.add_patch(Circle((0.12, y), 0.028, facecolor="white", edgecolor=color, lw=1.8))
        ax2.text(0.12, y, tag, ha="center", va="center", fontsize=8, fontweight="bold", color=color)
        ax2.text(0.18, y + 0.012, title, fontsize=10.5, fontweight="bold", va="center", color="#222")
        ax2.text(0.18, y - 0.028, note, fontsize=8.5, va="center", color="#666")
        y -= 0.105

    ax2.text(0.07, 0.22, "点完自动得到四个长度", fontsize=11, fontweight="bold")
    ax2.text(0.07, 0.155, "半径  =  ① → ②", fontsize=10, color="#0a7a0a")
    ax2.text(0.07, 0.105, "木质部  =  ③ → ④     韧皮部  =  ④ → ⑤     皮层  =  ⑤ → ⑥", fontsize=10, color="#333")
    ax2.text(0.07, 0.045, "存 JSON 坐标，不要把彩线画在 JPG 上当标签。", fontsize=9, color="#666")

    fig.savefig(OUT, dpi=160, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
