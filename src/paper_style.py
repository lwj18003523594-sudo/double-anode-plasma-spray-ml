# -*- coding: utf-8 -*-
"""论文图统一样式（V1.3 → V1.6 Nature 版式升级）

- 白色背景，禁止深色网页主题/渐变/阴影/3D 装饰；
- 刻度向内、字号统一；
- 英文图 Times New Roman，缺失自动回退 serif，不崩溃；
- 中文图中文用宋体系（Songti SC / SimSun / Noto Serif CJK SC），数字英文用 Times New Roman；
- 不下载/打包字体，只调用系统字体；
- 保存统一 bbox_inches="tight"，PNG 600 dpi，支持 SVG / PDF。

V1.6 变更（DESIGN_V1.6 §5.1，P0-5）：
- 色板改为 import plot_style 同源常量——本文件不再出现任何 hex 字面量；
- 新增 Nature 版式常量与 helper（单栏 89mm / 双栏 183mm / 中幅 120mm）；
- apply_paper_style 增强：spines 收窄为 0.8pt 中性灰、轴/刻度字色统一；
- 新增 add_stat_box 统一注脚框（替代角落图例，直接标注 n/R²/RMSE/CV）；
- Demo 水印换 PINK_ACCENT（样式不变）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# V1.6：色板同源——所有 hex 只定义于 src/plot_style.py，本文件一律 import
from .plot_style import (GRAY_INK, GRAY_MUTED, GRAY_REF, GRAY_BORDER,
                         BLUE_SIGNAL, BLUE_MID, BLUE_LIGHT, PINK_ACCENT, PINK_MID,
                         SEM_GOOD, SEM_BAD, QUAL_CYCLE, blue_ramp)  # noqa: F401（部分常量供 paper_output 复用 re-export）

EN_SERIF = ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "DejaVu Serif"]
CJK_SERIF = ["Songti SC", "STSong", "SimSun", "Noto Serif CJK SC", "Source Han Serif SC",
             "Songti TC", "Arial Unicode MS", "DejaVu Serif"]

# ---------------- Nature 版式常量（89 / 183mm 已拍板） ----------------
MM = 1 / 25.4                       # mm → inch
SINGLE_COL_IN = 89 * MM             # ≈ 3.50 in（Nature 单栏）
DOUBLE_COL_IN = 183 * MM            # ≈ 7.20 in（Nature 双栏）
ONE_HALF_COL_IN = 120 * MM          # 中幅（1.33 col）用于 A2/H 系多面板


def fig_single(h, /) -> tuple[float, float]:
    """单栏图尺寸（宽锁定 89mm，高度由调用方按内容定）。"""
    return (SINGLE_COL_IN, float(h))


def fig_double(h, /) -> tuple[float, float]:
    """双栏图尺寸（宽锁定 183mm）。"""
    return (DOUBLE_COL_IN, float(h))


def fig_mid(h, /) -> tuple[float, float]:
    """中幅图尺寸（宽 120mm，用于 R² 总览 / 不确定性等）。"""
    return (ONE_HALF_COL_IN, float(h))


BASE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": GRAY_REF,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.labelcolor": GRAY_INK,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "xtick.color": GRAY_MUTED,
    "ytick.color": GRAY_MUTED,
    "legend.fontsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": False,
    "axes.grid": False,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "legend.frameon": False,
    "axes.unicode_minus": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 600,
}


def _installed(names):
    """过滤出 matplotlib 实际识别的字体名（不崩溃，缺失自动回退）。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in names if n in installed]


def apply_paper_style(lang="zh"):
    """按图表语言设置 matplotlib rcParams。缺字体时 matplotlib 自动回退，不会崩溃。

    V1.6：BASE_RC 增强——顶/右 spines 收窄为 0.8pt 中性灰、轴标签/刻度字色统一。
    """
    plt.rcParams.update(BASE_RC)
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = False
    if lang == "en":
        serif_chain = _installed(EN_SERIF) + _installed(CJK_SERIF)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = serif_chain or ["DejaVu Serif"]
    else:
        # 中文宋体 + 数字/英文 Times：font.family 列表在 matplotlib>=3.7 支持逐字符回退
        chain = _installed(["Times New Roman", "Times"]) + _installed(CJK_SERIF)
        plt.rcParams["font.family"] = chain or ["DejaVu Sans"]


def add_stat_box(ax, note: str, *, loc="upper left", fontsize: float = 8) -> None:
    """统一注脚框：parity/residual 的 n/R²/RMSE/CV 标注。

    白底、GRAY_BORDER 边 0.6pt、GRAY_INK 字（默认 8pt）——直接标注替代角落图例
    （P0-5 验收 2）。note 为空串时不绘制。fontsize 可按图类型覆盖（如 9）。
    """
    if not note:
        return
    pos = {"upper left": (0.03, 0.97, "top", "left"),
           "upper right": (0.97, 0.97, "top", "right"),
           "lower left": (0.03, 0.03, "bottom", "left"),
           "lower right": (0.97, 0.03, "bottom", "right")}.get(loc, (0.03, 0.97, "top", "left"))
    ax.text(pos[0], pos[1], note, transform=ax.transAxes,
            va=pos[2], ha=pos[3], fontsize=fontsize, color=GRAY_INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=GRAY_BORDER, lw=0.6, alpha=0.9))


def add_demo_watermark(fig, lang="zh"):
    """Demo 模式水印：右下角，低透明度，不遮挡数据。V1.6：换 PINK_ACCENT 家族色。"""
    text = "DEMO DATA" if lang == "en" else "DEMO 数据"
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=13, color=PINK_ACCENT, alpha=0.55, style="italic", zorder=1000)


def save_figure(fig, stem, figures_dir, formats=("png", "svg"), lang="zh", demo=False):
    """保存论文图：PNG(600dpi)/SVG/PDF。返回保存路径列表。（V1.6 签名与行为不变）"""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if demo:
        add_demo_watermark(fig, lang)
    saved = []
    fmts = []
    for f in formats:
        if f == "png_svg":
            fmts += ["png", "svg"]
        else:
            fmts.append(f)
    for fmt in fmts:
        if fmt not in ("png", "svg", "pdf"):
            continue
        path = figures_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        saved.append(path)
    plt.close(fig)
    return saved
