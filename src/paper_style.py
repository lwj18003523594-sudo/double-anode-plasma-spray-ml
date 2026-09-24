# -*- coding: utf-8 -*-
"""论文图统一样式（V1.3）

- 白色背景，禁止深色网页主题/渐变/阴影/3D 装饰；
- 刻度向内、字号统一；
- 英文图 Times New Roman，缺失自动回退 serif，不崩溃；
- 中文图中文用宋体系（Songti SC / SimSun / Noto Serif CJK SC），数字英文用 Times New Roman；
- 不下载/打包字体，只调用系统字体；
- 保存统一 bbox_inches="tight"，PNG 600 dpi，支持 SVG / PDF。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

EN_SERIF = ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "DejaVu Serif"]
CJK_SERIF = ["Songti SC", "STSong", "SimSun", "Noto Serif CJK SC", "Source Han Serif SC",
             "Songti TC", "Arial Unicode MS", "DejaVu Serif"]

BASE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
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
    """按图表语言设置 matplotlib rcParams。缺字体时 matplotlib 自动回退，不会崩溃。"""
    plt.rcParams.update(BASE_RC)
    if lang == "en":
        serif_chain = _installed(EN_SERIF) + _installed(CJK_SERIF)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = serif_chain or ["DejaVu Serif"]
    else:
        # 中文宋体 + 数字/英文 Times：font.family 列表在 matplotlib>=3.7 支持逐字符回退
        chain = _installed(["Times New Roman", "Times"]) + _installed(CJK_SERIF)
        plt.rcParams["font.family"] = chain or ["DejaVu Sans"]


def add_demo_watermark(fig, lang="zh"):
    """Demo 模式水印：右下角，低透明度，不遮挡数据。"""
    text = "DEMO DATA" if lang == "en" else "DEMO 数据"
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=13, color="#c0392b", alpha=0.55, style="italic", zorder=1000)


def save_figure(fig, stem, figures_dir, formats=("png", "svg"), lang="zh", demo=False):
    """保存论文图：PNG(600dpi)/SVG/PDF。返回保存路径列表。"""
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
