# -*- coding: utf-8 -*-
"""V1.6 网页图表统一样式模块（Plotly Nature 模板 + 全站唯一色板定义处）。

定位（DESIGN_V1.6 §2.1）：
- 全平台网页图表（Plotly）样式唯一来源；
- matplotlib 侧（paper_style.py / paper_output.py）的色板常量唯一来源；
- 本模块顶层【不 import streamlit】（仅函数内惰性 import），保证被 paper_style 安全引用；
- 所有 hex 颜色只允许定义在本文件（verify_v16.py 扫描兜底）。

色板纪律：
- 中性灰 = 文字/网格/参考线；信号蓝 = 主方法/数据系列；强调粉 ≤2 处/页；
- 红/绿（SEM_*）仅用于方向性语义（涨跌/优劣），禁止装饰；
- 分类循环色 QUAL_CYCLE ≤6 类、兼容色弱，禁 plotly/seaborn 默认鲜艳色。
"""
from __future__ import annotations

# ============ 中性灰家族（文字/网格/参考线） ============
GRAY_INK    = "#2F3440"   # 正文主文字
GRAY_MUTED  = "#7A8290"   # 辅助文字 / 轴标签 / caption
GRAY_BORDER = "#E5E9EF"   # 边框 / 分隔线 / 卡片描边
GRAY_PAPER  = "#FAFBFC"   # 页面底色
GRAY_REF    = "#98A2B3"   # 参考线 / 1:1 虚线 / 网格（低饱和蓝灰）

# ============ 信号蓝家族（主方法 / 数据系列 / 主色） ============
BLUE_SIGNAL = "#3C6E9F"   # 图表主系列
BLUE_MID    = "#6B93BF"   # 第二系列 / 次级数据
BLUE_LIGHT  = "#A9C9E5"   # 淡填充 / 趋势带
BLUE_FAINT  = "#EAF3FB"   # 按钮底 / 选中态

# ============ 强调粉家族（智能按键实底 / 强调标注，≤2 处/页） ============
PINK_ACCENT = "#C4707F"   # 智能按键实底 / 强调
PINK_MID    = "#DB9AA6"   # 次级强调 / 徽标边
PINK_LIGHT  = "#E8B8C0"   # obj-card / 淡强调

# ============ 方向色（仅涨跌/优劣语义，禁止装饰） ============
SEM_GOOD = "#6FA57C"      # 低饱和绿（优 / 达标）
SEM_BAD  = "#C47070"      # 低饱和红（劣 / 超限）

# ============ 分类循环色（≤6 类，兼容色弱，禁 plotly/seaborn 默认） ============
QUAL_CYCLE = ["#3C6E9F", "#8A93A3", "#C4707F", "#6B93BF", "#A98F96", "#4C5A6E"]

WEB_FONT = 'Times New Roman, "Songti SC", SimSun, STSong, "Noto Serif CJK SC", serif'

# hero panel 布局约定（P0-4 验收 4）：主图统一高度
HERO_HEIGHT = 460   # 每个 Tab 主面板最低高度
SUB_HEIGHT  = 340   # 次级图（expander 内）高度


# ---------------- 色板工具 ----------------
def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (r, g, b)。"""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def blue_ramp(n: int) -> list[str]:
    """顺序/连续系列：BLUE_SIGNAL → 蓝灰等距插值 n 档（熔融类别、Stage 分级等）。

    n ≤ 1 时返回 [BLUE_SIGNAL]；n 越大插值越细。
    """
    n = int(n)
    if n <= 1:
        return [BLUE_SIGNAL]
    anchor_rgb = _hex_to_rgb(BLUE_SIGNAL)
    end_rgb = _hex_to_rgb("#4C5A6E")  # 蓝灰端点（同 QUAL_CYCLE 末位，观感连续）
    out = []
    for i in range(n):
        t = i / (n - 1)
        rgb = tuple(round(a + (b - a) * t) for a, b in zip(anchor_rgb, end_rgb))
        out.append(_rgb_to_hex(rgb))
    return out


# ---------------- Plotly Nature 模板 ----------------
def register_nature_template() -> None:
    """注册并设为默认 Plotly template "nature"。

    白底、WEB_FONT、GRAY_REF 网格、无鲜艳循环色；
    替代 app.py V1.5 的手工 template 设置（L36–42）。
    """
    import plotly.io as pio

    base = pio.templates["plotly_white"]
    tpl = base
    tpl.layout.font.family = WEB_FONT
    tpl.layout.font.size = 13
    tpl.layout.font.color = GRAY_INK
    tpl.layout.title.font.size = 17
    tpl.layout.title.font.color = GRAY_INK
    tpl.layout.colorway = list(QUAL_CYCLE)
    tpl.layout.paper_bgcolor = "white"
    tpl.layout.plot_bgcolor = "white"
    tpl.layout.xaxis.gridcolor = GRAY_BORDER
    tpl.layout.yaxis.gridcolor = GRAY_BORDER
    tpl.layout.xaxis.zerolinecolor = GRAY_REF
    tpl.layout.yaxis.zerolinecolor = GRAY_REF
    tpl.layout.xaxis.linecolor = GRAY_MUTED
    tpl.layout.yaxis.linecolor = GRAY_MUTED
    tpl.layout.legend.font.size = 12
    pio.templates["nature"] = tpl
    pio.templates.default = "nature"


def nature_layout(fig, *, title=None, height=None, width=None,
                  margin=None, legend="top", showlegend=None):
    """统一布局入口：白底、标题字号 15–19、边距规范、legend 横向置顶无框。

    返回同一 fig（可链式调用）。margin 缺省取 l=10, r=30, t=45, b=10 家族。
    """
    if title is not None:
        fig.update_layout(title=dict(text=title, font=dict(size=16)))
    layout = {}
    if height is not None:
        layout["height"] = height
    if width is not None:
        layout["width"] = width
    layout["margin"] = margin or dict(l=10, r=30, t=45, b=10)
    layout["paper_bgcolor"] = "white"
    layout["plot_bgcolor"] = "white"
    if showlegend is not None:
        layout["showlegend"] = showlegend
    if legend == "top":
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.0,
                                xanchor="right", x=1.0,
                                bgcolor="rgba(0,0,0,0)", borderwidth=0)
    elif legend is None:
        layout["showlegend"] = False
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRAY_BORDER, zerolinecolor=GRAY_REF, linecolor=GRAY_MUTED)
    fig.update_yaxes(gridcolor=GRAY_BORDER, zerolinecolor=GRAY_REF, linecolor=GRAY_MUTED)
    return fig


def apply_series_colors(fig, series_names=None, family="qual"):
    """按语义分配 color_discrete_map。

    family="qual"：QUAL_CYCLE 循环；family="blue"：blue_ramp(len(series)) 插值。
    series_names 为 None 时按 fig.data 中的 name 收集。
    """
    if series_names is None:
        names = []
        for tr in fig.data:
            nm = getattr(tr, "name", None)
            if nm and nm not in names:
                names.append(nm)
        series_names = names
    series_names = list(series_names or [])
    if not series_names:
        return fig
    if family == "blue":
        cycle = blue_ramp(len(series_names))
    else:
        cycle = QUAL_CYCLE
    mapping = {nm: cycle[i % len(cycle)] for i, nm in enumerate(series_names)}
    fig.update_layout(coloraxis=None) if False else None
    for tr in fig.data:
        nm = getattr(tr, "name", None)
        if nm in mapping:
            tr.marker.color = mapping[nm]
    # 记录映射，供 px 图 color 维度使用（plotly express 场景由调用方传 color_discrete_map）
    fig._nature_color_map = mapping  # noqa: SLF001（轻量约定，非公开 API）
    return fig


def discrete_map(series_names, family="qual") -> dict:
    """返回 {系列名: 颜色} 映射（供 px.* 的 color_discrete_map 参数直接使用）。"""
    series_names = list(series_names or [])
    if not series_names:
        return {}
    if family == "blue":
        cycle = blue_ramp(len(series_names))
    else:
        cycle = QUAL_CYCLE
    return {nm: cycle[i % len(cycle)] for i, nm in enumerate(series_names)}


# ---------------- 统一图入口 ----------------
def parity_chart(df, *, x, y, x_label, y_label, title=None, identity=True):
    """网页 parity 统一入口：散点 BLUE_SIGNAL、1:1 虚线 GRAY_REF、轴等比例。

    df 需含 x/y 两列（数值）。identity=True 时叠加 y=x 参考虚线。
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], mode="markers",
        marker=dict(color=BLUE_SIGNAL, size=9, opacity=0.75),
        name="数据点", showlegend=False))
    if identity:
        xs = [float(v) for v in df[x].dropna()]
        ys = [float(v) for v in df[y].dropna()]
        if xs and ys:
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            pad = (hi - lo) * 0.05 if hi > lo else 1.0
            fig.add_trace(go.Scatter(
                x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
                mode="lines", line=dict(dash="dash", color=GRAY_REF, width=1.2),
                name="y = x", showlegend=False))
            fig.update_xaxes(range=[lo - pad, hi + pad])
            fig.update_yaxes(range=[lo - pad, hi + pad], scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return nature_layout(fig, title=title, legend=None)


def importance_chart(imp_df, *, title, x_label="特征重要性"):
    """横向条形图统一入口：BLUE_SIGNAL 单色、条端直接标注数值（替代图例）。

    imp_df 需含两列：[名称列, 数值列]（按数值升序排列最佳，顶部为最重要）。
    """
    import plotly.graph_objects as go

    cols = list(imp_df.columns)
    name_col, val_col = cols[0], cols[1]
    df = imp_df.sort_values(val_col)
    fig = go.Figure(go.Bar(
        x=df[val_col], y=df[name_col], orientation="h",
        marker_color=BLUE_SIGNAL,
        text=[f"{float(v):.3f}" for v in df[val_col]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>重要性=%{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text=x_label)
    return nature_layout(fig, title=title, legend=None,
                         height=max(SUB_HEIGHT, 30 * len(df) + 90))


# ---------------- 统计诚信注脚 ----------------
def _fmt_stat(key: str, value) -> str:
    """注脚数值格式：n 取整；R²/MAE/F1 保留 3 位；RMSE/MAE_f1 用 3 位有效数字。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{key}={value}"
    import math
    if math.isnan(v):
        return f"{key}=NaN"
    if key == "n":
        return f"n={int(round(v))}"
    if key in ("r2", "mae", "mae_f1"):
        return f"{key.upper().replace('MAE_F1', 'MAE_F1')}={v:.3f}".replace("R2", "R²")
    return f"{key.upper()}={v:.3g}"


def stat_note(*, n=None, r2=None, rmse=None, mae=None, mae_f1=None,
              cv_method=None, errbar=None, extra=None) -> str:
    """统计诚信注脚文本：拼出 'n=45 ｜ R²=0.874 ｜ RMSE=0.213 ｜ CV: GroupKFold(5)'。

    全部缺失时返回空串。数值一律由调用方传入，本函数不计算。
    """
    parts = []
    for key, val in [("n", n), ("r2", r2), ("rmse", rmse), ("mae", mae), ("mae_f1", mae_f1)]:
        if val is None:
            continue
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                continue
        except Exception:
            pass
        parts.append(_fmt_stat(key, val))
    if cv_method:
        parts.append(f"CV: {cv_method}")
    if errbar:
        parts.append(f"误差棒: {errbar}")
    if extra:
        parts.append(str(extra))
    return " ｜ ".join(parts)


def add_stat_note(fig, note: str, *, position="bottom"):
    """把 stat_note 文本以 caption 注脚加到图内（P0-4 验收 2 的强制项）。

    position="bottom" → 图内左下角；"top" → 图内左上角。空串直接返回。
    """
    if not note:
        return fig
    import plotly.graph_objects as go

    if position == "top":
        y, yanchor = 1.0, "top"
    else:
        y, yanchor = 0.0, "bottom"
    fig.add_annotation(
        text=note, xref="paper", yref="paper", x=0.0, y=y,
        xanchor="left", yanchor=yanchor, showarrow=False,
        font=dict(size=11.5, color=GRAY_MUTED),
        bgcolor="rgba(255,255,255,0.75)", bordercolor=GRAY_BORDER,
        borderwidth=1, borderpad=3)
    return fig


def add_demo_watermark(fig):
    """Demo 水印：右上角「演示数据 DEMO」，PINK_ACCENT 半透明，不遮挡数据。

    真实数据模式一律不调用（P1-5）。
    """
    fig.add_annotation(
        text="演示数据 DEMO", xref="paper", yref="paper", x=1.0, y=1.0,
        xanchor="right", yanchor="top", showarrow=False,
        font=dict(size=12, color=PINK_ACCENT),
        opacity=0.7, bgcolor="rgba(255,255,255,0.6)")
    return fig


# ---------------- 下载行（P1-4 统一下载入口） ----------------
def fig_download_row(fig, data_df, stem: str, *, key: str) -> None:
    """图下方渲染『下载原图 PNG ｜ 源数据 CSV』两个 st.download_button。

    PNG 用 fig.to_image(format="png", scale=2)（需 kaleido）；
    kaleido 不可用时降级为仅 CSV 并 caption 说明。
    内部惰性 import streamlit（本模块顶层禁止依赖 streamlit）。
    """
    import streamlit as st  # 惰性 import（模块顶层不依赖 streamlit）

    stem = str(stem or "figure").replace("/", "_")
    c_png, c_csv = st.columns(2)
    png_ok, png_bytes = False, b""
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        png_ok = True
    except Exception:
        png_ok = False
    if png_ok:
        c_png.download_button("下载原图 PNG", png_bytes,
                              file_name=f"{stem}.png", mime="image/png",
                              key=f"{key}_png", use_container_width=True)
    else:
        c_png.caption("PNG 导出不可用（缺 kaleido）；可用图右上角相机按钮导出。")
    try:
        csv_bytes = data_df.to_csv(index=False).encode("utf-8-sig")
        c_csv.download_button("下载源数据 CSV", csv_bytes,
                              file_name=f"{stem}.csv", mime="text/csv",
                              key=f"{key}_csv", use_container_width=True)
    except Exception:
        c_csv.caption("源数据导出失败。")
