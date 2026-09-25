# -*- coding: utf-8 -*-
"""V1.6 全程可溯源机制核心（Traceable Evidence，P0-2）。

定位：证据链在【文本生成处】落地，不是事后包装——
EvidenceRegistry 随 quick_analysis / insights / figure_analysis 的生成流程创建与传递。

铁律（DESIGN_V1.6 §7.4 证据生成纪律）：
- 取值即拼接：结论文本只允许 registry.statement() 产出，模板 format(**values)；
- 缺证降级：required 任一键缺失 / NaN → 返回 DEGRADE_TEXT，绝不输出弱化套话；
- model 类结论必须带 CV 方式限定（statement 内置检查，缺 cv_method → 降级）；
- Demo 前缀「[演示数据] 」由 registry 统一加，不在调用方手写；
- manifest 落盘 runs/quick_analysis/<run_id>/evidence_manifest.json，随 ZIP 打包。

本模块顶层【不 import streamlit】（UI 组件惰性 import），与 plot_style 同策略。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# 三源分级标签（沿用 V1.5 insights 三源分级口径）
SOURCE_CLASS_LABEL = {
    "direct": "[直接数据]",   # n/mean/std/CV%/min–max，来自事实行（insights._fact_df）
    "model":  "[模型推断]",   # R²/RMSE/MAE/CV 方式与折数/OOF/SHAP/FI
    "meta":   "[数据来源]",   # run_id/数据版本 hash/模型版本/Demo 标记
}

DEGRADE_TEXT = "当前数据不足以支持该判断"
DEMO_PREFIX = "[演示数据] "

MANIFEST_NAME = "evidence_manifest.json"


@dataclass
class EvidenceItem:
    """单条证据：结论句 + 引用数值 + 计算与来源信息。"""
    eid: str                    # 证据 ID，run 内唯一（<模块前缀>.<语义 slug>）
    text: str                   # 结论句（已格式化；降级时 text=DEGRADE_TEXT）
    source_class: str           # "direct" | "model" | "meta"
    values: dict = field(default_factory=dict)   # 引用的全部数值（QA 抽查对照字段）
    computed_by: str = ""       # 计算入口，如 "insights.coverage_stats"
    data_version: str | None = None              # "auto_xxxx"
    model_version: str | None = None
    rows_filter: str = ""       # 如 "data_origin ∈ {experimental, literature, cfd}"
    files: list = field(default_factory=list)    # 来源文件相对路径
    demo: bool = False
    degraded: bool = False      # 是否缺证降级
    created_at: str = ""        # ISO 8601


class EvidenceRegistry:
    """证据注册表：run 内唯一；statement() 是结论文本唯一合法出口。"""

    def __init__(self, run_id: str, run_dir=None, *, demo: bool = False,
                 data_version: str | None = None, model_version: str | None = None):
        self.run_id = str(run_id)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.demo = bool(demo)
        self.data_version = data_version
        self.model_version = model_version
        self._items: dict[str, EvidenceItem] = {}

    # ---------------- 核心：取值即拼接 + 缺证降级 ----------------
    def statement(self, eid: str, template: str, *, source_class: str,
                  computed_by: str, values: dict, required=(),
                  rows_filter: str = "", files=None) -> str:
        """生成结论句（唯一合法出口）。

        - template.format(**values) 成功且 required 全部存在且非 NaN → 返回格式化句子；
        - required 任一缺失 / NaN → 返回 DEGRADE_TEXT（注册 degraded=True 的 item）；
        - source_class="model" 且 values 缺 "cv_method" → 降级（模型推断必须带 CV 限定）；
        - demo=True → 返回文本加 DEMO_PREFIX；
        - 成功则注册 EvidenceItem 并返回句子。
        """
        values = dict(values or {})
        required = tuple(required or ())
        degraded = False
        reason = ""

        missing = [k for k in required if k not in values or self._is_nan(values[k])]
        if missing:
            degraded, reason = True, f"缺失 required 度量: {missing}"
        elif source_class == "model" and not values.get("cv_method"):
            degraded, reason = True, "model 类结论缺少 cv_method 限定"

        if degraded:
            item = EvidenceItem(
                eid=eid, text=(DEMO_PREFIX if self.demo else "") + DEGRADE_TEXT,
                source_class=source_class,
                values={"__degrade_reason": reason, **{k: values.get(k) for k in missing}},
                computed_by=computed_by, data_version=self.data_version,
                model_version=self.model_version, rows_filter=rows_filter,
                files=list(files or []), demo=self.demo, degraded=True,
                created_at=datetime.now().isoformat(timespec="seconds"))
            self.register(item)
            return item.text

        try:
            text = template.format(**values)
        except (KeyError, IndexError, ValueError) as e:
            # 模板引用了未提供的键 → 同样按缺证降级处理，绝不输出半截句子
            item = EvidenceItem(
                eid=eid, text=(DEMO_PREFIX if self.demo else "") + DEGRADE_TEXT,
                source_class=source_class,
                values={"__degrade_reason": f"模板格式化失败: {e}"},
                computed_by=computed_by, data_version=self.data_version,
                model_version=self.model_version, rows_filter=rows_filter,
                files=list(files or []), demo=self.demo, degraded=True,
                created_at=datetime.now().isoformat(timespec="seconds"))
            self.register(item)
            return item.text

        # manifest 中落盘的 text 与界面展示句完全一致（含 Demo 前缀），保证三层对齐：
        # 结论句（UI）↔ manifest text（落盘）↔ values（QA 抽查）
        full_text = (DEMO_PREFIX if self.demo else "") + text
        item = EvidenceItem(
            eid=eid, text=full_text, source_class=source_class,
            values={k: v for k, v in values.items()},
            computed_by=computed_by, data_version=self.data_version,
            model_version=self.model_version, rows_filter=rows_filter,
            files=list(files or []), demo=self.demo, degraded=False,
            created_at=datetime.now().isoformat(timespec="seconds"))
        self.register(item)
        return full_text

    @staticmethod
    def _is_nan(v) -> bool:
        """None / NaN 判定（字符串等非数值类型不算 NaN）。"""
        if v is None:
            return True
        if isinstance(v, float) and math.isnan(v):
            return True
        return False

    # ---------------- 注册表操作 ----------------
    def register(self, item: EvidenceItem) -> None:
        self._items[item.eid] = item

    def get(self, eid: str) -> EvidenceItem | None:
        return self._items.get(eid)

    def items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    # ---------------- manifest 读写 ----------------
    def save(self, run_dir=None) -> Path:
        """写 evidence_manifest.json：{eid: {values, computed_by, data_version,
        model_version, rows_filter, files, text, source_class, demo, degraded, created_at}}。"""
        target = Path(run_dir) if run_dir is not None else self.run_dir
        if target is None:
            raise ValueError("save() 需要 run_dir（构造时传入或调用时指定）")
        target.mkdir(parents=True, exist_ok=True)
        payload = {item.eid: {
            "text": item.text,
            "source_class": item.source_class,
            "values": item.values,
            "computed_by": item.computed_by,
            "data_version": item.data_version,
            "model_version": item.model_version,
            "rows_filter": item.rows_filter,
            "files": list(item.files or []),
            "demo": item.demo,
            "degraded": item.degraded,
            "created_at": item.created_at,
        } for item in self._items.values()}
        path = Path(target) / MANIFEST_NAME
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, run_dir) -> "EvidenceRegistry | None":
        """读 manifest；不存在/损坏 → None（调用方走旧 run 兼容提示）。"""
        path = Path(run_dir) / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        reg = cls(run_id=Path(run_dir).name, run_dir=Path(run_dir),
                  demo=False)
        for eid, info in payload.items():
            info = info or {}
            reg.register(EvidenceItem(
                eid=str(eid), text=str(info.get("text", "")),
                source_class=str(info.get("source_class", "meta")),
                values=dict(info.get("values") or {}),
                computed_by=str(info.get("computed_by", "")),
                data_version=info.get("data_version"),
                model_version=info.get("model_version"),
                rows_filter=str(info.get("rows_filter", "")),
                files=list(info.get("files") or []),
                demo=bool(info.get("demo", False)),
                degraded=bool(info.get("degraded", False)),
                created_at=str(info.get("created_at", ""))))
        return reg


# ---------------- Streamlit UI 组件（惰性 import） ----------------
_BADGE_STYLE = {
    # source_class → (底色, 字色, 边色)；demo 一律覆盖为粉家族
    "direct": ("var(--blue-faint, #EAF3FB)", "var(--blue-signal, #3C6E9F)", "var(--blue-light, #A9C9E5)"),
    "model":  ("#FFFFFF", "var(--gray-ink, #2F3440)", "var(--gray-border, #E5E9EF)"),
    "meta":   ("var(--gray-paper, #FAFBFC)", "var(--gray-muted, #7A8290)", "var(--gray-border, #E5E9EF)"),
}


def badge_html(item: EvidenceItem, run_id: str | None = None) -> str:
    """证据徽标 HTML：[n=45 ｜ R²=0.874 ｜ Run_QA_xxx]，按 source_class 着色。

    direct=BLUE_FAINT 底/BLUE_SIGNAL 字；model=白底/GRAY_INK 字；
    meta=GRAY_PAPER 底/GRAY_MUTED 字；demo 一律 PINK_LIGHT 底/PINK_ACCENT 字。
    """
    if item.demo:
        bg, fg, bd = ("var(--pink-light, #E8B8C0)", "var(--pink-accent, #C4707F)",
                      "var(--pink-mid, #DB9AA6)")
    else:
        bg, fg, bd = _BADGE_STYLE.get(item.source_class, _BADGE_STYLE["meta"])
    parts = []
    for k in ("n", "R2", "RMSE", "MAE", "Pearson"):
        v = (item.values or {}).get(k)
        if v is None or EvidenceRegistry._is_nan(v):
            continue
        try:
            parts.append(f"{k}={float(v):.3g}".replace("R2", "R²"))
        except (TypeError, ValueError):
            parts.append(f"{k}={v}")
    label = SOURCE_CLASS_LABEL.get(item.source_class, "")
    rid = run_id or ""
    core = " ｜ ".join([p for p in parts if p] + ([f"Run_{rid}"] if rid else []))
    return (f'<span class="ev-badge" style="background:{bg};color:{fg};border:1px solid {bd};">'
            f'{label} {core}</span>')


def badge_span(text: str, source_class: str = "model", *, demo: bool = False) -> str:
    """无 run 上下文（界面态）的轻量徽标：只显示文本，配色规则同 badge_html。"""
    if demo:
        bg, fg, bd = ("var(--pink-light, #E8B8C0)", "var(--pink-accent, #C4707F)",
                      "var(--pink-mid, #DB9AA6)")
    else:
        bg, fg, bd = _BADGE_STYLE.get(source_class, _BADGE_STYLE["meta"])
    label = SOURCE_CLASS_LABEL.get(source_class, "")
    return f'<span class="ev-badge" style="background:{bg};color:{fg};border:1px solid {bd};">{label} {text}</span>'


def badge_row(sentence: str, item: EvidenceItem | None, *, key: str | None = None) -> None:
    """渲染『结论句 + 徽标』，徽标下挂『证据详情』expander：
    数值表(values) + 计算方式(computed_by) + rows_filter + 来源文件列表。"""
    import streamlit as st  # 惰性 import

    if item is None:
        st.markdown(sentence + " " + '<span class="ev-badge">旧版运行，无溯源数据</span>',
                    unsafe_allow_html=True)
        return
    st.markdown(f"{sentence} ｜ {badge_html(item)}", unsafe_allow_html=True)
    exp_key = f"ev_detail_{item.eid}" + (f"_{key}" if key else "")
    with st.expander("证据详情", expanded=False):
        if item.degraded:
            st.caption("该结论因引用度量缺失已降级（未输出套话）。")
        st.caption(f"证据 ID：{item.eid} ｜ 计算方式：{item.computed_by or '—'}"
                   f" ｜ 三源分级：{SOURCE_CLASS_LABEL.get(item.source_class, item.source_class)}")
        vals = {k: v for k, v in (item.values or {}).items()}
        if vals:
            st.table({"字段": list(vals.keys()), "数值": [str(v) for v in vals.values()]})
        if item.rows_filter:
            st.caption(f"行筛选：{item.rows_filter}")
        if item.data_version or item.model_version:
            st.caption(f"数据版本：{item.data_version or '—'} ｜ 模型版本：{item.model_version or '—'}")
        if item.files:
            st.caption("来源文件：" + "；".join(str(f) for f in item.files))
        if item.demo:
            st.caption("【演示数据】模拟数据，不可用于正式科研结论。")


def legacy_run_notice() -> None:
    """V1.5 旧 run（无 manifest）：显示「旧版运行，无溯源数据」兼容提示。"""
    import streamlit as st  # 惰性 import

    st.caption("旧版运行，无溯源数据（该 run 生成于 V1.5 之前，无 evidence_manifest）。")


def demo_banner() -> None:
    """Demo 顶部醒目条（配合 P1-5 全站水印）。"""
    import streamlit as st  # 惰性 import

    st.markdown(
        '<div class="demo-banner">⚠ 当前为模拟演示数据（DEMO）——仅用于软件功能验证，'
        '不代表真实实验规律，不得用于科研结论。</div>', unsafe_allow_html=True)
