from pathlib import Path
from datetime import datetime
import hashlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import ROOT, load_schema, load_objectives, groups
from src.data_utils import load_table, validate_data, numeric_completion, compute_default_inputs
from src.model_chain import train_chain_full, predict_chain
from src.optimize import random_pareto_search
from src.paper_output import PaperExporter, PaperOutputError, ensure_output_tree, save_snapshot
from src.research_utils import (model_staleness, log_event, classify_support,
                                support_label, SUPPORT_COLOR, training_domain,
                                fmt_value, fmt_metric, friendly_error,
                                replicate_stats, dataset_hash as df_dataset_hash,
                                pareto_row_support)
from src.ui_labels import (COLUMN_LABELS, STAGE_LABELS, STAGE_FULL_LABELS,
                           MODEL_KIND_LABELS, QUALITY_LABELS, zh, zh_short, zh_df,
                           TAB_INTROS, SMART_HUB, EVIDENCE_LABELS)
from src.modes import (MODES, DEFAULT_MODE, load_schema_for_mode, data_dir, data_file,
                       model_path as mode_model_path, output_prefix, suggest_cv,
                       MELTING_CLASS_ZH)
from src.literature import (list_sheets, pick_default_sheet, load_sheet,
                            suggest_melting_mapping, build_runtime_schema,
                            default_objectives_for)
from src.context import (build_context, context_status_items, context_detail_lines,
                         context_pill_text, context_cv_short)
from src import insights as insight_mod
from src import quick_analysis as qa
from src.figure_analysis import analyze_figure
from src import plot_style as pstyle          # V1.6：网页图表样式唯一来源
from src import evidence as evidence_mod      # V1.6：全程可溯源（P0-2）
# V1.7：核心价值验证体系（L1→L4，指令集逻辑的平台化落地）
from src import calibration as calib_mod            # L1 不确定度校准
from src import validation_tracker as vtrack_mod    # L2 推荐-验证闭环
from src import benchmark as bench_mod              # L3 闭环效率基准
from src import physics_check as phys_mod           # L4 物理一致性
from src import core_value_report as cvr_mod        # L1-L4 汇总总报告
from src import architecture_compare as arch_mod    # V1.8：架构消融对照

st.set_page_config(page_title="双阳极等离子喷涂智能工艺设计平台", layout="wide")

# V1.6（P0-4）：网页图表统一 Nature 模板（白底、衬线字体、同源色板、中性灰网格）。
# 色板与模板唯一注册处为 src/plot_style.py，替代 V1.5 的手工 template 设置。
pstyle.register_nature_template()

DEMO = ROOT / "data" / "demo" / "DEMO_双阳极喷涂数据.xlsx"
PARETO_OUT = ROOT / "outputs" / "optimization" / "Pareto_工艺优化结果.xlsx"

# ---------------- V1.4 顶部：喷涂工艺 / 研究模式 + 数据来源（在侧边栏最顶部） ----------------
with st.sidebar:
    st.header("喷涂工艺 / 研究模式")
    MODE_ID = st.radio("喷涂工艺 / 研究模式", list(MODES.keys()),
                       format_func=lambda m: MODES[m]["label"],
                       key="research_mode", label_visibility="collapsed")
    MODE = MODES[MODE_ID]
    st.caption(MODE["label"] + "（V1.4 多工艺架构，各模式数据与模型完全隔离）")
    SOURCE = st.radio("数据来源", MODE["sources"], key=f"src_{MODE_ID}",
                      label_visibility="collapsed")

MODE_SCHEMA = load_schema_for_mode(MODE_ID)
DATA_KIND = MODE["source_kind"].get(SOURCE, "literature")
USE_DEMO = DATA_KIND == "demo"
CURRENT_DATA_FILE = data_file(MODE_ID, DATA_KIND)
MODEL = mode_model_path(MODE_ID)
OUTPUT_PREFIX = output_prefix(MODE_ID, DATA_KIND)

# Schema：文献模式由用户变量选择构建运行时版本；其余模式使用配置（已合并材料/熔融可选层）
if MODE_ID == "literature":
    schema = st.session_state.get("lit_schema") or MODE_SCHEMA
else:
    schema = MODE_SCHEMA
obj_cfg = load_objectives()
if MODE_ID == "literature" and st.session_state.get("lit_objectives"):
    obj_cfg = {"objectives": dict(st.session_state["lit_objectives"]), "constraints": {}}
grp = groups(schema)
use_demo = USE_DEMO  # 兼容既有 DEMO 提示逻辑

# ---------------- V1.6 全局样式：宋体 + Times New Roman · Nature 淡雅科研风 ----------------
# 色板唯一定义处 src/plot_style.py：hex 只经 _CSS_VARS 注入，CSS 主体一律 var(--*) 引用。
_CSS_VARS = f"""
:root {{
  --gray-ink: {pstyle.GRAY_INK};
  --gray-muted: {pstyle.GRAY_MUTED};
  --gray-border: {pstyle.GRAY_BORDER};
  --gray-paper: {pstyle.GRAY_PAPER};
  --gray-ref: {pstyle.GRAY_REF};
  --blue-signal: {pstyle.BLUE_SIGNAL};
  --blue-mid: {pstyle.BLUE_MID};
  --blue-light: {pstyle.BLUE_LIGHT};
  --blue-faint: {pstyle.BLUE_FAINT};
  --pink-accent: {pstyle.PINK_ACCENT};
  --pink-mid: {pstyle.PINK_MID};
  --pink-light: {pstyle.PINK_LIGHT};
  --sem-good: {pstyle.SEM_GOOD};
  --sem-bad: {pstyle.SEM_BAD};
}}
"""
_CSS_BODY = """
/* 字体：英文/数字命中 Times New Roman，中文回落宋体系衬线（不打包字体文件，缺失自动回退） */
html, body, [class*="css"], .stApp, .stApp * {
  font-family: "Times New Roman", "Songti SC", "SimSun", STSong, "Noto Serif CJK SC", serif !important;
}
html, body, .stApp {background:var(--gray-paper); color:var(--gray-ink);}
/* V1.5 Bug 修复：全局隐藏 Material 图标连字（键盘双箭头/展开/菜单/上传等图标在
   字体未加载时会泄漏英文原文并出现重复残片），全部改用平台 Unicode/原生控件 */
span[data-testid="stIconMaterial"], svg[data-testid="stIconMaterial"] { display: none !important; }
[data-testid="stToolbar"] {display: none;}
footer {visibility: hidden;}

/* 字号层级（V1.5 spec 九）：主标题 30–32，二级 23–25，模块 18–20，正文 15–16，辅助 13–14 */
h1 {font-size: clamp(26px, 2.6vw, 32px) !important; font-weight: 700 !important; color:var(--gray-ink) !important;}
h2 {font-size: clamp(20px, 2.2vw, 25px) !important; font-weight: 700 !important; color:var(--gray-ink) !important;}
h3 {font-size: clamp(17px, 1.8vw, 20px) !important; font-weight: 600 !important; color:var(--gray-ink) !important;}
p, li, .stMarkdown {font-size: 15.5px; line-height: 1.6; text-align: left; color:var(--gray-ink);}
.stCaption, p.caption, small {font-size: 13.5px !important; line-height: 1.5; color:var(--gray-muted);}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {text-align: left;}

/* 按钮（spec 十三）：普通白底浅灰边框，hover 极淡蓝；其余动作按钮一律白底描边 */
.stButton > button, button[data-testid="stBaseButton-secondary"] {
  background:#FFFFFF; color:var(--gray-ink); border:1px solid var(--gray-border);
  border-radius:8px; font-size:15px; padding:6px 16px;
}
.stButton > button:hover, button[data-testid="stBaseButton-secondary"]:hover {
  background:var(--blue-faint); border-color:var(--blue-light); color:var(--gray-ink);
}
/* V1.6（P0-1）：primary = 智能操作区专属——全站唯一实底强调按钮（其余按钮已全部降为
   白底描边 secondary）；高 ≥56px、PINK_ACCENT 实底白字 18px、占满所在列宽 */
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {
  min-height:58px !important; width:100%;
  background:var(--pink-accent) !important; color:#FFFFFF !important;
  border:none !important; border-radius:12px; font-size:18px !important; font-weight:700;
  box-shadow:0 2px 6px rgba(47,52,64,0.10); white-space:normal; line-height:1.35;
}
button[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover {
  background:var(--pink-mid) !important; color:#FFFFFF !important;
}
button[data-testid="stBaseButton-primary"]:disabled {opacity:0.6;}

/* V1.6（P0-1）：智能操作区——首页第一视觉；两键为全站唯一实底强调按钮（见上方 primary 规则） */
.smart-hub-title {font-size:20px; font-weight:700; color:var(--gray-ink);
  border-left:4px solid var(--pink-accent); padding-left:12px; margin:4px 0 12px 0;}
.smart-btn-sub {font-size:13px; color:var(--gray-muted); margin-top:4px; text-align:center;}

/* V1.6（P1-3）：三步指示器 */
.step-wrap {display:flex; align-items:center; justify-content:center; gap:0;
  margin:12px 0 4px 0; flex-wrap:wrap;}
.step-node {padding:5px 16px; border-radius:999px; font-size:14px; font-weight:600;
  border:1px solid var(--gray-border); color:var(--gray-muted); background:#FFFFFF;}
.step-arrow {color:var(--gray-ref); margin:0 10px; font-size:16px;}
.step-node.active {border-color:var(--blue-signal); color:var(--blue-signal);
  background:var(--blue-faint);}
.step-node.done {border-color:var(--blue-signal); color:#FFFFFF; background:var(--blue-signal);}
.step-node.fail {border-color:var(--sem-bad); color:var(--sem-bad); background:#FFF5F6;}
.step-msg {font-size:13px; color:var(--gray-muted); text-align:center; margin:4px 0 8px 0;}

/* V1.6（P0-2）：证据徽标 / Demo 横幅 / 摘要卡 */
.ev-badge {display:inline-block; padding:2px 10px; border-radius:6px;
  font-size:12.5px; font-weight:600; white-space:nowrap; vertical-align:middle;}
.demo-banner {background:var(--pink-light); color:var(--pink-accent);
  border:1px solid var(--pink-mid); border-radius:8px; padding:8px 14px;
  font-size:14px; font-weight:600; margin:8px 0;}
.sum-card {background:#FFFFFF; border:1px solid var(--gray-border);
  border-top:3px solid var(--pink-accent); border-radius:10px;
  padding:16px 20px; margin:12px 0; box-shadow:0 1px 3px rgba(47,52,64,0.06);}
.sum-card .sum-head {display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center;
  font-size:14px; color:var(--gray-ink); margin-bottom:8px;}
.sum-card .sum-title {font-size:17px; font-weight:700; color:var(--gray-ink);}

/* 研究主线卡片（spec 十二）：白底 + 3px 顶部色带，淡蓝/淡樱花粉交替；不裁切文字 */
.flow-wrap {display:flex; flex-wrap:wrap; align-items:stretch; gap:8px; margin:12px 0 6px 0;}
.flow-card {flex:1 1 150px; min-width:0; background:#FFFFFF; border:1px solid var(--gray-border);
  border-top:3px solid var(--blue-light); border-radius:10px; padding:16px 12px; text-align:center;
  box-shadow:0 1px 3px rgba(47,52,64,0.06);}
.flow-card:nth-child(4n+1), .flow-card:nth-child(4n+3) {border-top-color:var(--blue-light);}
.flow-card:nth-child(4n+2), .flow-card:nth-child(4n+4) {border-top-color:var(--pink-light);}
.flow-card h4 {margin:0 0 8px 0; font-size:17px; color:var(--gray-ink); font-weight:700;
  white-space:normal; word-break:break-word; line-height:1.4;}
.flow-card p {margin:2px 0; font-size:13.5px; color:var(--gray-muted); line-height:1.6;}
.flow-arrow {display:flex; align-items:center; font-size:20px; color:var(--blue-light); padding:0 2px; flex:0 0 auto;}
.obj-card {background:#FFFFFF; border:1px solid var(--gray-border); border-left:3px solid var(--pink-light);
  border-radius:10px; padding:16px 12px; text-align:center; box-shadow:0 1px 3px rgba(47,52,64,0.06);}
.obj-card .name {font-size:17px; font-weight:700; color:var(--gray-ink);}
.obj-card .dir {font-size:14px; margin-top:4px; color:var(--gray-muted); font-weight:600;}

/* 状态徽标与状态栏（淡雅浅色） */
.status-pill {display:inline-block; padding:3px 14px; border-radius:999px; font-size:14px; font-weight:600;}
[data-testid="stExpander"] summary span[data-testid="stIconMaterial"],
[data-testid="stExpander"] summary svg[data-testid="stIconMaterial"] { display: none; }
[data-testid="stExpander"] summary::after {content: "▸"; float: right; margin-right: 8px;
  font-size: 15px; color:var(--gray-muted);}
[data-testid="stExpander"] details[open] summary::after {content: "▾";}
[data-testid="stFileUploader"] span[data-testid="stIconMaterial"],
[data-testid="stFileUploader"] svg[data-testid="stIconMaterial"] { display: none; }
[data-testid="stFileUploader"] button { font-size: 14px; }
.status-bar {display:flex; flex-wrap:wrap; gap:6px 12px; align-items:center;
  background:#F2F7FC; border:1px solid var(--gray-border); border-radius:8px;
  padding:8px 14px; margin:6px 0 10px 0; font-size:13.5px; color:var(--gray-ink);}
.sb-item {white-space:nowrap;}
div[data-testid="stMetric"] {white-space: nowrap;}
div[data-testid="stMetricLabel"] p {font-size: 13px !important; color:var(--gray-muted);}
div[data-testid="stMetricValue"] {font-size: 30px !important; color:var(--gray-ink);}
.chain-note {text-align:center; color:var(--gray-muted); font-size:14px; margin:0;}
/* 通用间距（8/12/16/24/32/48，P0-7 验收 3） */
.block-spacing {margin: 24px 0;}
hr {border: none; border-top: 1px solid var(--gray-border); margin: 24px 0;}
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid var(--gray-border);}
section[data-testid="stSidebar"] * {font-size: 14.5px;}
.sb-group-title {font-size:13px; font-weight:700; color:var(--gray-muted);
  letter-spacing:1px; margin:2px 0 4px 0;}
"""
st.markdown("<style>" + _CSS_VARS + _CSS_BODY + "</style>", unsafe_allow_html=True)

st.title("双阳极等离子喷涂智能工艺设计平台")
st.caption("结构/工艺参数 + 材料/粉末属性 → 射流与粒子状态 → 颗粒熔融与沉积状态（可选） → 缺陷网络 → 涂层性能 → 数据驱动逆向工艺设计 ｜ 平台版本 V1.7（核心价值验证 L1→L4）")


# ---------------- 通用辅助 ----------------
def _predict_chain_diagnostic(df_inputs, bundle, measured_states):
    """V1.8（方案 §6.2）诊断预测：实测过程状态直接进入 Stage 2/3。

    与 predict_chain 的区别仅在 Stage 1：实测状态以 pred_ 前缀构造下游特征
    （与训练时 Stage 2 吃 Stage 1 OOF 的列名口径一致），跳过 Stage 1 模型。
    实测状态不进入任何训练路径；本结果为诊断口径，不代表设计预测能力。
    """
    from src.features import add_physics_features as _apf

    def _align(X, feature_names):
        X = X.copy()
        for c in feature_names:
            if c not in X.columns:
                X[c] = np.nan
        return X[feature_names]

    base = _apf(df_inputs.reindex(columns=bundle["x_cols"]))
    mat_cols = bundle.get("material_cols") or []
    if mat_cols:
        base = pd.concat([base, df_inputs.reindex(columns=mat_cols)
                           .apply(pd.to_numeric, errors="coerce")], axis=1)
    # 实测状态 → pred_ 前缀（未实测的状态由 Stage 1 模型预测补充）
    states = pd.DataFrame(index=df_inputs.index)
    for t, m in bundle["stage1_models"].items():
        states[t] = m.predict(_align(base, bundle["stage1_features"]))
    for t, v in measured_states.items():
        states[t] = float(v)          # 实测覆盖（诊断口径）

    melt_cont = melt_class = melt_pred = None
    if bundle.get("stage15_enabled"):
        X15 = _align(pd.concat([base, states.add_prefix("pred_")], axis=1),
                     bundle["stage15_features"])
        melt_cont = pd.DataFrame(index=df_inputs.index)
        melt_class = pd.DataFrame(index=df_inputs.index)
        melt_pred = pd.DataFrame(index=df_inputs.index)
        for t, m in bundle["stage15_models"].items():
            if bundle["stage15_kind"].get(t) == "classification":
                codes = np.asarray(m.predict(X15), dtype=float)
                melt_pred[f"pred_{t}"] = codes
                classes = bundle.get("stage15_classes", {}).get(t) or []
                melt_class[t] = [classes[int(c)] if 0 <= int(c) < len(classes) else str(c)
                                 for c in codes]
            else:
                vals15 = np.asarray(m.predict(X15), dtype=float)
                melt_pred[f"pred_{t}"] = vals15
                melt_cont[t] = vals15

    parts2 = [base, states.add_prefix("pred_")] + ([melt_pred] if melt_pred is not None else [])
    X2 = _align(pd.concat(parts2, axis=1), bundle["stage2_features"])
    defects = pd.DataFrame(index=df_inputs.index)
    for t, m in bundle["stage2_models"].items():
        defects[t] = m.predict(X2)

    parts3 = [base, states.add_prefix("pred_"), defects.add_prefix("pred_")] \
        + ([melt_pred] if melt_pred is not None else [])
    X3 = _align(pd.concat(parts3, axis=1), bundle["stage3_features"])
    perf = pd.DataFrame(index=df_inputs.index)
    unc = pd.DataFrame(index=df_inputs.index)
    for t, info in bundle["stage3_models"].items():
        m = info["model"]
        perf[t] = m.predict(X3)
        if info["kind"] == "GaussianProcess":
            Xi = m.named_steps["imputer"].transform(X3)
            Xs = m.named_steps["scale"].transform(Xi)
            mean, std = m.named_steps["model"].predict(Xs, return_std=True)
            perf[t] = mean
            unc[t + "_std"] = std
        elif info["kind"] == "RandomForest":
            Xi = m.named_steps["imputer"].transform(X3)
            rf = m.named_steps["model"]
            tree_preds = np.column_stack([tree.predict(Xi) for tree in rf.estimators_])
            unc[t + "_std"] = tree_preds.std(axis=1)
    return states, defects, perf, unc, {"continuous": melt_cont or pd.DataFrame(),
                                         "class": melt_class or pd.DataFrame()}


def ensure_thermal_margin(df_):
    """V1.4：粒子温度与熔点都在时可计算 thermal_margin_C（显示为
    「粒子温度相对熔点差值」，严禁称为熔融率；温度超过熔点 ≠ 完全熔融）。"""
    if df_ is None or "thermal_margin_C" in df_.columns:
        return df_
    if "particle_temperature_C" in df_.columns and "melting_temperature_C" in df_.columns:
        pt = pd.to_numeric(df_["particle_temperature_C"], errors="coerce")
        mt = pd.to_numeric(df_["melting_temperature_C"], errors="coerce")
        if pt.notna().any() and mt.notna().any():
            df_ = df_.copy()
            df_["thermal_margin_C"] = pt - mt
    return df_


def current_df():
    """按当前模式 + 数据来源加载当前数据（会话内缓存；上传后自动刷新）。"""
    cache_key = f"df_{MODE_ID}_{DATA_KIND}"
    if st.session_state.get(cache_key) is not None:
        return st.session_state[cache_key]
    if DATA_KIND == "demo":
        df_ = load_table(DEMO) if DEMO.exists() else None
    else:
        f_ = CURRENT_DATA_FILE
        # 兼容迁移：dual 真实数据旧路径 data/user/current_data.xlsx
        if (not f_.exists() and MODE_ID == "dual_anode" and DATA_KIND == "real"
                and (ROOT / "data" / "user" / "current_data.xlsx").exists()):
            f_ = ROOT / "data" / "user" / "current_data.xlsx"
        df_ = load_table(f_) if f_.exists() else None
    df_ = ensure_thermal_margin(df_)
    st.session_state[cache_key] = df_
    return df_


def completeness_pct(df):
    """schema 覆盖字段的总体数据完整率（%）。"""
    cols = []
    for sec in ["structure_inputs", "process_inputs", "process_states",
                "defect_network", "performance_outputs"]:
        cols += [c for c in schema[sec] if c in df.columns]
    if not cols:
        return 0.0
    s = df[cols].apply(pd.to_numeric, errors="coerce")
    return float(s.notna().mean().mean() * 100)


def get_bundle():
    """V1.4：按模式隔离的模型（bundle 会话缓存按模式分键，绝不跨模式继承）。"""
    key = f"bundle_{MODE_ID}"
    if st.session_state.get(key):
        return st.session_state[key]
    if MODEL.exists():
        try:
            return joblib.load(MODEL)
        except Exception:
            return None
    return None


def model_status_badge():
    """模型状态徽标（V1.6 色板走 CSS 变量，同源 plot_style）：区分「等待数据」与「未训练」，严格按当前模式判定。"""
    if current_df() is None:
        status = "wait_data"
    else:
        status = st.session_state.get(f"train_status_{MODE_ID}") or \
            ("done" if MODEL.exists() else "none")
    label, color, bg = {
        "none": ("未训练", "var(--gray-muted)", "var(--gray-paper)"),
        "wait_data": ("等待数据", "var(--pink-accent)", "#FFF5F6"),
        "running": ("训练中", "var(--blue-signal)", "var(--blue-faint)"),
        "done": ("✓ 模型训练完成", "var(--blue-signal)", "var(--blue-faint)"),
        "fail": ("训练失败", "var(--sem-bad)", "#FBEDEF"),
    }[status]
    st.markdown(
        f'<span class="status-pill" style="color:{color};background:{bg};'
        f'border:1px solid var(--gray-border);">模型状态：{label}</span>',
        unsafe_allow_html=True)


def stage_result_table(names, value_row, std_map=None, decimals=2):
    rows = []
    for n in names:
        r = {"预测指标": COLUMN_LABELS.get(n, n),
             "预测值": fmt_value(n, value_row[n])}
        if std_map is not None:
            std = std_map.get(n + "_std")
            if std is not None and not pd.isna(std):
                r["预测标准差"] = round(float(std), 2)
        rows.append(r)
    return pd.DataFrame(rows)


DEMO_RESULT_NOTICE = "以下结果仅用于验证平台功能，不代表真实双阳极喷涂规律。"


# ---------------- V1.6 通用辅助：Tab 导语 + status-pill（P0-7 验收 1） ----------------
def render_tab_header(tab_key, ctx):
    """每个 Tab 顶部：一句话导语（ui_labels.TAB_INTROS）+ 同源 status-pill（CTX 只读）。

    首页 / ③ / ⑧ 等所有页面的 n、数据版本、模型版本、CV 方式均来自同一 CTX（P1-2）。
    """
    intro = TAB_INTROS.get(tab_key, "")
    if intro:
        st.caption(intro)
    st.markdown(
        f'<span class="status-pill" style="color:var(--blue-signal);'
        f'background:var(--blue-faint);border:1px solid var(--blue-light);">'
        f'{context_pill_text(ctx)}</span>', unsafe_allow_html=True)


def _qa_step_of(name):
    """把 quick_analysis 的步骤名映射到三步指示器的 1/2/3（P1-3）。"""
    if any(k in name for k in ("读取", "检查")):
        return 1
    if any(k in name for k in ("训练", "解释", "优化")):
        return 2
    return 3


def render_step_indicator():
    """三步指示器（P1-3）：① 导入 → ② 分析 → ③ 结论。

    状态存 st.session_state["smart_step"]：{step, state, msg}；
    state ∈ idle|running|done|fail；失败停在该步并显示原因（P0-3 验收 4）。
    """
    ss = st.session_state.get("smart_step") or {"step": 1, "state": "idle", "msg": ""}
    cur, state = int(ss.get("step", 1)), ss.get("state", "idle")
    nodes = []
    for i, label in enumerate(SMART_HUB["steps"], start=1):
        if state == "fail" and i == cur:
            cls = "step-node fail"
        elif i < cur or (i == cur and state == "done"):
            cls = "step-node done"
        elif i == cur and state == "running":
            cls = "step-node active"
        else:
            cls = "step-node"
        nodes.append(f'<span class="{cls}">{label}</span>')
        if i < len(SMART_HUB["steps"]):
            nodes.append('<span class="step-arrow">──▶</span>')
    st.markdown(f'<div class="step-wrap">{"".join(nodes)}</div>', unsafe_allow_html=True)
    if state == "fail" and ss.get("msg"):
        st.markdown(f'<div class="step-msg">失败原因：{ss["msg"]}</div>', unsafe_allow_html=True)
    elif state == "running" and ss.get("msg"):
        st.markdown(f'<div class="step-msg">{ss["msg"]}</div>', unsafe_allow_html=True)


def _smart_run_analysis(file_obj):
    """执行一键分析并驱动三步指示器（P1-3 / P0-3 验收 4）。

    进度回调写入 st.session_state["smart_step"]；任一步失败停在出错步并显示原因，
    不产出半成品结论；成功后自动展开摘要卡（smart_summary_run_id）。
    """
    st.session_state["smart_step"] = {"step": 1, "state": "running", "msg": "数据读取与检查"}
    _prog = st.status("智能分析运行中…", expanded=True)
    _log_lines = []

    def _cb(name, status_, msg=""):
        tag = {"running": "▶", "done": "✓", "skip": "跳过", "fail": "失败"}.get(status_, "·")
        _log_lines.append(f"{tag} {name}" + (f"（{msg}）" if msg and status_ in ("fail", "skip") else ""))
        _prog.write("\n".join(_log_lines[-12:]))
        step_no = _qa_step_of(name)
        if status_ == "fail":
            st.session_state["smart_step"] = {"step": step_no, "state": "fail",
                                              "msg": f"{name}：{msg or '未知原因'}"}
        elif status_ == "running":
            st.session_state["smart_step"] = {"step": step_no, "state": "running", "msg": name}

    try:
        results = qa.quick_analyze_file(file_obj, progress_cb=_cb,
                                        demo=st.session_state.get("qa_uploaded_demo", False),
                                        confirm_maps=st.session_state.get("qa_confirm_maps"))
    except Exception as e:
        st.session_state["smart_step"] = {"step": 2, "state": "fail",
                                          "msg": friendly_error(e)}
        _prog.update(label="智能分析失败", state="error")
        st.error(f"智能分析失败：{friendly_error(e)}")
        return

    ok_runs = [r for r in results if not r["record"].get("failed")]
    summary_lines = []
    for r in results:
        steps_txt = "，".join(f"{n}:{s}" + (f"（{rsn}）" if rsn and s in ("skip", "fail") else "")
                             for n, s, rsn in r["steps"])
        ok = not r["record"].get("failed")
        summary_lines.append(
            f"**Run {r['run_id']}**（{r['mode']}，工作表 {r['record'].get('sheet')}）："
            + ("完成 ｜ " + steps_txt if ok else "未完成（数据准备检查未通过或训练失败）｜ " + steps_txt))
        if r.get("package"):
            summary_lines.append(f"- 结果目录：`{r['package'].get('run_dir')}`"
                                 f"（图 {r['package'].get('figures')} 张，"
                                 f"含 Insight_Report、图表解读与 evidence_manifest）")
    st.markdown("\\n\\n".join(summary_lines))
    st.session_state["qa_state"] = {"last_results": [
        {k: r[k] for k in ("run_id", "mode", "record")} for r in results]}

    if ok_runs:
        # 默认展示最近一次成功 run（已拍板决策 §八.3）
        st.session_state["smart_summary_run_id"] = ok_runs[-1]["run_id"]
        st.session_state["smart_step"] = {"step": 3, "state": "done", "msg": ""}
        _prog.update(label="智能分析完成", state="complete")
    else:
        # 全部失败：如实展示失败原因与检查项，禁止半成品结论（P0-3 验收 4）
        _fail_reasons = []
        for r in results:
            for n, s, rsn in r["steps"]:
                if s == "fail" and rsn:
                    _fail_reasons.append(f"{n}：{rsn}")
        _warns = []
        for r in results:
            _warns += r.get("warnings") or []
        st.session_state["smart_step"] = {
            "step": 2, "state": "fail",
            "msg": "；".join(_fail_reasons[:3]) or "数据准备检查未通过"}
        _prog.update(label="智能分析未完成", state="error")
        if _fail_reasons:
            st.error("未通过的检查项：\n- " + "\n- ".join(_fail_reasons[:6]))
        if _warns:
            st.warning("提示：\n- " + "\n- ".join(_warns[:6]))


def render_smart_summary(run_id):
    """智能分析摘要卡（P0-3）：顶部元信息 / 证据区 / 结论区（徽标可溯源）/ 行动区。

    数据源：qa.load_smart_summary(run_id) + EvidenceRegistry.load(run_dir)；
    旧 run 无 manifest → evidence.legacy_run_notice()（已拍板兼容方案）。
    """
    summ = qa.load_smart_summary(run_id)
    run_dir = qa.QUICK_ROOT / str(run_id)
    if summ is None:
        st.info(f"运行 {run_id} 未生成智能摘要（可能为 V1.5 旧版运行或缺 summary.json）。")
        evidence_mod.legacy_run_notice()
        return
    registry = evidence_mod.EvidenceRegistry.load(run_dir)
    demo = bool(summ.get("demo"))
    head = summ.get("header") or {}

    st.markdown('<div class="sum-card">', unsafe_allow_html=True)
    # ① 顶部：Run 元信息 + status-pill（全部真实计算值）
    _pills = [f"研究模式：{summ.get('mode_label', summ.get('mode', '—'))}",
              f"n={head.get('n', '—')}",
              f"数据版本 {head.get('data_version') or '—'}",
              f"模型版本 {str(head.get('model_version') or '—').split('_')[-1]}",
              f"CV: {head.get('cv_method') or '—'}"]
    if head.get("stage15"):
        _pills.append("Stage 1.5 已启用")
    if demo:
        _pills.append("演示数据")
    st.markdown(
        f'<div class="sum-head"><span class="sum-title">智能分析摘要 · Run {run_id}</span>'
        + "".join(f'<span class="status-pill" style="color:var(--blue-signal);'
                  f'background:var(--blue-faint);border:1px solid var(--blue-light);">{p}</span>'
                  for p in _pills) + "</div>", unsafe_allow_html=True)
    if demo:
        st.markdown(f'<div class="demo-banner">{EVIDENCE_LABELS["demo_notice"]}</div>',
                    unsafe_allow_html=True)

    # ② 证据区：各 Stage 指标小表 + Top 5 特征重要性（统一图入口）
    c_ev1, c_ev2 = st.columns([3, 2])
    with c_ev1:
        stage_rows = []
        _stage_name = {"stage1": "一级模型", "stage15": "Stage 1.5",
                       "stage2": "二级模型", "stage3": "三级模型"}
        for stage, rows in (summ.get("stages") or {}).items():
            for r in rows:
                stage_rows.append({
                    "模型层级": _stage_name.get(stage, stage),
                    "目标": zh(r.get("target", "")),
                    "n": r.get("n"),
                    **({"R²": round(float(r["R2"]), 3), "RMSE": round(float(r["RMSE"]), 3),
                        "MAE": round(float(r["MAE"]), 3)} if "R2" in r else
                       {"Accuracy": r.get("Accuracy"), "平衡准确率": r.get("Balanced_Accuracy")}),
                })
        if stage_rows:
            st.dataframe(pd.DataFrame(stage_rows), width="stretch", hide_index=True)
    with c_ev2:
        tf = summ.get("top_features") or []
        if tf:
            imp_df = pd.DataFrame({"影响因素": [zh(f["feature"]) for f in tf],
                                   "重要性": [f["importance"] for f in tf]})
            fig_imp = pstyle.importance_chart(imp_df, title="Top 5 特征重要性（一级模型首目标）")
            if demo:
                pstyle.add_demo_watermark(fig_imp)
            st.plotly_chart(fig_imp, width="stretch", key=f"sum_imp_{run_id}")
    if summ.get("coverage"):
        st.caption(f"数据覆盖：{summ['coverage']}")
    if summ.get("pareto"):
        st.caption(f"Pareto：非支配方案 {summ['pareto']['n_pareto']} 个"
                   f"（候选 {summ['pareto']['n_candidates']}，满足约束 {summ['pareto']['n_feasible']}）")

    # ③ 结论区：每句带证据徽标，可点开溯源（P0-2 验收 1）
    st.markdown("**结论（每条可点开「证据详情」溯源）**")
    conclusions = summ.get("conclusions") or []
    if not conclusions:
        st.caption("本 run 未生成证据化结论（可能为旧版运行）。")
        evidence_mod.legacy_run_notice()
    for c in conclusions:
        item = registry.get(c["eid"]) if registry is not None else None
        evidence_mod.badge_row(c["text"], item, key=run_id)

    # ④ 行动区：下一步指引（Streamlit 无法编程切 Tab，降级为指引 + run_id 复制）+ 下载 ZIP
    st.markdown("**下一步**：前往 ③ 模型训练 查看完整 CV 指标 ｜ ⑤ 模型解析 查看 SHAP/PDP ｜ "
                "⑥ 数据洞察 查看三源分级 ｜ ⑦ 逆向设计 查看 Pareto ｜ ⑧ 结果输出 下载论文结果包。")
    _c_zip = run_dir / "results" / "results.zip"
    _zip_alt = run_dir / "results.zip"
    _zip_path = _c_zip if _c_zip.exists() else (_zip_alt if _zip_alt.exists() else None)
    a1, a2 = st.columns(2)
    a1.code(run_id, language=None)
    a1.caption("Run ID（一键复制，用于运行历史 / 设为正式模型）")
    if _zip_path is not None:
        with open(_zip_path, "rb") as f:
            a2.download_button("下载结果包 ZIP（含 evidence_manifest.json）", f,
                               file_name=_zip_path.name, mime="application/zip",
                               key=f"sum_zip_{run_id}", use_container_width=True)
    else:
        a2.caption("结果包 ZIP 见运行目录（本 run 未打包或已清理）。")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- 左侧操作栏（V1.6 P0-7 验收 4：模式/来源 → 分隔 → 推荐流程 → 分隔 → 状态摘要） ----------------
with st.sidebar:
    # 分组一：模式与数据来源提示（顶部模式选择已在 V1.4 区块）
    if USE_DEMO:
        evidence_mod.demo_banner()
    elif DATA_KIND == "literature":
        st.info("【文献数据模式】\n\n请在「数据管理」页面上传文献数据表，"
                "文献模式需自行选择 X / 熔融 / 缺陷 / Y 变量。")
    else:
        st.info("【真实实验数据模式】\n\n请先在「数据管理」页面上传真实数据。")

    st.markdown('<div class="sb-group-title">推荐流程</div>', unsafe_allow_html=True)
    st.markdown("\n".join(MODE["flow"]))

    st.markdown('<div class="sb-group-title">状态摘要</div>', unsafe_allow_html=True)
    model_status_badge()

    _sd = current_df()
    if not USE_DEMO and _sd is not None:
        st.markdown('---')
        st.markdown("**当前数据概况**")
        st.write(f"文件名称：{st.session_state.get('data_name', CURRENT_DATA_FILE.name)}")
        st.write(f"数据记录数：{len(_sd)}")
        st.write(f"独立喷涂批次：{_sd['batch_id'].nunique() if 'batch_id' in _sd else '-'}")
        st.write(f"数据完整率：{completeness_pct(_sd):.1f} %")

# ---------------- V1.5 CurrentContext：全局唯一状态源 + 状态栏 + 详细信息 ----------------
_d0 = current_df()
_b0 = get_bundle()
CTX = build_context(MODE_ID, SOURCE, df=_d0, bundle=_b0,
                    model_file_exists=MODEL.exists(), source_kind=DATA_KIND,
                    active_run_id=st.session_state.get("active_run_id"))
st.session_state["ctx"] = CTX
st.markdown('<div class="status-bar">' + " ｜ ".join(
    f'<span class="sb-item">{p}</span>' for p in context_status_items(CTX)) + "</div>",
    unsafe_allow_html=True)
with st.expander("详细信息（dataset hash / 完整模型版本 / 时间戳）", expanded=False):
    for _ln in context_detail_lines(CTX):
        st.caption(_ln)
_stale, _mh, _ch = (None, None, None)
if _b0 and _d0 is not None:
    _stale, _mh, _ch = model_staleness(_b0, _d0)
    log_event("data_load", f"mode={MODE_ID} kind={DATA_KIND} rows={len(_d0)} "
                           f"model={_b0.get('model_version')} stale={_stale}")
if _stale is True:
    st.warning("当前实验数据已经发生变化，现有模型并非基于最新数据训练，建议重新训练。")

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "① 平台首页", "② 数据管理", "③ 模型训练", "④ 工艺预测",
    "⑤ 模型解析", "⑥ 数据洞察与实验反馈", "⑦ 逆向设计", "⑧ 结果输出"
])

# ---------------- ① 平台首页（V1.6：智能操作区 + 摘要卡 + 研究主线 + 历史 + 状态面板） ----------------
with tab0:
    render_tab_header("tab0", CTX)

    # ---------- V1.8（方案 §3）：研究工作台——当前该做什么（阻塞原因 + 下一步 + 待办） ----------
    with st.container(border=True):
        _todo = vtrack_mod.pending_counts()
        _w_kb, _w_next = st.columns([3, 2])
        with _w_kb:
            # 阻塞原因判定（按流程链顺序，第一个不满足的即当前阻塞点）
            if not CTX.get("dataset_loaded"):
                _block, _act = "数据未加载", "② 数据管理：上传或选择数据来源"
            elif not CTX.get("model_trained"):
                _block, _act = "模型未训练", "③ 模型训练：开始训练（约 1–3 分钟）"
            elif get_bundle() is None:
                _block, _act = "模型已过期（数据已变化）", "③ 模型训练：用最新数据重新训练"
            else:
                _block, _act = None, "⑦ 逆向设计 → 导出验证任务单，安排实验"
            st.markdown(f"**下一步**：" + (_act if _block is None
                        else f"当前阻塞于「{_block}」——{_act}"))
            # 待办徽标（待测候选 / 待审核实测）
            _todo_bits = []
            if _todo["pending_validation"]:
                _todo_bits.append(f"待回灌候选 {_todo['pending_validation']} 个（⑥ 上传判定）")
            _cal_b = get_bundle()
            if _cal_b is not None and calib_mod.calibration_status(
                    _cal_b.get("model_version")).get("done") is False:
                _todo_bits.append("L1 未校准（③ 运行校准）")
            if _todo_bits:
                st.caption(" ｜ ".join(_todo_bits))
            else:
                st.caption("暂无待办事项。")
        with _w_next:
            # 一个随状态变化的主要操作（V1.8 方案 §3：工作台首屏原则）
            if CTX.get("dataset_loaded") and CTX.get("model_trained") and get_bundle() is not None:
                st.caption("核心动作：⑦ 逆向设计寻优（含验证任务单导出）")
            else:
                st.caption("核心动作：② 加载数据 → ③ 训练")

    # ---------- ⓪ 智能操作区（Smart Actions Hub，P0-1）：首屏第一视觉 ----------
    _smart_run_now = False
    with st.container():
        st.markdown(f'<div class="smart-hub-title">{SMART_HUB["title"]}</div>',
                    unsafe_allow_html=True)
        _qa_hist_now = [h for h in qa.latest_runs(10) if not h.get("failed")]
        c_imp, c_ana = st.columns(2)
        with c_imp:
            # 「⬆ 智能一键导入」：点击展开内联拖放区（key="qa_uploader"，行为与 V1.5 一致）
            if st.button(SMART_HUB["btn_import"], type="primary", key="smart_import_btn"):
                st.session_state["smart_import_open"] = not st.session_state.get("smart_import_open", False)
            st.caption(SMART_HUB["btn_import_hint"])
        with c_ana:
            _pending_file = st.session_state.get("qa_file_cache")
            if _pending_file is not None:
                # 就绪态：有文件未分析 → 直接触发一键分析
                if st.button(SMART_HUB["btn_analyze"], type="primary", key="smart_analyze_btn"):
                    _smart_run_now = True
                st.caption(f"{SMART_HUB['ready_hint']}（文件：{_pending_file.name}）")
            elif _qa_hist_now:
                # 结果态：有最新 run → 刷新摘要卡
                if st.button(f"{SMART_HUB['btn_analyze']}（查看最新结果）", type="primary",
                             key="smart_analyze_btn"):
                    st.session_state["smart_summary_run_id"] = _qa_hist_now[0]["run_id"]
                st.caption(SMART_HUB["result_hint"])
            else:
                # 空态：半透明（ghost 样式），点击引导导入
                if st.button(SMART_HUB["btn_analyze_empty"], key="smart_analyze_btn"):
                    st.info(SMART_HUB["empty_guide"])
                st.caption("尚无待分析文件与历史运行记录。")

        # 内联拖放区（智能一键导入展开时显示； uploader key 与 V1.5 完全一致）
        if st.session_state.get("smart_import_open"):
            qa_file = st.file_uploader("将 Excel / CSV 拖到这里（支持 XLSX、XLS、CSV）",
                                       type=["xlsx", "xls", "csv"], key="qa_uploader")
            if qa_file is not None:
                st.checkbox("上传文件为模拟演示数据", value=False, key="qa_uploaded_demo",
                            help="仅模拟数据勾选；文献实测数据不应继承首页演示模式。")
                file_hash = hashlib.sha256(qa_file.getvalue()).hexdigest()
                if st.session_state.get("qa_file_hash") != file_hash:
                    st.session_state["qa_file_hash"] = file_hash
                    st.session_state["qa_confirm_maps"] = {}
                st.session_state["qa_file_cache"] = qa_file
                try:
                    qa_sheets = list_sheets(qa_file)
                    qa_data_sheets, qa_meta_sheets = qa.classify_sheets(qa_sheets)
                    st.info(f"识别到 {len(qa_data_sheets)} 个数据工作表（已跳过说明/元数据表："
                            f"{'、'.join(qa_meta_sheets) or '无'}），每个将独立运行；"
                            "数据安全检查未通过时会明确拦截并说明原因。")
                    confirmed = {}
                    role_choices = {"忽略": "ignore", "工艺输入 X": "x",
                                    "过程状态 Y": "states", "涂层缺陷 Y": "defects",
                                    "涂层性能 Y": "performance", "材料属性": "material"}
                    with st.expander("确认未识别字段与涂层性能优化方向", expanded=False):
                        for sheet in qa_data_sheets:
                            frame = load_sheet(qa_file, sheet)
                            selected, ambiguous = qa.infer_field_map(frame)
                            st.markdown(f"**{sheet}**：输入 {selected['x']}；过程目标 {selected['states']}；"
                                        f"涂层性能 {selected['performance']}")
                            mapping = {}
                            for col in ambiguous:
                                key = hashlib.sha256(f"{file_hash}/{sheet}/{col}".encode()).hexdigest()[:16]
                                label = st.selectbox(f"字段「{col}」的角色", list(role_choices),
                                                     key=f"qa_role_{key}")
                                mapping[col] = role_choices[label]
                            perf = list(selected["defects"]) + list(selected["performance"]) + [
                                c for c, role in mapping.items() if role in ("defects", "performance")]
                            orientations = {}
                            for col in perf:
                                key = hashlib.sha256(f"{file_hash}/{sheet}/{col}/goal".encode()).hexdigest()[:16]
                                label = st.selectbox(f"涂层缺陷/性能「{col}」的优化方向",
                                                     ["仅预测", "最大化", "最小化"],
                                                     key=f"qa_goal_{key}")
                                orientations[col] = {"仅预测": "prediction_only",
                                                     "最大化": "maximize", "最小化": "minimize"}[label]
                            mapping["_directions"] = orientations
                            confirmed[sheet] = mapping
                        st.caption("多目标优化至少需要两个已训练的缺陷/性能目标，且有涂层性能预测层；"
                                   "Ar/H₂ 压力（psi）按压力输入，不会按气体流量处理。")
                    st.session_state["qa_confirm_maps"] = confirmed
                except Exception as e:
                    st.error(f"文件读取失败：{friendly_error(e)}")
        elif st.session_state.get("qa_file_cache") is not None:
            st.caption(f"已缓存待分析文件：{st.session_state['qa_file_cache'].name}"
                       "（点击「✦ 智能分析」开始；重新点击「⬆ 智能一键导入」可更换文件）")

    # 三步指示器（P1-3）：在智能操作区之下、分析执行之后渲染，确保显示最新状态
    render_step_indicator()
    if _smart_run_now and st.session_state.get("qa_file_cache") is not None:
        _smart_run_analysis(st.session_state["qa_file_cache"])
        render_step_indicator()  # 分析完成后即时刷新指示器终态

    # ---------- ⓛ 智能分析摘要卡（Smart Summary Card，P0-3） ----------
    if st.session_state.get("smart_summary_run_id"):
        _sum_runs = [h for h in qa.latest_runs(10) if not h.get("failed")]
        if len(_sum_runs) > 1:
            _pick = st.selectbox("查看运行", [h["run_id"] for h in _sum_runs],
                                 index=[h["run_id"] for h in _sum_runs].index(
                                     st.session_state["smart_summary_run_id"])
                                 if st.session_state["smart_summary_run_id"] in
                                 [h["run_id"] for h in _sum_runs] else 0,
                                 key="sum_run_pick")
            st.session_state["smart_summary_run_id"] = _pick
        render_smart_summary(st.session_state["smart_summary_run_id"])

    # ---------- ⓜ 研究主线卡片 flow-card（V1.5 原样保留） ----------
    st.markdown("---")
    st.subheader(f"研究主线 · {MODE['label']}")
    flow = MODE["research_line"]
    cards = []
    for i, (t, items) in enumerate(flow):
        cards.append(f'<div class="flow-card"><h4>{t}</h4>{"".join(f"<p>{x}</p>" for x in items.split("<br>"))}</div>')
        if i < len(flow) - 1:
            cards.append('<div class="flow-arrow">→</div>')
    st.markdown(f'<div class="flow-wrap">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown('<p class="chain-note">喷枪结构/工艺 + 材料/粉末属性 → 射流与粒子状态 → 颗粒熔融与沉积（可选）'
                ' → 缺陷网络 → 涂层性能 → 多目标逆向设计 → 实验验证与模型更新</p>',
                unsafe_allow_html=True)

    # ---------- ⓝ 运行历史 / 设为正式模型（折叠 expander，V1.5 原样保留） ----------
    if st.button("进入常规分析模式（数据管理 → 训练 → 预测 → 解析 → 洞察 → 逆向 → 输出）",
                 key="qa_expert_hint"):
        st.info("请依次使用顶部导航的 ②–⑧ 页；每个环节可单独控制。")

    qa_hist = qa.latest_runs(10)
    if qa_hist:
        with st.expander(f"运行历史（Run History，最近 {len(qa_hist)} 次）", expanded=False):
            hist_rows = []
            for h in qa_hist:
                hist_rows.append({
                    "run_id": h["run_id"], "时间": h["date"],
                    "研究模式": h["research_mode"], "数据表": h.get("sheet"),
                    "模型版本": h.get("model_version") or "—",
                    "Stage 1.5": "启用" if h.get("stage15") else "未启用",
                    "验证方式": (h.get("validation_method") or "—")[:40],
                })
            st.dataframe(pd.DataFrame(hist_rows), width="stretch", hide_index=True)
            st.caption("【设为正式模型】需在下方选择 run 并主动确认；Quick Run 默认不进入正式 models/。")
            _promote_id = st.text_input("要设为正式模型的 run_id", key="qa_promote_id")
            if st.button("设为正式模型（需确认）", key="qa_promote_btn"):
                if _promote_id:
                    try:
                        dst = qa.promote_to_formal(_promote_id, MODE_ID)
                        st.success(f"已把 Quick Run {_promote_id} 设为 {MODE['label']} 正式模型：{dst}")
                    except Exception as e:
                        st.error(f"设置失败：{e}")
                else:
                    st.warning("请先填写 run_id。")

    # ---------- ⓞ 平台状态面板（P1-2 升级：与 ③⑧ 同源 CTX，禁止二次计算） ----------
    st.markdown("---")
    st.subheader("平台状态")
    d = current_df()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("当前实验数据量", CTX.get("record_count") if CTX.get("dataset_loaded") else "-")
    c2.metric("喷涂批次数", CTX.get("batch_count") if CTX.get("batch_count") is not None else "-")
    c3.metric("数据完整率", f"{completeness_pct(d):.1f} %" if d is not None else "-")
    c4.metric("数据版本", CTX.get("dataset_version") or "—")
    c5.metric("模型状态", ("✓ 已训练" if CTX.get("model_trained") else "未训练")
              + f" ｜ {context_cv_short(CTX)}")
    c6.metric("数据模式", "模拟数据" if use_demo else "真实数据")
    st.caption(f"研究模式：{CTX.get('research_mode_label')} ｜ 模型版本：{CTX.get('model_version') or '—'}"
               f" ｜ 状态与 ③ 模型训练 / ⑧ 结果输出 同源（同一 CTX）。")

    # ---------------- V1.7/V1.8 · 核心价值验证四维证据徽标（首页概览） ----------------
    # V1.8（方案 §3/§10）：四个证据维度独立判定（非闯关）；绑定当前模型/数据才显示有效通过
    _cv_status = cvr_mod.core_value_status(
        (get_bundle() or {}).get("model_version") if get_bundle() else None,
        (CTX.get("dataset_hash") if isinstance(CTX, dict) else None))
    _pill = {"L1": ("L1 校准", "③ 模型训练"), "L2": ("L2 覆盖+达标", "⑥ 数据洞察"),
             "L3": ("L3 离线策略评估", "⑥ 数据洞察"), "L4": ("L4 物理合理性", "⑤ 模型解析")}
    _cols_cv = st.columns(4)
    for _cc, (_lk, (_lt, _loc)) in zip(_cols_cv, _pill.items()):
        _v = _cv_status["layers"].get(_lk) or {}
        if _v.get("passed"):
            _txt, _bg, _fg = f"✓ {_lt}", "#EAF3FB", "#3D5A78"
        elif _v.get("done") and _v.get("fresh") is False:
            _txt, _bg, _fg = f"🕘 {_lt}（历史）", "#F2F7FC", "#98A2B3"
        elif _v.get("done"):
            _txt, _bg, _fg = f"△ {_lt}（未通过）", "#FFF5F6", "#B77C87"
        else:
            _txt, _bg, _fg = f"○ {_lt}（未运行）", "#F2F7FC", "#7A8290"
        _cc.markdown(
            f'<span class="status-pill" style="color:{_fg};background:{_bg};">{_txt} · {_loc}</span>',
            unsafe_allow_html=True)
    _eff = _cv_status.get("effective") or []
    _stale = _cv_status.get("stale") or []
    if _eff:
        _eff_txt = "、".join(_eff)
        _stale_txt = (" ｜ 历史结果（旧模型/数据，不计入当前）：" + "、".join(_stale) + "。") if _stale else ""
        st.caption("当前上下文下有效证据维度：**" + _eff_txt + "**（结果绑定当前模型/数据）。"
                   + _stale_txt
                   + " 总报告在「⑧ 结果输出 → 核心价值验证总报告」。")
    else:
        _stale_txt2 = ("；历史结果（旧模型/数据）：" + "、".join(_stale)) if _stale else ""
        st.caption("核心价值验证（L1 校准 ｜ L2 覆盖+达标 ｜ L3 离线策略评估 ｜ L4 物理合理性）"
                   "当前上下文下暂无有效通过的维度"
                   + _stale_txt2
                   + "。建议从「③ 模型训练 → L1 校准」开始补齐当前证据。")

    if use_demo and d is not None:
        evidence_mod.demo_banner()

# ---------------- ② 数据管理 ----------------
with tab1:
    st.subheader("数据管理与质量检查")
    render_tab_header("tab1", CTX)
    if USE_DEMO:
        d = current_df()
        evidence_mod.demo_banner()
    else:
        # V1.4/V1.5：按模式/来源上传；修复上传按钮图标连字重复残片（隐藏图标 + 明确格式说明）
        uploaded = st.file_uploader("上传数据文件", type=["xlsx", "xls", "csv"],
                                    key=f"up_{MODE_ID}_{DATA_KIND}")
        st.caption("支持 XLSX、XLS、CSV")
        if uploaded is not None:
            try:
                sheets = list_sheets(uploaded)
                sheet_choice = None
                if len(sheets) > 1:
                    # V1.4 spec 三十六：多工作表必须先选择，禁止默认拿 README/说明表当训练数据
                    _def = pick_default_sheet(sheets)
                    sheet_choice = st.selectbox(
                        "【选择数据工作表】", sheets,
                        index=sheets.index(_def) if _def in sheets else 0,
                        key=f"sheet_{MODE_ID}_{DATA_KIND}_{uploaded.name}")
                    st.caption(f"共 {len(sheets)} 个工作表，默认已跳过说明类工作表。")
                d = load_sheet(uploaded, sheet_choice)
                dest = data_dir(MODE_ID, DATA_KIND)
                dest.mkdir(parents=True, exist_ok=True)
                with pd.ExcelWriter(CURRENT_DATA_FILE, engine="openpyxl") as w:
                    d.to_excel(w, sheet_name="Data", index=False)
                st.session_state["data_name"] = uploaded.name
                st.session_state[f"df_{MODE_ID}_{DATA_KIND}"] = ensure_thermal_margin(d)
                st.success(f"已加载{SOURCE}：{uploaded.name}"
                           + (f"（工作表：{sheet_choice}）" if sheet_choice else ""))
            except Exception:
                d = current_df()
                st.error("数据读取失败：请检查文件是否为有效的 Excel / CSV，且包含实验数据表。")
        else:
            d = current_df()
            if d is not None:
                st.success(f"当前使用{SOURCE}（{CURRENT_DATA_FILE.name}）。")

    if d is not None:
        # V1.3.2：数据记录数、独立喷涂批次数、独立试样数三者的定义彼此独立，不再混用
        sample_ids = pd.to_numeric(d["sample_id"], errors="coerce").dropna() \
            if "sample_id" in d.columns else pd.Series(dtype="float64")
        n_samples = sample_ids.nunique() if len(sample_ids) else None
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("数据记录数", len(d))
        m2.metric("独立喷涂批次数", d["batch_id"].nunique() if "batch_id" in d else "-")
        m3.metric("独立试样数", n_samples if n_samples is not None else "未填写 sample_id")
        m4.metric("完整数据比例", f"{completeness_pct(d):.1f} %")
        issues = validate_data(d, schema)
        m5.metric("异常字段数量", len(issues))
        if "sample_id" not in d.columns:
            st.caption("说明：当前数据未包含 sample_id 列，独立试样数暂无法统计；"
                       "数据记录数按行计数，与独立试样数是不同概念。")

        # V1.6（P0-7 验收 2）：异常字段检查为 hero（首屏可见）；预览/完整度收进次级证据区
        if issues:
            st.warning("⚠ 检测到以下数据问题")
            for x in issues:
                st.write("•", x)
        else:
            st.success("✓ 基础数据格式检查通过")

        with st.expander("次级证据：数据预览（前 30 行）", expanded=False):
            preview = d.head(30).copy()
            num_cols = preview.select_dtypes("number").columns
            preview[num_cols] = preview[num_cols].round(3)
            st.dataframe(zh_df(preview), width="stretch")

        target_cols = grp["process_states"] + grp["defect_network"] + grp["performance_outputs"]
        comp = numeric_completion(d, target_cols)
        if len(comp):
            with st.expander("次级证据：关键字段数据完整度", expanded=False):
                comp_disp = comp.copy()
                comp_disp["column"] = comp_disp["column"].map(zh)
                comp_disp = comp_disp.rename(columns=QUALITY_LABELS)
                comp_disp["缺失比例（%）"] = comp_disp["缺失比例（%）"].round(1)
                st.dataframe(comp_disp, width="stretch", hide_index=True)

        with st.expander("次级证据：重复性统计（重复测量，n/mean/std/CV%）", expanded=False):
            stats_df = replicate_stats(d, "sample_id") or replicate_stats(d, "experiment_id")
            if stats_df is None or stats_df.empty:
                st.caption("当前数据未检测到重复测量记录（同一试样/实验条件下 ≥2 次测量）。"
                           "填写模板中的 sample_id / replicate_id 后可自动统计。")
            else:
                st.caption("以下统计仅用于展示重复性，不会用均值覆盖原始数据。")
                disp_rep = stats_df.copy()
                disp_rep["字段"] = disp_rep["字段"].map(zh)
                disp_rep["mean"] = disp_rep["mean"].round(3)
                disp_rep["std"] = disp_rep["std"].round(3)
                st.dataframe(disp_rep, width="stretch", hide_index=True)

        # ---------------- V1.4 文献模式变量选择（spec 37/29/38） ----------------
        if MODE_ID == "literature" and d is not None:
            st.markdown("---")
            st.markdown("**文献模式变量选择**（不假设所有文献都符合完整链条）")
            cols_ = [str(c) for c in d.columns]
            c1, c2 = st.columns(2)
            sel_x = c1.multiselect("输入变量 X（至少 1 个）", cols_, key="lit_x")
            sel_mat = c2.multiselect("材料 / 粉末属性（可选）", cols_, key="lit_mat")
            c3, c4 = st.columns(2)
            sel_states = c3.multiselect("过程状态变量（可选）", cols_, key="lit_states")
            sel_defects = c4.multiselect("缺陷变量（可选）", cols_, key="lit_defects")
            sel_perf = st.multiselect("性能目标 Y（可选）", cols_, key="lit_perf")

            st.markdown("**熔融状态字段映射（Stage 1.5，可选）**")
            st.caption("平台只提出「建议映射」，最终以您确认的选择为准；"
                       "温度超过熔点不等于完全熔融，平台不会用温度阈值自动生成熔融标签。")
            sugg = suggest_melting_mapping(cols_)
            melt_fields = [f for f, sp in (MODE_SCHEMA.get("melting_states") or {}).items()
                           if sp.get("role") != "metadata" and f != "melting_data_source"]
            melt_map = {}
            for f in melt_fields:
                opts = ["（不映射）"] + cols_
                _def_col = sugg.get(f, "（不映射）")
                pick = st.selectbox(f"{zh(f)}（{f}） ← 数据列", opts,
                                    index=opts.index(_def_col) if _def_col in opts else 0,
                                    key=f"lit_melt_{f}")
                if pick != "（不映射）":
                    melt_map[f] = pick
            if sugg:
                st.caption("建议映射：" + ("；".join(f"{zh(k)} ← {v}" for k, v in sugg.items())
                                          or "无相近列名"))

            if st.button("应用变量选择并构建运行时 Schema", key="lit_apply"):
                if not sel_x:
                    st.error("至少选择 1 个输入变量 X。")
                else:
                    sel = {"x": sel_x, "states": sel_states, "defects": sel_defects,
                           "performance": sel_perf, "material": sel_mat, "melting_map": melt_map}
                    sch = build_runtime_schema(d, sel)
                    st.session_state["lit_schema"] = sch
                    st.session_state["lit_objectives"] = default_objectives_for(d, sch)
                    # 熔融字段按用户确认的映射重命名为规范字段名（Stage 1.5 目标名）
                    _ren = {v: k for k, v in melt_map.items()}
                    if _ren:
                        d = d.rename(columns=_ren)
                        st.session_state[f"df_{MODE_ID}_{DATA_KIND}"] = d
                        CURRENT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
                        with pd.ExcelWriter(CURRENT_DATA_FILE, engine="openpyxl") as w:
                            d.to_excel(w, sheet_name="Data", index=False)
                        st.caption("已将映射列重命名为规范熔融字段：" +
                                   "；".join(f"{a} → {b}" for a, b in _ren.items()))
                    st.session_state[f"bundle_{MODE_ID}"] = None  # 数据角色变化，旧模型不再展示
                    st.success("运行时 Schema 已构建，请前往「模型训练」。")
            if st.session_state.get("lit_schema"):
                _sch = st.session_state["lit_schema"]
                st.caption(f"当前运行时 Schema：X {len(_sch['process_inputs'])} ｜ "
                           f"过程状态 {len(_sch['process_states'])} ｜ 缺陷 {len(_sch['defect_network'])} ｜ "
                           f"性能 {len(_sch['performance_outputs'])} ｜ 熔融 {len(_sch.get('melting_states') or {})}")
                _has_groups = "batch_id" in d.columns and d["batch_id"].nunique() >= 2
                _cv_txt, _cv_kind = suggest_cv(len(d), _has_groups)
                st.caption(f"CV 方式自动建议（小样本）：{_cv_txt}（存在真实独立批次时优先按批次分组；"
                           f"严禁人为制造 batch_id）")

# ---------------- ③ 模型训练 ----------------
with tab2:
    # V1.5 spec 五十八：主标题 + 副标题 + Stage 1.5 小标签（不用超长标题）
    st.subheader("三级代理模型训练")
    render_tab_header("tab2", CTX)
    st.caption("结构 / 工艺 / 材料 → 过程状态 → 缺陷 → 性能 ｜ "
               f"当前模式：{MODE['label']}（{MODE['train_desc']}）")
    st.markdown('<span class="status-pill" style="color:var(--pink-accent);background:#FFF5F6;'
                'border:1px solid var(--pink-light);">'
                'Stage 1.5：颗粒熔融与沉积状态（可选）</span>', unsafe_allow_html=True)

    if MODE_ID == "dual_anode":
        st.markdown("**一级模型：射流与粒子状态预测**")
        st.markdown("结构/工艺参数（+材料属性，若有） → 射流与粒子状态")
        st.markdown("方法：XGBoost + 按喷涂批次分组交叉验证 + OOF（折外预测）")
        st.markdown("预测：电弧电压、射流温度、射流速度、粒子温度、粒子速度")
        st.divider()
        st.markdown("**Stage 1.5（可选）：颗粒熔融与沉积状态预测**")
        st.markdown("粒子热动力状态 + 材料属性 + 关键工艺参数 → 熔融状态/熔融指数/铺展状态")
        st.markdown("方法：XGBoost 回归/分类（数据本身具有熔融标签时自动启用；无标签自动跳过）")
        st.divider()
        st.markdown("**二级模型：涂层缺陷网络预测**")
        st.markdown("工艺参数 + 一级模型 OOF 过程状态（+ 熔融层 OOF，若启用） → 涂层缺陷网络")
        st.markdown("预测：孔隙率、层间未结合率、未熔颗粒比例、裂纹密度、缺陷连通度")
        st.divider()
        st.markdown("**三级模型：涂层性能预测**")
        st.markdown("预测过程状态 + 预测缺陷网络（+ 预测熔融状态，若启用） + 关键工艺参数 → 涂层性能")
        st.markdown("方法：高斯过程回归（GP）或随机森林（RF），并输出预测不确定性")
    else:
        st.markdown(f"当前模式：**{MODE['label']}**。"
                    "链条按可用数据自动裁剪：无过程状态列时直接从输入预测缺陷/性能；"
                    "无熔融标签时 Stage 1.5 自动跳过。")

    d = current_df()
    if d is None:
        st.warning("请先导入数据。")
    else:
        if use_demo:
            st.info(DEMO_RESULT_NOTICE)
        # V1.5 spec 六十六/六十七：小样本提示 + 数据准备检查（禁止直接 traceback）
        from src.quick_analysis import data_readiness_issues
        _crit, _warn = data_readiness_issues(d, schema)
        if _crit or _warn:
            with st.expander("【数据准备检查】", expanded=bool(_crit)):
                for c in _crit:
                    st.error(f"⚠ {c}")
                for w in _warn:
                    st.warning(f"· {w}")
        if len(d) < 10:
            st.warning("当前数据量极少，结果仅适合作为软件功能测试或探索性分析。")
        elif len(d) <= 25:
            st.info("样本量较小（n ≤ 25）：文献/无批次数据将自动使用 LOOCV。")

        if MODE_ID == "literature" and not st.session_state.get("lit_schema"):
            st.warning("文献模式请先在「数据管理」页完成变量选择并构建运行时 Schema。")
        _can_train = not _crit
        if _can_train and st.button("🚀 开始模型训练"):
            st.session_state[f"train_status_{MODE_ID}"] = "running"
            with st.spinner("正在训练与全链交叉验证（每个折叠重新训练各级模型，约需 1-3 分钟，请保持页面打开）..."):
                try:
                    bundle = train_chain_full(d, schema, MODEL)
                    st.session_state[f"bundle_{MODE_ID}"] = bundle
                    st.session_state[f"train_status_{MODE_ID}"] = "done"
                    log_event("train", f"mode={MODE_ID} model={bundle.get('model_version')} "
                                       f"stage15={bundle.get('stage15_enabled')} "
                                       f"dataset_hash={bundle.get('dataset_hash', '')[:16]}")
                    st.success("✓ 模型训练完成"
                               + ("（Stage 1.5 熔融层已启用）" if bundle.get("stage15_enabled")
                                  else "（当前数据缺少熔融标签，Stage 1.5 未启用，主链不受影响）"))
                except Exception as e:
                    st.session_state[f"train_status_{MODE_ID}"] = "fail"
                    st.error(friendly_error(e, d, schema))

        bundle = get_bundle()
        if bundle is not None:
            cv = bundle.get("chain_cv") or {}
            cv_metrics = cv.get("metrics") or {}
            st.markdown("**全链分组交叉验证指标（按喷涂批次分组，同一批次不跨训练/验证）**")
            rows, weak = [], []
            has_cv = any(cv_metrics.get(s) for s in ["stage1", "stage2", "stage3"])
            if has_cv:
                for stage in ["stage1", "stage2", "stage3"]:
                    for t, met in (cv_metrics.get(stage) or {}).items():
                        if stage == "stage3":
                            kind = bundle["stage3_models"].get(t, {}).get("kind", "XGBoost")
                        else:
                            kind = "XGBoost"
                        rows.append({
                            "模型层级": STAGE_LABELS.get(stage, stage),
                            "预测指标": zh(t),
                            "决定系数 R²": round(float(met["R2"]), 4),
                            "均方根误差 RMSE": round(float(met["RMSE"]), 4),
                            "平均绝对误差 MAE": round(float(met["MAE"]), 4),
                            "模型类型": MODEL_KIND_LABELS.get(kind, kind),
                        })
                        if float(met["R2"]) < 0:
                            weak.append(zh(t))
            else:
                # 兼容 V1.1 旧模型文件：一级/二级仍为分组 OOF 指标
                st.caption("当前加载的是旧版本模型（无全链交叉验证记录），以下仅显示一级/二级分组 OOF 指标；建议重新训练。")
                for stage in ["stage1", "stage2"]:
                    for t, met in (bundle.get("metrics", {}).get(stage) or {}).items():
                        rows.append({
                            "模型层级": STAGE_LABELS.get(stage, stage),
                            "预测指标": zh(t),
                            "决定系数 R²": round(float(met["R2"]), 4),
                            "均方根误差 RMSE": round(float(met["RMSE"]), 4),
                            "平均绝对误差 MAE": round(float(met["MAE"]), 4),
                            "模型类型": MODEL_KIND_LABELS.get("XGBoost", "XGBoost"),
                        })
                        if float(met["R2"]) < 0:
                            weak.append(zh(t))
            if rows:
                df_rows = pd.DataFrame(rows)
                # V1.6（P0-4）：R² 总览改走 plot_style 统一色板（QUAL_CYCLE 按 Stage 语义）
                fig_r2 = px.bar(df_rows, x="决定系数 R²", y="预测指标", color="模型层级",
                                orientation="h", title="各预测目标交叉验证 R² 总览",
                                color_discrete_map=pstyle.discrete_map(
                                    ["一级模型", "二级模型", "三级模型"]),
                                custom_data=["均方根误差 RMSE", "平均绝对误差 MAE"])
                fig_r2.update_yaxes(categoryorder="array",
                                    categoryarray=list(df_rows["预测指标"])[::-1])
                fig_r2.add_vline(x=0, line_width=1, line_dash="dot",
                                 line_color=pstyle.GRAY_REF)
                fig_r2.update_traces(text=[f"{v:.3f}" for v in df_rows["决定系数 R²"]],
                                     textposition="outside", cliponaxis=False,
                                     hovertemplate="%{y}<br>R²=%{x:.4f}"
                                                   "<br>RMSE=%{customdata[0]:.4g}<br>MAE=%{customdata[1]:.4g}<extra></extra>")
                # V1.6（P0-4 验收 2）：统计诚信注脚（CV 方式）+ Demo 水印
                pstyle.nature_layout(fig_r2, height=max(pstyle.HERO_HEIGHT - 100,
                                                        34 * len(df_rows) + 90))
                pstyle.add_stat_note(fig_r2, pstyle.stat_note(
                    n=int(len(d)), cv_method=cv.get("method")))
                if use_demo:
                    pstyle.add_demo_watermark(fig_r2)
                st.plotly_chart(fig_r2, width="stretch", key="cv_r2_overview")
                st.dataframe(df_rows, width="stretch", hide_index=True)
                # V1.6（P0-2）：CV 指标表旁挂证据徽标（model 类：R²/RMSE/n/CV）
                _badge_vals = []
                for stage in ["stage1", "stage2", "stage3"]:
                    for t, met in (cv_metrics.get(stage) or {}).items():
                        _oof_s = (cv.get("oof_predictions") or {}).get(stage)
                        _n_t = int(_oof_s[t].dropna().shape[0]) if (
                            _oof_s is not None and hasattr(_oof_s, "columns")
                            and t in getattr(_oof_s, "columns", [])) else int(len(d))
                        _badge_vals.append(f"{zh_short(t)} R²={float(met['R2']):.3f} (n={_n_t})")
                st.markdown(
                    evidence_mod.badge_span(
                        "CV: " + (cv.get("method") or "—") + " ｜ " + " ｜ ".join(_badge_vals[:4])
                        + ("…" if len(_badge_vals) > 4 else ""),
                        source_class="model", demo=use_demo),
                    unsafe_allow_html=True)
                st.caption("徽标数值来自 model_chain.train_chain_full 的全链分组交叉验证"
                           "（OOF 折外预测）；Quick Run 的结论可在首页摘要卡逐条溯源。")
                if st.button("📋 复制模型信息", key="btn_copy_info"):
                    info_lines = [
                        f"模型层级：{r['模型层级']}｜目标：{r['预测指标']}｜模型类型：{r['模型类型']}｜"
                        f"R²：{r['决定系数 R²']}｜RMSE：{r['均方根误差 RMSE']}｜MAE：{r['平均绝对误差 MAE']}｜"
                        f"CV 方法：GroupKFold（按喷涂批次分组）"
                        for r in rows]
                    info_lines.append(f"数据版本：{(bundle.get('dataset_hash') or '-')[:16]}｜"
                                      f"模型版本：{bundle.get('model_version', '-')}")
                    st.code("\n".join(info_lines), language=None)
            if cv.get("method") and cv["method"] != "none":
                st.caption("指标均为外层分组交叉验证结果（每个折叠重新训练三级模型后仅预测验证批次），"
                           "非训练集拟合值；最终部署模型使用全部数据单独训练，用于实际预测。")
            if weak:
                st.warning(f"数据显示{'、'.join(weak)}在当前交叉验证设置下预测能力较弱"
                           "（R² < 0）；建议后续检查样本量、变量覆盖范围及数据噪声。")

            # V1.4：Stage 1.5 熔融层指标（启用时单独展示，不混入三级主链指标表）
            m15 = cv_metrics.get("stage15") or {}
            if m15:
                st.markdown("**Stage 1.5 颗粒熔融与沉积状态（可选层，已启用）**")
                rows15 = []
                for t, met in m15.items():
                    if "R2" in met:
                        rows15.append({"熔融目标": zh(t), "类型": "回归",
                                       "决定系数 R²": round(float(met["R2"]), 4),
                                       "均方根误差 RMSE": round(float(met["RMSE"]), 4),
                                       "平均绝对误差 MAE": round(float(met["MAE"]), 4)})
                    else:
                        rows15.append({"熔融目标": zh(t), "类型": "分类",
                                       "Accuracy": round(float(met["Accuracy"]), 4),
                                       "平衡准确率": round(float(met["Balanced_Accuracy"]), 4),
                                       "F1 (macro)": round(float(met["F1_macro"]), 4)})
                st.dataframe(pd.DataFrame(rows15), width="stretch", hide_index=True)
                st.caption("Stage 1.5 与主链使用同一套按批次分组的防泄漏交叉验证；"
                           "类别不平衡时请重点参考平衡准确率而非 Accuracy。")
            elif bundle.get("stage15_enabled") is False:
                st.caption("当前数据缺少颗粒熔融状态标签，Stage 1.5 未启用。")

        # ---------------- V1.8 P1 · 架构消融对照（直连 vs 双层 vs 缺陷增强链） ----------------
        with st.expander("架构消融对照（课题双层主链 vs 直连基线 vs 缺陷增强三级链）",
                         expanded=False):
            st.caption("在**完全相同的外层分组交叉验证**下比较三种代理模型架构的端到端性能："
                       "① 直连（工艺→性能）；② 双层（工艺→过程状态 OOF→性能，课题主线）；"
                       "③ 缺陷增强链（工艺→过程→缺陷→性能，V1.7 三级链）。"
                       "按方案 §5.2 晋级规则：缺陷层须在多数目标稳定降低误差才晋级为正式路径，"
                       "否则三级链保留为研究支路。")
            _arch_st = arch_mod.architecture_status()
            if _arch_st.get("done"):
                st.markdown(evidence_mod.badge_span(
                    f"晋级判定：{_arch_st.get('promote')} ｜ {_arch_st['generated_at']}"
                    + ("（历史结果）" if _arch_st.get("freshness") != "fresh" else ""),
                    source_class="model", demo=use_demo), unsafe_allow_html=True)
            _d_arch = current_df()
            if _d_arch is None:
                st.info("请先在「② 数据管理」加载数据。")
            elif st.button("▶ 运行架构消融对照（约 3–8 分钟，三种架构各跑一轮外层 CV）",
                           key="arch_run"):
                _prog = st.status("架构消融对照运行中…", expanded=True)
                _arch_lines = []

                def _arch_cb(name, frac=None):
                    _arch_lines.append(f"· {name}")
                    _prog.write("\n".join(_arch_lines[-8:]))

                try:
                    with st.spinner(""):
                        _arch_res = arch_mod.compare_architectures(
                            _d_arch, schema, progress_cb=_arch_cb)
                    rows_arch = []
                    for t, r in _arch_res["comparison"].items():
                        rows_arch.append({
                            "性能目标": zh(t),
                            "直连 R²": r.get("direct"),
                            "双层 R²": r.get("two_layer"),
                            "缺陷增强 R²": r.get("three_chain"),
                            "双层增益": r.get("two_layer_gain"),
                            "缺陷层增益": r.get("defect_gain"),
                            "胜者": {"direct": "直连", "two_layer": "双层",
                                     "three_chain": "缺陷增强"}.get(r.get("winner"), "—"),
                        })
                    st.dataframe(pd.DataFrame(rows_arch), width="stretch", hide_index=True)
                    st.info(_arch_res["promote_text"])
                    st.caption("选型证据声明：本对照分数用于选择正式预测路径（同轮比较），"
                               "非无偏最终性能；被选架构的最终性能以独立外层验证为准。"
                               "完整报告：outputs/architecture/架构对照报告.md")
                    if use_demo:
                        evidence_mod.demo_banner()
                except Exception as e:
                    _prog.update(label="架构对照失败", state="error")
                    st.error(friendly_error(e, current_df(), schema))

        # ---------------- V1.7 · L1 不确定度校准（核心价值验证第一层） ----------------
        st.markdown("---")
        with st.container(border=True):
            st.markdown("**L1 · 不确定度校准（模型给出的 σ 是否可信）**")
            st.caption("带 σ 的全链交叉验证（折划分规则与平台全链 CV 一致）→ 逐目标 1σ/2σ 实际覆盖率"
                       "（期望 ≈68% / ≈95%）→ 校准因子。校准后工艺预测与逆向设计的 σ 自动乘以该因子；"
                       "这是 L2 命中判定与 L3 悲观惩罚的基础。")
            # V1.8（方案 §6.3/§10）：k=1 仅表示未缩放 ≠ 已可信；状态含时效性
            _cal_bundle = get_bundle()
            _cal_status = calib_mod.calibration_status(
                (_cal_bundle or {}).get("model_version") if _cal_bundle else None,
                (CTX.get("dataset_hash") if isinstance(CTX, dict) else None))
            if _cal_status.get("done") and _cal_status.get("freshness") == "fresh":
                st.markdown(evidence_mod.badge_span(
                    f"已校准（当前模型/数据） ｜ {_cal_status['n_targets']} 目标"
                    f"（校准良好 {_cal_status['n_good']}） ｜ {_cal_status['generated_at']}",
                    source_class="model", demo=use_demo), unsafe_allow_html=True)
            elif _cal_status.get("done"):
                st.markdown(evidence_mod.badge_span(
                    f"历史校准结果 ｜ {_cal_status.get('freshness_note', '')}",
                    source_class="model", demo=use_demo), unsafe_allow_html=True)
            else:
                st.caption("当前未校准（无校准文件；k=1 仅表示 σ 未做缩放，不代表不确定度已可信）——"
                           "建议运行校准后再采信区间结论。")
            _d_cal = current_df()
            if _d_cal is None:
                st.info("请先在「② 数据管理」加载数据后再运行校准。")
            elif st.button("▶ 运行 L1 校准（约 1–2 分钟）", type="primary", key="cal_run"):
                _prog = st.status("L1 校准运行中（带 σ 全链交叉验证）…", expanded=True)
                _cal_lines = []

                def _cal_cb(name, frac=None):
                    _cal_lines.append(f"· {name}")
                    _prog.write("\n".join(_cal_lines[-10:]))

                try:
                    _cal_res = calib_mod.run_calibration(
                        _d_cal, schema, bundle=_cal_bundle,
                        dataset_hash=(CTX.get("dataset_hash") if isinstance(CTX, dict) else None),
                        progress_cb=lambda n, f=None: _cal_cb(n))
                    _rows_cal = []
                    for t, info in _cal_res["targets"].items():
                        if "cov_1sigma" in info:
                            _rows_cal.append({
                                "性能目标": zh(t), "有效 n": info["n"],
                                "1σ 覆盖率": f"{info['cov_1sigma']*100:.1f}%",
                                "2σ 覆盖率": f"{info['cov_2sigma']*100:.1f}%",
                                "z 方差": info["z_var"], "校准因子 k": info["factor"],
                                "判定": info["verdict"]})
                        else:
                            _rows_cal.append({"性能目标": zh(t), "有效 n": info.get("n", 0),
                                              "判定": info.get("verdict", "样本不足")})
                    _prog.update(label="L1 校准完成", state="complete")
                    st.dataframe(pd.DataFrame(_rows_cal), width="stretch", hide_index=True)
                    # 校准曲线（网页交互版，plotly 统一模板）
                    import plotly.graph_objects as go
                    _fig_cal = go.Figure()
                    _ok_t = [t for t, i in _cal_res["targets"].items() if "cov_1sigma" in i]
                    for _i, _t in enumerate(_ok_t):
                        _nom, _act = calib_mod.calibration_curve_points(
                            _cal_res["oof_with_std"], _t)
                        if _nom is None:
                            continue
                        _fig_cal.add_trace(go.Scatter(
                            x=_nom, y=_act, mode="lines+markers", name=zh(_t),
                            line=dict(color=pstyle.QUAL_CYCLE[_i % len(pstyle.QUAL_CYCLE)], width=2),
                            marker=dict(size=4)))
                    _fig_cal.add_trace(go.Scatter(
                        x=[0, 100], y=[0, 100], mode="lines", name="理想校准线",
                        line=dict(color=pstyle.GRAY_REF, width=1.5, dash="dash")))
                    pstyle.nature_layout(_fig_cal, title="校准曲线（名义置信度 vs 实际覆盖率）",
                                         height=400)
                    _fig_cal.update_xaxes(title_text="名义置信度 / %", range=[0, 100])
                    _fig_cal.update_yaxes(title_text="实际覆盖率 / %", range=[0, 100])
                    st.plotly_chart(_fig_cal, width="stretch", key="cal_curve_fig")
                    st.markdown(evidence_mod.badge_span(
                        " ".join(f"{zh(t)}: 1σ={i['cov_1sigma']*100:.0f}%, k={i['factor']:.2f}"
                                 for t, i in _cal_res["targets"].items() if "cov_1sigma" in i),
                        source_class="model", demo=use_demo), unsafe_allow_html=True)
                    st.caption("计算方式：calibration.run_calibration（带 σ 全链 CV，折外预测）；"
                               "因子已写入 models/calibration.json，④ 工艺预测 / ⑦ 逆向设计 / "
                               "L3 效率基准的 σ 自动乘以该因子。完整报告：outputs/calibration/校准报告.md")
                    if use_demo:
                        evidence_mod.demo_banner()
                except Exception as e:
                    _prog.update(label="L1 校准失败", state="error")
                    st.error(friendly_error(e, _d_cal, schema))

# ---------------- ④ 工艺预测 ----------------
with tab3:
    st.subheader("单组工艺参数全链预测")
    render_tab_header("tab3", CTX)
    st.markdown("输入一组喷枪结构与工艺参数，系统将依次预测：\n\n"
                "射流/粒子状态 → 涂层缺陷网络 → 涂层性能")
    if get_bundle() is None:
        st.warning("请先完成模型训练。")
    else:
        bundle = get_bundle()
        if use_demo:
            st.info(DEMO_RESULT_NOTICE)

        vals = {}
        # V1.3.2：默认值优先取当前训练数据中位数（截断到设备范围），无数据回退 config.default，
        # 再回退范围中点——确保默认值落在训练数据支持域内，避免"默认即外推"。
        _defaults_pred = compute_default_inputs(schema, current_df())
        st.markdown("**【喷枪结构参数】**")
        qb1, qb2, _ = st.columns([1, 1, 2])
        if qb1.button("↺ 恢复默认参数", key="btn_reset_pred"):
            for sec in ["structure_inputs", "process_inputs"]:
                for col, spec in schema[sec].items():
                    st.session_state["pred_" + col if sec == "structure_inputs" else "predp_" + col] = \
                        float(_defaults_pred[col])
            st.rerun()
        if qb2.button("📊 使用当前数据中位数", key="btn_median_pred"):
            d4 = current_df()
            if d4 is not None:
                for sec in ["structure_inputs", "process_inputs"]:
                    for col, spec in schema[sec].items():
                        if col in d4.columns:
                            med = pd.to_numeric(d4[col], errors="coerce").median()
                            if pd.notna(med):
                                st.session_state["pred_" + col if sec == "structure_inputs" else "predp_" + col] = \
                                    float(np.clip(med, spec["min"], spec["max"]))
                st.rerun()
        st.caption("恢复默认参数优先使用当前训练数据中位数（截断到设备允许范围）；仅当无训练数据时才"
                   "回退配置默认值，再回退配置范围中点。不会修改数据源。")

        cols = st.columns(3)
        i = 0
        for col, spec in schema["structure_inputs"].items():
            if spec.get("categorical") or "min" not in spec or "max" not in spec:
                continue  # 分类列（如 torch_type）不参与数值输入与模型特征
            default = _defaults_pred[col]
            vals[col] = cols[i % 3].number_input(
                zh(col),
                min_value=float(spec["min"]), max_value=float(spec["max"]),
                value=float(default), key="pred_" + col)
            i += 1
        st.markdown("**【喷涂工艺参数】**")
        cols = st.columns(3)
        i = 0
        for col, spec in schema["process_inputs"].items():
            if spec.get("categorical") or "min" not in spec or "max" not in spec:
                continue
            default = _defaults_pred[col]
            vals[col] = cols[i % 3].number_input(
                zh(col),
                min_value=float(spec["min"]), max_value=float(spec["max"]),
                value=float(default), key="predp_" + col)
            i += 1

        # ---------------- V1.8 P1（方案 §6.2）：设计预测 vs 诊断预测两种任务分开 ----------------
        st.markdown("---")
        _pred_mode = st.radio(
            "预测任务类型（两种口径分别报告，不得混用）",
            ["设计预测（默认：只输入喷涂前可确定参数，用于逆向优化）",
             "诊断预测（可填入实测过程状态，评估特定实验/测量信息价值）"],
            index=0, key="pred_mode_radio",
            help="设计预测只用喷涂前可获得的参数做全链推断；诊断预测允许用已实测的"
                 "射流/粒子状态替代 Stage 1 模型预测。两者误差口径不同："
                 "诊断高分不得当成设计预测能力。")
        measured_states = {}
        if _pred_mode.startswith("诊断"):
            st.caption("勾选并填入【已实测】的过程状态（将直接进入 Stage 2/3，跳过 Stage 1 模型预测）。"
                       "未勾选的状态仍由 Stage 1 模型推断。")
            _d_diag = current_df()
            _diag_cols = st.columns(3)
            _k = 0
            for t in bundle["state_cols"]:
                if t not in (schema.get("process_states") or {}):
                    continue
                _dom = (bundle.get("training_domain") or {}).get(t) or {}
                _med = None
                if _d_diag is not None and t in _d_diag.columns:
                    _med = pd.to_numeric(_d_diag[t], errors="coerce").median()
                _c1, _c2 = _diag_cols[_k % 3].columns(2)
                _use = _c1.checkbox(zh_short(t), key=f"diag_use_{t}",
                                    help=f"实测{zh_short(t)}（勾选后按实测值进入下游）")
                _v = _c2.number_input(
                    "值", min_value=-1e6, max_value=1e6,
                    value=float(_med) if pd.notna(_med) else
                    float((_dom.get("min", 0) + _dom.get("max", 1)) / 2),
                    key=f"diag_val_{t}", label_visibility="collapsed")
                if _use:
                    measured_states[t] = float(_v)
                _k += 1
            if measured_states:
                st.info(f"诊断模式：{len(measured_states)} 个实测状态将直接进入下游"
                        f"（{('、'.join(zh_short(t) for t in measured_states))}）——"
                        "本结果为诊断口径，不代表设计预测能力。")
        st.caption("V1.8 铁律：诊断预测的实测状态不进入设计模型训练（训练输入始终只用 OOF 预测），"
                   "两种口径的误差分别报告。")

        # 训练数据覆盖范围检查（范围来源于训练数据，并非设备极限）——V1.6 收进次级证据区
        domain = bundle.get("training_domain") or {}
        if domain:
            with st.expander("次级证据：训练数据覆盖范围检查（范围来源于训练数据，并非设备极限）",
                             expanded=False):
                lines = []
                has_extrap = False
                for col, v in vals.items():
                    dom = domain.get(col)
                    if dom is None:
                        continue
                    status = classify_support(v, dom)
                    if status == "extrap":
                        has_extrap = True
                    lines.append(f'<span style="color:{SUPPORT_COLOR[status]};">●</span> '
                                 f'{zh(col)}：{support_label(status)}'
                                 f'<span style="color:var(--gray-muted);">（训练数据 {dom["min"]:.1f}–{dom["max"]:.1f}，'
                                 f'中位数 {dom["median"]:.1f}）</span>')
                st.markdown("<br>".join(lines), unsafe_allow_html=True)
                if has_extrap:
                    st.error("当前输入超出了模型训练数据覆盖范围，属于外推预测，结果可靠性可能降低。"
                             "如仍在设备允许范围内，可继续预测，但请谨慎解读结果。")
                st.caption("范围分级来源于当前训练数据，并非设备极限。")

        if st.button("开始全链预测"):
            device_violation = \
                [c for c in vals
                 if c in schema["structure_inputs"] and
                 (vals[c] < schema["structure_inputs"][c]["min"] or vals[c] > schema["structure_inputs"][c]["max"])] + \
                [c for c in vals
                 if c in schema["process_inputs"] and
                 (vals[c] < schema["process_inputs"][c]["min"] or vals[c] > schema["process_inputs"][c]["max"])]
            if device_violation:
                st.error("以下参数超出设备允许范围，请修改后再预测：" + "、".join(zh(c) for c in device_violation))
            else:
                try:
                    X = pd.DataFrame([vals])
                    # V1.8（方案 §6.2）诊断模式：实测状态覆盖 Stage 1 预测
                    if measured_states:
                        states, defects, perf, unc, melt = _predict_chain_diagnostic(
                            X, bundle, measured_states)
                    else:
                        states, defects, perf, unc, melt = predict_chain(X, bundle, return_melting=True)
                    std_map = unc.iloc[0].to_dict() if len(unc.columns) else {}
                    # V1.7（L1 联动）：σ 乘 L1 校准因子（无校准文件时因子=1，显示不变）
                    _cal_factors = calib_mod.load_factors()
                    if _cal_factors:
                        std_map = {k: (v * _cal_factors.get(k[:-4], 1.0) if isinstance(v, (int, float))
                                       and k.endswith("_std") and np.isfinite(v) else v)
                                   for k, v in std_map.items()}
                    log_event("predict", f"mode={MODE_ID} inputs={ {k: round(float(v), 2) for k, v in vals.items()} }")

                    st.markdown("**【第一阶段：射流与粒子状态】**")
                    st.dataframe(stage_result_table(bundle["state_cols"], states.iloc[0]),
                                 width="stretch", hide_index=True)
                    st.markdown('<p class="chain-note">↓</p>', unsafe_allow_html=True)

                    # V1.4：Stage 1.5 熔融层（未启用时明确提示，主链继续）
                    if bundle.get("stage15_enabled") and melt is not None:
                        st.markdown("**【Stage 1.5：颗粒熔融与沉积状态】**")
                        rows_m = []
                        for t in melt["continuous"].columns:
                            v_m = fmt_value(t, melt["continuous"].iloc[0][t])
                            if t == "overheating_risk" and isinstance(v_m, (int, float)):
                                lv = "低" if v_m < 0.33 else ("中" if v_m < 0.66 else "高")
                                rows_m.append({"熔融指标": zh(t),
                                               "预测值": f"{v_m}（风险等级参考：{lv}）"})
                            else:
                                rows_m.append({"熔融指标": zh(t), "预测值": v_m})
                        for t in melt["class"].columns:
                            raw = str(melt["class"].iloc[0][t])
                            rows_m.append({"熔融指标": zh(t),
                                           "预测值": MELTING_CLASS_ZH.get(raw, raw)})
                        st.dataframe(pd.DataFrame(rows_m), width="stretch", hide_index=True)
                        st.markdown('<p class="chain-note">↓</p>', unsafe_allow_html=True)
                    else:
                        st.info("当前模型未建立颗粒熔融状态预测。")

                    st.markdown("**【第二阶段：涂层缺陷网络】**")
                    st.dataframe(stage_result_table(bundle["defect_cols"], defects.iloc[0]),
                                 width="stretch", hide_index=True)
                    st.markdown('<p class="chain-note">↓</p>', unsafe_allow_html=True)

                    st.markdown("**【第三阶段：涂层性能】**")
                    st.dataframe(stage_result_table(bundle["perf_cols"], perf.iloc[0], std_map),
                                 width="stretch", hide_index=True)
                    if _cal_factors:
                        _k_show = "、".join(f"{zh(t)} k={k:.2f}" for t, k in _cal_factors.items())
                        st.caption(f"σ 已按 L1 校准因子修正（{_k_show}）；"
                                   "校准来自带 σ 的全链交叉验证（③ 模型训练 → L1）。")

                    # V1.5 spec 四十一 + V1.6 四段式：历史数据参考收进次级证据区
                    with st.expander("次级证据：历史数据参考（最近 3–5 个真实实验/文献数据）",
                                     expanded=False):
                        d_ref = current_df()
                        near = insight_mod.nearest_experiments(vals, d_ref,
                                                               input_cols=list(vals.keys()), k=5)
                        if len(near):
                            st.dataframe(near, width="stretch", hide_index=True)
                            domain_ref = bundle.get("training_domain") or {}
                            if domain_ref:
                                bad_cols = [c for c, v in vals.items()
                                            if c in domain_ref and not (domain_ref[c]["min"] <= v <= domain_ref[c]["max"])]
                                edge_cols = [c for c, v in vals.items()
                                             if c in domain_ref and (v < domain_ref[c]["q1"] or v > domain_ref[c]["q3"])
                                             and domain_ref[c]["min"] <= v <= domain_ref[c]["max"]]
                                if bad_cols:
                                    st.warning("当前输入相对历史数据存在外推风险：" + "、".join(zh(c) for c in bad_cols))
                                elif edge_cols:
                                    st.caption("当前输入接近历史数据边界：" + "、".join(zh(c) for c in edge_cols))
                                else:
                                    st.caption("当前输入位于历史数据覆盖范围内（数据支持充分）。")
                        else:
                            st.caption("当前无可参考的历史真实数据。")
                except Exception as e:
                    st.error(friendly_error(e, current_df(), schema))

# ---------------- ⑤ 模型解析 ----------------
with tab4:
    st.subheader("模型解析与关键因素分析")
    render_tab_header("tab4", CTX)
    if get_bundle() is None:
        st.warning("请先训练模型。")
    else:
        bundle = get_bundle()
        if use_demo:
            st.info(DEMO_RESULT_NOTICE)

        stage_disp = st.selectbox("模型层级", list(STAGE_FULL_LABELS.values()))
        stage_key = {v: k for k, v in STAGE_FULL_LABELS.items()}[stage_disp]

        if stage_key == "stage3":
            st.info("当前模型暂不支持该解释方式。")
        else:
            model_dict = bundle[f"{stage_key}_models"]
            if not model_dict:
                st.info("当前模型暂不支持该解释方式。")
            else:
                target_map = {zh(t): t for t in model_dict.keys()}
                target_disp = st.selectbox("预测目标", list(target_map.keys()))
                target = target_map[target_disp]

                pipe = model_dict[target]
                booster = pipe.named_steps["model"]
                names = bundle[f"{stage_key}_features"]
                # V1.3.2：网页预览默认 Top 15 并限制图形宽度、减少右侧空白；
                # 论文导出图尺寸由 paper_output / paper_style 单独控制，不受网页预览影响。
                topn_disp = st.selectbox("网页预览特征数量（Top N）", [15, 10, 20, 30],
                                         index=0, key="an_topn")
                imp = pd.DataFrame({
                    "影响因素": [zh(f) for f in names],
                    "特征重要性": booster.feature_importances_,
                }).sort_values("特征重要性", ascending=False).head(topn_disp)

                st.markdown("**XGBoost 特征重要性**")
                # V1.6（P0-4/P1-4）：统一图入口 + 统计注脚 + Demo 水印 + 下载三件套
                fig = pstyle.importance_chart(
                    imp, title=f"{zh_short(target)}关键影响因素排序 Top {len(imp)}")
                # V1.8（方案 §2 诊断③）：n 取「该模型训练时的样本量」（fold_info 真实记录），
                # 当前数据未加载时如实标「未知」，严禁填 0 冒充有效统计。
                _df_an = current_df()
                _n_an = None
                try:
                    _fi = (get_bundle().get("chain_cv") or {}).get("fold_info") or []
                    if _fi:
                        _n_an = int(max(f.get("n_train", 0) for f in _fi))
                except Exception:
                    _n_an = None
                if _n_an is None and _df_an is not None:
                    _n_an = int(len(_df_an))
                from src.status_binding import fmt_n
                pstyle.add_stat_note(fig, pstyle.stat_note(
                    n=_n_an,
                    cv_method=(get_bundle().get("chain_cv") or {}).get("method")))
                if _n_an is None:
                    st.caption("注脚样本量：未知（当前未加载数据，且模型未记录训练样本量）。"
                               "V1.8 规则：未知不填 0。")
                if use_demo:
                    pstyle.add_demo_watermark(fig)
                try:
                    st.plotly_chart(fig, width="stretch", key="an_imp_fig")
                except Exception:
                    st.plotly_chart(fig, key="an_imp_fig")
                pstyle.fig_download_row(fig, imp, f"Fig_Importance_{target}", key="an_imp_dl")

        # ---------------- V1.4：颗粒熔融与沉积状态分析（Stage 1.5） ----------------
        st.markdown("---")
        st.markdown("### 颗粒熔融与沉积状态分析")
        if not (bundle.get("stage15_enabled") and bundle.get("stage15_models")):
            st.info("当前数据缺少颗粒熔融状态标签，Stage 1.5 未启用。")
        else:
            st.success("Stage 1.5：已启用（数据本身具有熔融状态标签）")
            kinds = bundle.get("stage15_kind") or {}
            m15 = ((bundle.get("chain_cv") or {}).get("metrics") or {}).get("stage15") or {}
            _oof_all = (bundle.get("chain_cv") or {}).get("oof_predictions") or {}
            oof15 = _oof_all.get("stage15") if _oof_all.get("stage15") is not None else {}
            rows15 = []
            for t, met in m15.items():
                kind_txt = "回归" if kinds.get(t) == "regression" else "分类"
                n15 = int(oof15[t].dropna().shape[0]) if t in oof15 else "-"
                if "R2" in met:
                    rows15.append({"熔融目标": zh(t), "模型类型": "XGBoost 回归", "CV 样本量": n15,
                                   "决定系数 R²": round(float(met["R2"]), 4),
                                   "均方根误差 RMSE": round(float(met["RMSE"]), 4),
                                   "平均绝对误差 MAE": round(float(met["MAE"]), 4)})
                else:
                    rows15.append({"熔融目标": zh(t), "模型类型": "XGBoost 分类", "CV 样本量": n15,
                                   "Accuracy": round(float(met["Accuracy"]), 4),
                                   "平衡准确率": round(float(met["Balanced_Accuracy"]), 4),
                                   "F1 (macro)": round(float(met["F1_macro"]), 4)})
            st.dataframe(pd.DataFrame(rows15), width="stretch", hide_index=True)
            cvm = (bundle.get("chain_cv") or {}).get("method") or "-"
            st.caption(f"CV 方式：{cvm}；评价仅使用折外（OOF）预测，与主链同一防泄漏原则。")

            d_an = current_df()
            if d_an is not None:
                # 核心图 1：颗粒温度—速度状态图（按真实类别着色；无类别则纯散点）
                _tcol = next((c for c in d_an.columns if "particle_temperature" in str(c).lower()), None)
                _vcol = next((c for c in d_an.columns if "particle_velocity" in str(c).lower()), None)
                _cls = next((c for c in ["melting_state_class", "splat_state_class"]
                             if c in d_an.columns and d_an[c].notna().sum() > 0), None)
                if _tcol and _vcol:
                    _tt = pd.to_numeric(d_an[_tcol], errors="coerce")
                    _vv = pd.to_numeric(d_an[_vcol], errors="coerce")
                    _dd = d_an.loc[_tt.notna() & _vv.notna()].copy()
                    _dd["_T"], _dd["_V"] = _tt[_dd.index], _vv[_dd.index]
                    if _cls:
                        _dd["_cls"] = _dd[_cls].astype(str).map(
                            lambda x: MELTING_CLASS_ZH.get(x, x))
                        _cls_levels = list(pd.unique(_dd["_cls"]))
                        fig_tv = px.scatter(_dd, x="_V", y="_T", color="_cls",
                                            labels={"_V": zh(_vcol), "_T": zh(_tcol), "_cls": "熔融状态"},
                                            title="颗粒温度—速度状态图（按熔融状态类别着色，类别来自数据标签）",
                                            color_discrete_map=pstyle.discrete_map(_cls_levels),
                                            hover_data=[c for c in ["melting_index", "melting_fraction_pct"]
                                                        if c in _dd.columns])
                    else:
                        fig_tv = px.scatter(_dd, x="_V", y="_T",
                                            labels={"_V": zh(_vcol), "_T": zh(_tcol)},
                                            title="颗粒温度—速度状态图（无熔融类别标签，仅散点，不划分熔融区）")
                    pstyle.nature_layout(fig_tv, height=pstyle.HERO_HEIGHT)
                    if use_demo:
                        pstyle.add_demo_watermark(fig_tv)
                    st.plotly_chart(fig_tv, width="stretch", key="h_tv_map")
                else:
                    st.caption("当前数据不足，无法生成该图。（颗粒温度—速度状态图需要粒子温度/速度列）")

                # 核心图 2：熔融状态分布（仅有真实类别标签时）
                if _cls:
                    cnt = d_an[_cls].dropna().astype(str).value_counts()
                    dist_df = pd.DataFrame({
                        "熔融状态": [MELTING_CLASS_ZH.get(c, c) for c in cnt.index],
                        "样本数": cnt.values,
                    })
                    dist_df["百分比（%）"] = (dist_df["样本数"] / dist_df["样本数"].sum() * 100).round(1)
                    fig_dist = px.bar(dist_df, x="熔融状态", y="样本数", color="熔融状态",
                                      title="颗粒熔融状态分布（类别来自数据/文献标签）",
                                      text=dist_df["百分比（%）"].astype(str) + "%",
                                      color_discrete_map=pstyle.discrete_map(list(dist_df["熔融状态"])))
                    pstyle.nature_layout(fig_dist, height=pstyle.SUB_HEIGHT, legend=None)
                    if use_demo:
                        pstyle.add_demo_watermark(fig_dist)
                    st.plotly_chart(fig_dist, width="stretch", key="h_dist")
                else:
                    st.caption("当前数据不足，无法生成该图。（熔融状态分布需要类别标签列）")

                # 核心图 3：熔融模型评价（连续→Parity OOF；分类→混淆矩阵 OOF）——次级证据区
                with st.expander("次级证据：熔融模型评价（Parity / 混淆矩阵，OOF）", expanded=False):
                    _cont15 = [t for t, k in kinds.items() if k == "regression" and t in oof15]
                    for t in _cont15:
                        o = oof15[t].dropna()
                        a = pd.to_numeric(d_an.loc[o.index, t], errors="coerce") if t in d_an.columns else None
                        if a is None or a.notna().sum() < 2:
                            st.caption(f"当前数据不足，无法生成 {zh(t)} 的 Parity 图。")
                            continue
                        met = m15.get(t) or {}
                        _p_df = pd.DataFrame({"实验值": a.values, "OOF 预测值": o.values}).dropna()
                        fig_p = pstyle.parity_chart(_p_df, x="实验值", y="OOF 预测值",
                                                    x_label="实验值", y_label="OOF 预测值",
                                                    title=f"{zh(t)}：实验值 vs OOF 预测值")
                        pstyle.add_stat_note(fig_p, pstyle.stat_note(
                            n=int(len(_p_df)), r2=met.get("R2"), rmse=met.get("RMSE"),
                            mae=met.get("MAE"), cv_method=cvm))
                        if use_demo:
                            pstyle.add_demo_watermark(fig_p)
                        st.plotly_chart(fig_p, width="stretch", key=f"h_parity_{t}")
                    _cat15 = [t for t, k in kinds.items() if k == "classification" and t in oof15]
                    for t in _cat15:
                        o = oof15[t].dropna()
                        if t not in d_an.columns:
                            continue
                        a_lab = d_an.loc[o.index, t].astype(str)
                        p_lab = o.astype(str)
                        classes = sorted(set(a_lab.unique()) | set(p_lab.unique()))
                        cm = pd.crosstab(a_lab, p_lab).reindex(index=classes, columns=classes, fill_value=0)
                        fig_cm = px.imshow(cm.values, x=classes, y=classes, text_auto=True,
                                           labels={"x": "OOF 预测类别", "y": "真实类别"},
                                           title=f"{zh(t)}：混淆矩阵（OOF，非训练集拟合值）",
                                           color_continuous_scale="Blues")
                        pstyle.nature_layout(fig_cm, height=pstyle.SUB_HEIGHT)
                        if use_demo:
                            pstyle.add_demo_watermark(fig_cm)
                        st.plotly_chart(fig_cm, width="stretch", key=f"h_cm_{t}")

                # 核心图 4/5：熔融 → 缺陷 / 性能（关联趋势，非因果）——次级证据区
                _melt_col = "melting_index" if "melting_index" in d_an.columns else \
                    ("melting_fraction_pct" if "melting_fraction_pct" in d_an.columns else None)
                if _melt_col:
                    _pairs_d = [c for c in ["porosity_pct", "lamellar_gap_pct",
                                            "unmelted_particle_pct", "defect_connectivity_index"]
                                if c in d_an.columns]
                    _pairs_p = [c for c in ["bond_strength_MPa", "hardness_HV",
                                            "wear_rate_mg_m", "coupled_damage_rate"]
                                if c in d_an.columns]
                    for _tag, _pairs, _ttl in [
                            ("def", _pairs_d, "熔融指数 → 缺陷指标（关联趋势，非因果）"),
                            ("perf", _pairs_p, "熔融指数 → 涂层性能（关联趋势，非因果）")]:
                        if not _pairs:
                            st.caption(f"当前数据不足，无法生成该图。（{_ttl}：缺少对应列）")
                            continue
                        _mx = pd.to_numeric(d_an[_melt_col], errors="coerce")
                        _rows = {"_mx": _mx}
                        for _c in _pairs:
                            _rows[zh(_c)] = pd.to_numeric(d_an[_c], errors="coerce")
                        _cdf = pd.DataFrame(_rows).dropna()
                        if _cdf.empty:
                            st.caption(f"当前数据不足，无法生成该图。（{_ttl}）")
                            continue
                        with st.expander(f"次级证据：{_ttl}", expanded=False):
                            fig_c = px.scatter(_cdf, x="_mx", y=list(_cdf.columns[1:]),
                                               labels={"_mx": zh(_melt_col), "value": "指标值", "variable": "指标"},
                                               title=_ttl)
                            pstyle.nature_layout(fig_c, height=pstyle.SUB_HEIGHT)
                            if use_demo:
                                pstyle.add_demo_watermark(fig_c)
                            st.plotly_chart(fig_c, width="stretch", key=f"h_corr_{_tag}")
                else:
                    st.caption("当前数据不足，无法生成该图。（缺少 melting_index / melting_fraction_pct）")

                # V1.4 spec 18：多材料归一化热状态 temperature_ratio（仅热状态指标，非熔融分数）
                if "melting_temperature_C" in d_an.columns and _tcol:
                    _mt = pd.to_numeric(d_an["melting_temperature_C"], errors="coerce")
                    _pt = pd.to_numeric(d_an[_tcol], errors="coerce")
                    _ratio = (_pt + 273.15) / (_mt + 273.15)
                    if _ratio.notna().sum() >= 3:
                        fig_r = px.histogram(x=_ratio.dropna(), nbins=30,
                                             labels={"x": "粒子温度 / 材料熔点温度比",
                                                     "y": "样本数"},
                                             title="粒子温度 / 材料熔点温度比（归一化热状态指标，非熔融分数）")
                        pstyle.nature_layout(fig_r, height=pstyle.SUB_HEIGHT, legend=None)
                        if use_demo:
                            pstyle.add_demo_watermark(fig_r)
                        st.plotly_chart(fig_r, width="stretch", key="h_ratio")
                        st.caption("temperature_ratio = T_particle(K) / T_melting(K)，"
                                   "仅作为热状态归一化指标，不得解释为熔融分数。")

                # Stage 1.5 特征重要性（XGBoost）
                _m15_models = bundle.get("stage15_models") or {}
                _t0 = next((t for t, k in kinds.items() if k == "regression"
                            and hasattr(_m15_models.get(t), "named_steps")), None)
                if _t0:
                    _pipe = _m15_models[_t0]
                    _imp = pd.Series(_pipe.named_steps["model"].feature_importances_,
                                     index=bundle.get("stage15_features") or []).sort_values(ascending=False).head(15)
                    _imp_df = pd.DataFrame({"影响因素": [zh(f) for f in _imp.index],
                                            "特征重要性": _imp.values})
                    fig_i = pstyle.importance_chart(
                        _imp_df, title=f"影响{zh(_t0)}的因素（Stage 1.5 XGBoost 重要性 Top {len(_imp)}）")
                    if use_demo:
                        pstyle.add_demo_watermark(fig_i)
                    st.plotly_chart(fig_i, width="stretch", key="h_imp")

        # ---------------- V1.7 · L4 物理一致性检验（核心价值验证第四层） ----------------
        st.markdown("---")
        with st.container(border=True):
            st.markdown("**L4 · 物理合理性诊断（模型规律是否符合喷涂物理）**")
            st.caption("对每条物理期望（config/physics_expectations.yaml，可自定义）做两类证据交叉诊断："
                       "① 事实行 Spearman 相关符号（直接观测）；② 对应层级模型 PDP 部分依赖曲线的"
                       " Theil-Sen 稳健斜率符号（模型推断）。四级判定：支持 / 证据不足 / 局部相反 / 超出适用域。"
                       "【V1.8 声明】两类证据同源（同一数据、同一模型），不构成独立机理验证；"
                       "期望方向本身是待检验假设，非预写结论。")
            _ph_st = phys_mod.physics_status(
                (bundle or {}).get("model_version") if bundle else None,
                (CTX.get("dataset_hash") if isinstance(CTX, dict) else None))
            if _ph_st.get("done") and _ph_st.get("freshness") == "fresh":
                _s = _ph_st["summary"]
                st.markdown(evidence_mod.badge_span(
                    f"支持 {_s['SUPPORT']} ｜ 证据不足 {_s['INSUFFICIENT']}"
                    f" ｜ 局部相反 {_s['OPPOSITE']} ｜ {_ph_st['generated_at']}",
                    source_class="model", demo=use_demo), unsafe_allow_html=True)
            elif _ph_st.get("done"):
                st.markdown(evidence_mod.badge_span(
                    f"历史诊断结果 ｜ {_ph_st.get('freshness_note', '')}",
                    source_class="model", demo=use_demo), unsafe_allow_html=True)
            if st.button("▶ 运行 L4 物理合理性诊断", type="primary", key="phys_run"):
                _d_ph = current_df()
                if _d_ph is None:
                    st.info("请先在「② 数据管理」加载数据。")
                else:
                    try:
                        with st.spinner("逐条计算 Spearman 与 PDP 稳健斜率…"):
                            _ph_res = phys_mod.run_physics_check(_d_ph, bundle, schema)
                        rows_ph = []
                        for v in _ph_res["verdicts"]:
                            rows_ph.append({
                                "#": v["idx"] + 1,
                                "物理期望": f"{zh(v['x'])} → {zh(v['y'])}（期望{v['dir']}）",
                                "Spearman ρ": (f"{v['spearman']:+.2f}" if v["spearman"] is not None else "不可用"),
                                "n": v["n_pairs"] or "—",
                                "PDP 斜率": (f"{v['pdp_slope']:+.3g}" if v["pdp_slope"] is not None else "不可用"),
                                "判定": phys_mod.VERDICT_ZH.get(v["verdict"], v["verdict"]),
                                "适用范围": v.get("scope", "—"),
                            })
                        st.dataframe(pd.DataFrame(rows_ph), width="stretch", hide_index=True)
                        _s = _ph_res["summary"]
                        st.markdown(evidence_mod.badge_span(
                            f"n={_s['n_checkable']} 可检验 ｜ 支持 {_s['SUPPORT']}"
                            f" ｜ 证据不足 {_s['INSUFFICIENT']}"
                            + (f" ｜ 合理率 {_s['pass_ratio']*100:.0f}%" if _s["pass_ratio"] is not None else ""),
                            source_class="model", demo=use_demo), unsafe_allow_html=True)
                        st.caption("计算方式：physics_check.run_physics_check；数据证据仅统计"
                                   " experimental / literature / CFD 事实行；模型证据对 GPR / RF / XGBoost "
                                   "统一采用 PDP（GPR 无 TreeSHAP 的严谨替代）。"
                                   "完整报告：outputs/physics/物理合理性诊断报告.md（含局部相反项原因与补实验建议）。"
                                   "【V1.8】两类证据同源，本检验为合理性诊断，不构成独立机理验证。")
                        _fails = [v for v in _ph_res["verdicts"] if v["verdict"] == "OPPOSITE"]
                        if _fails:
                            with st.expander(f"局部相反条目原因分析与补实验建议（{len(_fails)} 条）",
                                             expanded=True):
                                for v in _fails:
                                    st.markdown(f"**{zh(v['x'])} → {zh(v['y'])}**")
                                    for _n_ in v["notes"]:
                                        st.caption(f"可能原因：{_n_}")
                                    for _a_ in v["advice"]:
                                        st.caption(f"补实验建议：{_a_}")
                        if use_demo:
                            evidence_mod.demo_banner()
                    except Exception as e:
                        st.error(friendly_error(e, current_df(), schema))

# ---------------- ⑥ 数据洞察与实验反馈（V1.5 新增；V1.6 证据徽标化） ----------------
# ⚠️ tab 变量错位警示：tab5 = ⑥ 数据洞察、tab6 = ⑦ 逆向设计、tab7 = ⑧ 结果输出
with tab5:
    st.subheader("数据洞察与实验反馈")
    render_tab_header("tab5", CTX)
    st.caption("三源分级：【直接观测】实验/文献实测 ｜【模型推断】预测/SHAP/PDP ｜【优化建议】Pareto 与实验建议。"
               "model_prediction / optimization_candidate 不会参与真实事实统计，也不会自动进入训练集。")
    d_in = current_df()
    if d_in is None:
        st.warning("请先在「数据管理」页加载数据。")
    else:
        if use_demo:
            evidence_mod.demo_banner()

        with st.container(border=True):
            st.markdown("**【直接观测】数据覆盖**")
            _cov_stats = insight_mod.coverage_stats(d_in, schema)
            st.caption(insight_mod.coverage_text(d_in, schema))
            # V1.6（P0-2）：direct 类证据徽标（界面态无 run registry，用轻量徽标展示关键数值）
            st.markdown(evidence_mod.badge_span(
                f"n={_cov_stats['n_fact']} ｜ 输入变量 {len(_cov_stats['coverage'])} 个"
                + (f" ｜ 重复测量试样 {_cov_stats['repeat_samples']} 个"
                   if _cov_stats.get("repeat_samples") else ""),
                source_class="direct", demo=use_demo), unsafe_allow_html=True)
            st.caption("计算方式：insights.coverage_stats（仅 experimental/literature/CFD 行参与事实统计）")
            cov_df = _cov_stats["coverage"]
            if len(cov_df):
                cov_disp = cov_df.copy()
                cov_disp["变量"] = cov_disp["变量"].map(zh)
                st.dataframe(cov_disp, width="stretch", hide_index=True)

        with st.container(border=True):
            st.markdown("**【直接观测】正向实验规律（相关趋势，非因果）**")
            corr_df = insight_mod.correlation_report(d_in, schema)
            if len(corr_df):
                disp_corr = corr_df.copy()
                disp_corr["输入"] = disp_corr["输入"].map(zh)
                disp_corr["输出"] = disp_corr["输出"].map(zh)
                st.dataframe(disp_corr, width="stretch", hide_index=True)
                # V1.6（P0-2）：direct 类证据徽标（首位相关对真实数值）
                _r0c = corr_df.iloc[0]
                st.markdown(evidence_mod.badge_span(
                    f"n={int(_r0c['n'])} ｜ Pearson={float(_r0c['Pearson']):+.3f}"
                    f" ｜ Spearman={float(_r0c['Spearman']):+.3f}",
                    source_class="direct", demo=use_demo), unsafe_allow_html=True)
                st.caption("计算方式：insights.correlation_report；仅基于 experimental / literature / CFD 行计算；"
                           "未做显著性检验，不使用统计意义上的“显著”，不构成因果结论。")
                if st.button("生成科研解读（变量相关性）", key="ins_corr_an"):
                    md = analyze_figure(
                        "corr", target="变量相关性", n=int(len(d_in)),
                        pearson=float(corr_df.iloc[0]["Pearson"]),
                        spearman=float(corr_df.iloc[0]["Spearman"]),
                        trend_desc=f"数据显示 {zh(corr_df.iloc[0]['输入'])} 与 "
                                   f"{zh(corr_df.iloc[0]['输出'])} 之间存在相关趋势。",
                        demo=use_demo,
                        eid="fig.ins_corr_top", registry=None)  # 界面态无 run registry
                    st.markdown(md)
            else:
                st.caption("当前数据量不足以计算稳定的相关趋势。")

        b_in = get_bundle()
        with st.container(border=True):
            st.markdown("**【模型推断】模型规律与关键变量**")
            if b_in is None:
                st.caption("尚未训练模型；请先在「模型训练」页训练。")
            else:
                factors, stability = insight_mod.model_key_factors(b_in)
                if factors:
                    k0 = list(factors.keys())[0]
                    imp_df = factors[k0].reset_index()
                    imp_df.columns = ["特征", "重要性"]
                    imp_df["特征"] = imp_df["特征"].map(zh)
                    st.markdown(f"当前模型（{k0}）特征重要性 Top {len(imp_df)}：")
                    st.dataframe(imp_df, width="stretch", hide_index=True)
                    # V1.6（P0-2）：model 类证据徽标（必须带 CV 方式限定）
                    st.markdown(evidence_mod.badge_span(
                        f"CV: {(b_in.get('chain_cv') or {}).get('method') or '—'} ｜ "
                        f"Top1 {imp_df.iloc[0]['特征']}（{float(imp_df.iloc[0]['重要性']):.3f}）",
                        source_class="model", demo=use_demo), unsafe_allow_html=True)
                    if stability:
                        st.markdown("**跨折稳定的重要变量**（模型中较稳定的重要变量）：")
                        st.caption("；".join(f"{zh(k)}（{v}）" for k, v in stability.items()))
                    else:
                        st.caption("（跨折特征重要性记录不足 3 折，稳定性统计暂缺）")
                    st.caption("计算方式：insights.model_key_factors；模型重要性仅表征统计关联"
                               "（“在当前模型中……”），不构成因果关系。")
                else:
                    st.caption("当前模型无可用特征重要性（例如 GP / RF 目标）。")

        with st.container(border=True):
            st.markdown("**【需进一步验证】值得复核的数据**")
            anom = insight_mod.anomaly_review(d_in, b_in)
            if len(anom):
                st.dataframe(anom, width="stretch", hide_index=True)
                _a0 = anom.iloc[0]
                st.markdown(evidence_mod.badge_span(
                    f"复核建议 {len(anom)} 条 ｜ 首条：{_a0['对象']}（{_a0['目标']}）",
                    source_class="direct", demo=use_demo), unsafe_allow_html=True)
                st.caption("计算方式：insights.anomaly_review；以上仅为复核建议；平台不会自动删除或修改任何数据。")
            else:
                st.caption("未检测到明显异常（或当前无模型 OOF 记录）。")

        with st.container(border=True):
            st.markdown("**【优化建议】建议补充实验区域**")
            gaps = insight_mod.data_gap_suggestions(d_in, schema, b_in)
            if len(gaps):
                st.dataframe(gaps, width="stretch", hide_index=True)
                st.markdown(evidence_mod.badge_span(
                    f"建议补充区域 {len(gaps)} 处（稀疏/边界/空白参数区）",
                    source_class="direct", demo=use_demo), unsafe_allow_html=True)
                st.caption("计算方式：insights.data_gap_suggestions；仅为数据采集建议，"
                           "最终实验方案由研究者确定。")
            else:
                st.caption("当前数据覆盖较均匀，暂无明显空白区建议。")

        if "data_origin" in d_in.columns:
            st.caption(f"data_origin 分布：{d_in['data_origin'].value_counts().to_dict()} — "
                       "仅 experimental / literature / CFD 行参与事实统计；"
                       "model_prediction / optimization_candidate 永不自动进入训练集。")

        # ---------------- V1.7 · L2 推荐-验证闭环（核心价值验证第二层） ----------------
        st.markdown("---")
        with st.container(border=True):
            st.markdown("**L2 · 推荐-验证闭环（推荐即命中）**")
            _l2_flow = [
                ("1 导出任务单", "⑦ 逆向设计寻优后点击「导出验证任务单」"),
                ("2 安排实验", "按任务单中的推荐工艺参数做喷涂实验"),
                ("3 回填实测值", "在任务单 Excel 的状态/缺陷/性能列填入实测值"),
                ("4 上传判定", "在下方上传，自动判定 |实测-预测| ≤ k·σ_cal 命中"),
            ]
            _fcols = st.columns(4)
            for _fc, (_ft, _fd) in zip(_fcols, _l2_flow):
                _fc.markdown(f'<div class="flow-card"><h4>{_ft}</h4><p>{_fd}</p></div>',
                             unsafe_allow_html=True)
            _vt_st = vtrack_mod.validation_status(
                (get_bundle() or {}).get("model_version") if get_bundle() else None,
                (CTX.get("dataset_hash") if isinstance(CTX, dict) else None))
            if _vt_st.get("done"):
                _h1 = _vt_st.get("last_hit_1sigma")
                _sp = _vt_st.get("last_spec_pass_rate")
                _stale = _vt_st.get("freshness") != "fresh"
                st.markdown(evidence_mod.badge_span(
                    ("历史判定（推荐基线已过期）" if _stale else "已判定")
                    + f" {_vt_st['n_rounds']} 轮 ｜ 最新 1σ 覆盖率 "
                    + (f"{_h1*100:.0f}%" if _h1 is not None else "待填写")
                    + (f" ｜ 工程达标率 {_sp*100:.0f}%" if _sp is not None else "")
                    + f" ｜ {_vt_st['last_judged_at']}",
                    source_class="direct", demo=use_demo), unsafe_allow_html=True)
            _vt_file = st.file_uploader("上传已填实测值的验证任务单（Excel）",
                                        type=["xlsx"], key="vt_upload")
            if _vt_file is not None and st.button("▶ 判定（双口径：区间覆盖 + 工程达标）",
                                                   type="primary", key="vt_judge"):
                try:
                    _vt_df = pd.read_excel(_vt_file, sheet_name=0)
                    with st.spinner("匹配候选预测并逐目标判定（冻结预测基线）…"):
                        _vt_rep = vtrack_mod.judge_validation(
                            _vt_df, demo=use_demo, spec=vtrack_mod.DEFAULT_SPEC,
                            bundle=get_bundle())
                    if _vt_rep.get("hit_rate_1sigma") is not None:
                        _r1, _r2 = _vt_rep["hit_rate_1sigma"], _vt_rep["hit_rate_2sigma"]
                        st.success(f"【口径一 · 区间覆盖】{_vt_rep['n_judged_pairs']} 个目标-样本对："
                                   f"1σ {_r1*100:.0f}% ｜ 2σ {_r2*100:.0f}%")
                        if _vt_rep.get("spec_pass_rate") is not None:
                            st.info(f"【口径二 · 工程达标】单目标达标率 "
                                    f"{_vt_rep['spec_pass_rate']*100:.0f}%"
                                    + (f" ｜ 联合达标 {_vt_rep['joint_pass_count']}/"
                                       f"{_vt_rep['joint_den']}"
                                       if _vt_rep.get("joint_den") else " ｜ 联合达标：未完成（缺测）"))
                        st.markdown(evidence_mod.badge_span(
                            f"1σ 覆盖 {_r1*100:.0f}% ｜ 2σ 覆盖 {_r2*100:.0f}%"
                            + (f" ｜ 达标 {_vt_rep['spec_pass_rate']*100:.0f}%"
                               if _vt_rep.get("spec_pass_rate") is not None else "")
                            + f" ｜ 判定对数 {_vt_rep['n_judged_pairs']}",
                            source_class="direct", demo=use_demo), unsafe_allow_html=True)
                        if _vt_rep.get("calibration_note"):
                            st.caption(f"⚠ {_vt_rep['calibration_note']}")
                        if _r1 < 0.6:
                            _wt = _vt_rep.get("worst_target")
                            _wt_txt = f"（偏差最大目标：{zh(_wt)}）" if _wt else ""
                            st.warning(f"1σ 覆盖率低于 60%{_wt_txt}：建议检查 L1 校准因子是否已应用、"
                                       "偏差最大目标在稀疏参数区域补充实验，并把逆向设计候选的悲观惩罚收紧。")
                    else:
                        st.info("本轮没有可判定记录：任务单中的实测值尚未填写（或 σ 不可用）。"
                                "填写后重新上传即可，不做任何编造判定。")
                    if _vt_rep.get("n_pending"):
                        st.warning(f"{_vt_rep['n_pending']} 条 REC 记录未匹配到候选存档"
                                   "（可能使用了被清理的旧任务单）。")
                    st.caption("判定基准：candidates 存档的校准后预测均值 ± σ_cal；"
                               "σ 不可用或实测缺失的记录如实标记待填写。"
                               "报告：outputs/validation/验证命中率报告.md；判定完成后核心价值总报告自动更新。")
                    if use_demo:
                        evidence_mod.demo_banner()
                except Exception as e:
                    st.error(friendly_error(e, None, schema))

        # ---------------- V1.7 · L3 闭环效率基准（核心价值验证第三层） ----------------
        with st.expander("L3 · 闭环效率基准对比（模型引导 vs 随机试错，核心价值主证据）",
                         expanded=False):
            st.caption("重放模拟竞赛：同一初始子集出发，模型引导策略（悲观惩罚 + 综合最优选点）与"
                       "随机基线每轮各补 5 个点，比较达到目标规格所需实验次数。"
                       "真实标签全部取自当前数据集（严禁以模型预测冒充真值）。")
            _d_bm = current_df()
            _b_in = get_bundle()
            if _d_bm is None or _b_in is None:
                st.info("需要已加载的数据与已训练的模型。")
            else:
                _spec_cols = st.columns(3)
                _bm_spec = {}
                for _i, (_t, _key, _dir) in enumerate([
                        ("porosity_pct", "max", "≤"), ("bond_strength_MPa", "min", "≥"),
                        ("coupled_damage_rate", "max", "≤")]):
                    _def = bench_mod.DEFAULT_SPEC[_t][_key]
                    _bm_spec[_t] = {_key: float(_spec_cols[_i].number_input(
                        f"{zh(_t)} {_dir}", min_value=0.0, max_value=1e4,
                        value=float(_def), step=0.1, key=f"bm_spec_{_t}"))}
                _bm_c1, _bm_c2, _bm_c3 = st.columns(3)
                _bm_init = int(_bm_c1.number_input("初始样本数", 10, 40, 15, key="bm_init"))
                _bm_batch = int(_bm_c2.number_input("每轮补点数", 3, 10, 5, key="bm_batch"))
                _bm_seeds = int(_bm_c3.selectbox("随机种子数", [3, 5, 10], index=1, key="bm_seeds"))
                st.caption(f"演示数据 280 行约需 {_bm_seeds} × 1–2 分钟；种子越多结论越稳。")
                if st.button("▶ 运行 L3 效率基准", type="primary", key="bm_run"):
                    _prog = st.status("L3 重放竞赛运行中…", expanded=True)
                    _bm_lines = []

                    def _bm_cb(name, frac=None):
                        _bm_lines.append(f"· {name}")
                        _prog.write("\n".join(_bm_lines[-10:]))

                    try:
                        with st.spinner(""):
                            _bm_res = bench_mod.run_benchmark(
                                _d_bm, schema, spec=_bm_spec, n_init=_bm_init,
                                n_batch=_bm_batch, n_seeds=_bm_seeds, progress_cb=_bm_cb,
                                bundle=_b_in)
                        if _bm_res.get("feasible"):
                            _sv = _bm_res["saved_experiments"]
                            _g_m, _r_m = _bm_res["guided_median_steps"], _bm_res["random_median_steps"]
                            if _sv is not None and _sv > 0:
                                st.success(f"达到目标规格：模型引导 {_g_m:.0f} 次 vs 随机 {_r_m:.0f} 次"
                                           f"（中位数，{_bm_seeds} 种子全部达成）——离线重放节省 {_sv:.0f} 次实验。")
                            else:
                                _sr = _bm_res.get("guided_success_ratio")
                                st.warning("未全部达成或未跑赢（V1.8 规则：不强行填节省次数；"
                                           "保留各种子状态与成功比例"
                                           + (f"：引导更快 {_bm_res.get('guided_success_count')}/"
                                              f"{_bm_res.get('n_both_completed')}（{_sr*100:.0f}%）"
                                              if _sr is not None else "") + "）。详见下方解释与排查建议。")
                            st.markdown(evidence_mod.badge_span(
                                f"引导中位 {_g_m} ｜ 随机中位 {_r_m}"
                                + (f" ｜ 节省 {_sv} 次" if _sv is not None else "")
                                + f" ｜ 达标点占比 {_bm_res['n_ok_ratio']*100:.0f}%",
                                source_class="model", demo=use_demo), unsafe_allow_html=True)
                            # 网页版对比曲线（plotly 统一模板）
                            import plotly.graph_objects as _go
                            _fig_bm = _go.Figure()
                            for _curves, _color, _label in (
                                    (_bm_res["random_curves"], pstyle.GRAY_MUTED, "随机基线"),
                                    (_bm_res["guided_curves"], pstyle.BLUE_SIGNAL, "模型引导")):
                                _grid, _mean, _std = bench_mod._curve_stats(_curves)
                                _fig_bm.add_trace(_go.Scatter(
                                    x=_grid, y=_mean, mode="lines", name=_label,
                                    line=dict(color=_color, width=2.5)))
                                _fig_bm.add_trace(_go.Scatter(
                                    x=np.concatenate([_grid, _grid[::-1]]),
                                    y=np.concatenate([np.clip(_mean - _std, 0, None), (_mean + _std)[::-1]]),
                                    fill="toself", fillcolor=_color, opacity=0.12,
                                    line=dict(width=0), name=f"{_label} ±1σ", showlegend=False))
                            pstyle.nature_layout(_fig_bm, title="离线策略评估（累计实验次数 vs 归一化超体积）",
                                                 height=380)
                            _fig_bm.update_xaxes(title_text="累计实验次数")
                            _fig_bm.update_yaxes(title_text="归一化超体积（MC 估计）")
                            st.plotly_chart(_fig_bm, width="stretch", key="bm_curve_fig")
                            st.info(_bm_res["explanation"])
                            st.caption("计算方式：benchmark.run_benchmark；论文级曲线已存 "
                                       "outputs/benchmark/闭环效率对比.png；报告：效率对比报告.md。"
                                       "真实标签全部来自数据集重放，非模型预测。")
                        else:
                            st.warning(_bm_res["explanation"])
                        if use_demo:
                            evidence_mod.demo_banner()
                    except Exception as e:
                        _prog.update(label="L3 运行失败", state="error")
                        st.error(friendly_error(e, _d_bm, schema))

# ---------------- ⑦ 逆向设计 ----------------
# ⚠️ tab 变量错位警示：tab6 = ⑦ 逆向设计
with tab6:
    st.subheader("多目标工艺逆向设计")
    render_tab_header("tab6", CTX)
    st.markdown("基于训练完成的三级代理模型，在设定工艺范围和工程约束条件下"
                "搜索 Pareto 非支配工艺方案。")
    if get_bundle() is None:
        st.warning("请先训练模型。")
    else:
        bundle = get_bundle()
        if use_demo:
            st.info(DEMO_RESULT_NOTICE)

        o1, o2, o3 = st.columns(3)
        o1.markdown('<div class="obj-card"><div class="name">孔隙率</div>'
                    '<div class="dir">↓ 最小化</div></div>', unsafe_allow_html=True)
        o2.markdown('<div class="obj-card"><div class="name">结合强度</div>'
                    '<div class="dir">↑ 最大化</div></div>', unsafe_allow_html=True)
        o3.markdown('<div class="obj-card"><div class="name">耦合损伤</div>'
                    '<div class="dir">↓ 最小化</div></div>', unsafe_allow_html=True)

        dep_min = obj_cfg.get("constraints", {}).get("deposition_efficiency_pct", {}).get("min", 45)
        feed_min = obj_cfg.get("constraints", {}).get("powder_feed_g_min", {}).get("min", 15)
        st.markdown(f"**工程约束**：沉积效率 ≥ {dep_min:g} %；送粉速率 ≥ {feed_min:g} g/min（当前设置下限）")

        # V1.3.2：固定结构参数默认值同样优先使用当前训练数据中位数，避免默认即外推
        _defaults_opt = compute_default_inputs(schema, current_df())
        fixed = {}
        st.markdown("**【固定喷枪结构参数】**")
        cols = st.columns(3)
        for i, (col, spec) in enumerate(schema["structure_inputs"].items()):
            if spec.get("categorical") or "min" not in spec or "max" not in spec:
                continue
            fixed[col] = cols[i % 3].number_input(
                zh(col),
                min_value=float(spec["min"]), max_value=float(spec["max"]),
                value=float(_defaults_opt[col]),
                key="opt_" + col)

        n = st.slider("全局候选工艺点数量", 500, 5000, 2500, 500)
        st.caption("候选点越多，搜索越充分，但计算时间会增加。")

        # V1.3.2：优化搜索域选择——默认在训练数据支持域内寻优，杜绝默认无提示外推测优
        search_domain_disp = st.radio(
            "优化搜索域",
            ["当前训练数据支持域（默认）", "设备允许可行域（探索性，存在外推风险）"],
            index=0, horizontal=False,
            help="训练数据支持域：全部候选点位于模型训练数据覆盖范围内，无外推；"
                 "设备允许可行域：候选点扩展到设备允许范围，仅用于探索性分析。")
        sd = "device" if search_domain_disp.startswith("设备") else "training"
        if sd == "device":
            st.warning("已选择设备允许可行域（探索性）：部分候选点将位于训练数据覆盖范围之外，"
                       "属于外推预测。所有外推候选方案会被明确标记“存在外推风险”，请谨慎解读。")
        else:
            st.caption("当前在训练数据支持域内寻优：所有候选方案均位于模型训练数据覆盖范围内，不产生外推。")

        # V1.4 spec 27/28：可选熔融状态约束（默认关闭；不假设熔融越高越好）
        melt_constraints = {}
        if bundle.get("stage15_enabled") and bundle.get("stage15_models"):
            with st.expander("熔融状态约束（可选，默认关闭）"):
                st.caption("平台不假设“熔融越高越好”——过热、飞溅、破碎同样可能降低涂层质量；"
                           "如需寻找适宜熔融状态窗口，建议同时设置下限与过热风险上限。")
                if st.checkbox("启用熔融状态约束", key="opt_melt_enable"):
                    _m15m = bundle["stage15_models"]
                    if "melting_index" in _m15m:
                        _mi = st.slider("熔融指数下限（melting_index ≥ x）", 0.0, 2.0, 0.6, 0.05, key="opt_mi")
                        melt_constraints["melting_index"] = {"min": float(_mi)}
                    if "melting_fraction_pct" in _m15m:
                        _mf = st.slider("熔融比例下限（melting_fraction_pct ≥ x，%）", 0.0, 100.0, 60.0, 5.0, key="opt_mf")
                        melt_constraints["melting_fraction_pct"] = {"min": float(_mf)}
                    if "overheating_risk" in _m15m:
                        _oh = st.slider("过热风险上限（overheating_risk ≤ x）", 0.0, 1.0, 0.5, 0.05, key="opt_oh")
                        melt_constraints["overheating_risk"] = {"max": float(_oh)}

        if st.button("🧭 开始多目标逆向寻优"):
            with st.spinner("正在搜索 Pareto 工艺窗口..."):
                _obj_call = {"objectives": obj_cfg.get("objectives", {}),
                             "constraints": {**obj_cfg.get("constraints", {}), **melt_constraints}}
                front, stats = random_pareto_search(
                    bundle, schema, _obj_call, fixed_values=fixed, n_candidates=n,
                    return_stats=True, search_domain=sd)
            if front.empty:
                st.error("没有找到满足当前约束的候选点，请放宽约束或扩大参数范围。")
            else:
                if sd == "device":
                    s1, s2, s3, s4 = st.columns(4)
                else:
                    s1, s2, s3 = st.columns(3)
                s1.metric("本次搜索候选点数量", stats["n_candidates"])
                s2.metric("满足工程约束点数量", stats["n_feasible"])
                s3.metric("Pareto 非支配方案数量", stats["n_pareto"])
                if sd == "device":
                    s4.metric("标记外推风险方案数", stats.get("n_extrap", 0))

                st.markdown("**Pareto 非支配工艺方案**")
                show_cols = list(schema["process_inputs"].keys()) + [
                    "pred_porosity_pct", "pred_bond_strength_MPa",
                    "pred_coupled_damage_rate", "pred_deposition_efficiency_pct"]
                show_cols = [c for c in show_cols if c in front.columns]
                disp = front[show_cols].copy()
                pred_rename = {
                    "pred_porosity_pct": "预测孔隙率（%）",
                    "pred_bond_strength_MPa": "预测结合强度（MPa）",
                    "pred_coupled_damage_rate": "预测耦合损伤",
                    "pred_deposition_efficiency_pct": "预测沉积效率（%）",
                }
                rename = {c: zh(c) for c in schema["process_inputs"].keys()}
                rename.update({k: v for k, v in pred_rename.items() if k in disp.columns})
                disp = disp.rename(columns=rename).round(2)
                disp.insert(0, "方案编号", range(1, len(disp) + 1))
                domain6 = bundle.get("training_domain") or {}
                if domain6:
                    disp.insert(1, "数据支持状态", [
                        support_label(pareto_row_support(
                            {c: front.iloc[i][c] for c in schema["process_inputs"]}, domain6))
                        for i in range(len(disp))])
                # V1.3.2：设备允许域模式下展示逐方案外推风险标记
                if "extrapolation_risk" in front.columns:
                    disp.insert(2 if domain6 else 1, "外推风险",
                                list(front["extrapolation_risk"]))
                st.dataframe(disp.head(30), width="stretch")

                st.markdown("**Pareto 工艺参数范围**")
                win = []
                for c in schema["process_inputs"].keys():
                    s = front[c]
                    win.append({"工艺参数": zh(c),
                                "最小值": round(float(s.min()), 2),
                                "最大值": round(float(s.max()), 2),
                                "中位数": round(float(s.median()), 2)})
                st.dataframe(pd.DataFrame(win), width="stretch", hide_index=True)
                if sd == "training":
                    st.caption("以上为当前模型与当前约束条件下的 Pareto 工艺参数范围。"
                               "本次搜索域为当前训练数据支持域，全部方案均无外推。")
                else:
                    st.caption("以上为当前模型与当前约束条件下的 Pareto 工艺参数范围。"
                               "本次搜索域为设备允许可行域（探索性），标记“存在外推风险”的方案为外推预测结果。")

                # V1.5 spec 四十二/四十三 + V1.6 四段式：Pareto 证据支持收进次级证据区
                d_ev = current_df()
                if d_ev is not None:
                    ev = insight_mod.pareto_evidence(front, d_ev, bundle.get("training_domain") or {},
                                                     bundle, list(schema["process_inputs"].keys()),
                                                     max_rows=10)
                    if len(ev):
                        with st.expander("次级证据：Pareto 方案证据支持（数据支持度 / 邻近实验 / 不确定性）",
                                         expanded=False):
                            st.dataframe(ev, width="stretch", hide_index=True)
                            st.caption("数据支持程度与邻近实验来自训练数据（直接观测）；预测不确定性为模型输出"
                                       "（模型推断）；证据不包含材料机理判断。")
                            with st.expander("为什么推荐这个点（逐方案推荐依据）", expanded=False):
                                for _i, evr in ev.iterrows():
                                    st.markdown(f"**{evr['方案']}**")
                                    st.caption(insight_mod.recommendation_basis(evr))
                                    st.markdown("---")

                PARETO_OUT.parent.mkdir(parents=True, exist_ok=True)
                front.to_excel(PARETO_OUT, index=False)
                st.caption(f"结果已自动导出：{PARETO_OUT.name}")

                # ---------------- V1.7 · L2 联动：导出验证任务单（推荐即命中入口） ----------------
                st.markdown("---")
                with st.container(border=True):
                    st.markdown("**导出验证任务单（L2 推荐-验证闭环第 1 步）**")
                    st.caption("从前沿取 Top-N 推荐候选生成实验任务单（输入参数已预填、状态/缺陷/性能列"
                               "留空待实验填写、experiment_id 按 REC- 编号）。实验完成后回「⑥ 数据洞察 → "
                               "推荐-验证闭环」上传判定命中率；任务单格式与平台数据模板同构，也可在"
                               "「② 数据管理」直接并入训练数据。")
                    _vt_n = st.number_input("导出候选数（Top N）", 3, 20, 8, key="vt_n")
                    if st.button("⬆ 导出验证任务单", type="primary", key="vt_export"):
                        try:
                            _vt_path, _vt_ver, _vt_cand = vtrack_mod.export_validation_sheet(
                                front, schema, n_top=int(_vt_n), bundle=bundle, demo=use_demo)
                            st.success(f"验证任务单已导出：{_vt_path.name}")
                            st.caption(f"版本号 {_vt_ver} ｜ 实际导出 {len(_vt_cand)} 个候选"
                                       + (f"（前沿共 {len(front)} 个方案，请求 Top {_vt_n}）"
                                          if len(_vt_cand) < _vt_n else "")
                                       + f" ｜ 候选预测存档：outputs/validation/candidates_{_vt_ver}.csv（判定基准）。")
                            with open(_vt_path, "rb") as _f:
                                st.download_button("⬇ 下载验证任务单（Excel）", _f,
                                                   file_name=_vt_path.name,
                                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                   key="vt_download")
                            _cal_f = calib_mod.load_factors()
                            st.caption(("σ 已按 L1 校准因子修正（k 见 ③ 模型训练 → L1）。"
                                        if _cal_f else "当前无 L1 校准文件，σ 因子=1；"
                                        "建议先运行校准使命中判定更可信。"))
                            if use_demo:
                                evidence_mod.demo_banner()
                        except Exception as e:
                            st.error(friendly_error(e, current_df(), schema))

# ---------------- ⑧ 结果输出 ----------------
# ⚠️ tab 变量错位警示：tab7 = ⑧ 结果输出（tab5=⑥洞察、tab6=⑦逆向），勿按序号猜变量
with tab7:
    render_tab_header("tab7", CTX)
    ensure_output_tree()
    bundle7 = get_bundle()
    if bundle7 is None:
        st.warning("请先在「模型训练」页完成训练。")
    else:
        if use_demo:
            evidence_mod.demo_banner()

        d7 = current_df()
        # V1.6（P1-2）：指标区与 status-pill 同源去重——n/数据版本/模型版本/CV 均来自同一 CTX
        m11, m12, m13 = st.columns(3)
        m11.metric("当前数据集", "模拟数据" if use_demo else "真实数据")
        m12.metric("数据版本", CTX.get("dataset_version") or "未知（旧模型）")
        m13.metric("模型版本", CTX.get("model_version") or "未知（旧模型）")
        m21, m22, m23 = st.columns(3)
        m21.metric("样本数", CTX.get("sample_count") or (len(d7) if d7 is not None else "-"))
        m22.metric("独立批次数", d7["batch_id"].nunique() if d7 is not None and "batch_id" in d7 else "-")
        m23.metric("交叉验证方法", context_cv_short(CTX) if context_cv_short(CTX) != "—" else "-")

        lang = st.radio("图表语言", ["中文", "English"], horizontal=True)
        fmt = st.radio("图片格式", ["PNG", "SVG", "PNG + SVG"], horizontal=True)
        plang = "zh" if lang == "中文" else "en"
        pformats = {"PNG": ("png",), "SVG": ("svg",), "PNG + SVG": ("png_svg",)}[fmt]

        # ---------------- V1.7 · 核心价值验证总报告（L1→L4 汇总） ----------------
        st.markdown("---")
        with st.container(border=True):
            st.markdown("**核心价值验证总报告（L1→L4 汇总）**")
            st.caption("汇总四个证据维度（V1.8 独立判定，非闯关）：L1 不确定度校准 ｜ "
                       "L2 推荐即命中（区间覆盖 + 工程达标）｜ L3 离线策略评估 ｜ "
                       "L4 物理合理性诊断；一页式结论回答「当前上下文下哪些维度有效、"
                       "还缺哪条证据、下一步做什么实验」。缺失维度明确写「暂无证据」，不编造；"
                       "历史结果（旧模型/数据）单独标注，不计入当前状态。")
            if st.button("▶ 生成 / 刷新核心价值验证总报告", type="primary", key="cvr_gen"):
                try:
                    with st.spinner("汇总 L1–L4 产物…"):
                        _cvr_path, _ = cvr_mod.generate_core_value_report(
                            trigger="⑧ 结果输出 · 手动生成",
                            current_model_version=(bundle7 or {}).get("model_version") if bundle7 else None,
                            current_dataset_hash=(CTX.get("dataset_hash") if isinstance(CTX, dict) else None))
                    st.success(f"总报告已生成：{_cvr_path}")
                    st.markdown(_cvr_path.read_text(encoding="utf-8"))
                    with open(_cvr_path, "rb") as _f:
                        st.download_button("⬇ 下载总报告（Markdown）", _f,
                                           file_name=_cvr_path.name, key="cvr_download")
                except Exception as e:
                    st.error(friendly_error(e, current_df(), schema))
            else:
                if cvr_mod.REPORT_PATH.exists():
                    st.caption(f"已有总报告：{cvr_mod.REPORT_PATH.name}（点击上方按钮刷新 / 预览）。")

        def _run_dir():
            ts = st.session_state.get("out_run_ts")
            if ts is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["out_run_ts"] = ts
            return ROOT / "outputs" / "reports" / f"Run_{ts}"

        def _exporter():
            return PaperExporter(current_df(), schema, bundle7, lang=plang,
                                 formats=pformats, demo=use_demo, prefix=OUTPUT_PREFIX,
                                 run_dir=_run_dir())

        def _show_saved(out):
            items = out[0]
            for p in items:
                sp = str(p)
                if sp.endswith(".png"):
                    st.image(sp)
                    st.caption(f"已保存：{p.name}")
                elif sp.endswith(".svg") or sp.endswith(".pdf"):
                    st.caption(f"已保存：{p.name}")
                elif sp.endswith(".csv"):
                    st.caption(f"数据文件：{p.name}")
                elif sp.endswith(".json"):
                    st.caption(f"元数据：{p.name}")
                elif sp.endswith(".xlsx"):
                    st.caption(f"表格文件：{p.name}")

        def _make_bundle():
            with st.spinner("正在生成完整科研结果包（A–I：性能 / Parity / 重要性 / SHAP / 熔融 / "
                            "Pareto / 不确定性 / 数据洞察 / 图表科研解读 / 元数据 / ZIP）..."):
                try:
                    ex = PaperExporter(current_df(), schema, bundle7, lang=plang,
                                       formats=pformats, demo=use_demo, prefix=OUTPUT_PREFIX,
                                       run_prefix="Paper_Output")
                    run_dir, zip_path, paths = ex.build_bundle(n_candidates=2500)
                    st.session_state["last_zip"] = str(zip_path)
                    st.success(f"论文图表包已生成：{run_dir.name}（图 {len(paths['figures'])} 张 / "
                               f"表 {len(paths['tables'])} 份 / 数据 {len(paths['data'])} 份 / "
                               f"元数据 {len(paths['metadata'])} 份）")
                    st.caption("结果包内含证据清单（evidence_manifest.json），报告中每条结论均可按 eid 溯源到数据与模型版本。")
                    st.caption(f"目录：{run_dir}")
                    st.caption(f"摘要：{paths.get('summary', '')}")
                    if Path(zip_path).exists():
                        with open(zip_path, "rb") as f:
                            st.download_button("下载完整论文结果包（ZIP）", f,
                                               file_name=Path(zip_path).name,
                                               mime="application/zip", key="dl_zip")
                except Exception as e:
                    import traceback
                    st.info(f"生成失败：{e}\n\n```\n{traceback.format_exc()}\n```")

        # V1.5：置顶醒目入口——一键生成完整科研结果包（复用既有 A–I 模块）
        st.markdown("---")
        with st.container(border=True):
            ctop1, ctop2 = st.columns([3, 2])
            ctop1.markdown("**📦 完整科研结果包（推荐入口）**")
            ctop1.caption("一键生成 A–I 全部模块：数据概况 / 模型性能 / 可解释性 / 全链预测 / "
                          "Pareto / 不确定性 / 验证模板 / 熔融状态 / 数据洞察，"
                          "附图表科研解读（*_Analysis.md）、Insight_Report、元数据与 ZIP 打包。")
            if ctop2.button("📦 生成完整科研结果包", key="btn_bundle_top"):
                _make_bundle()
        st.markdown("---")

        with st.expander("A｜数据集与变量概况", expanded=False):
            if st.button("生成 A 模块输出", key="btn_a"):
                try:
                    ex = _exporter()
                    p, _ = ex.table_a1()
                    st.caption(f"Table_A1 已保存：{Path(p).name}")
                    for fn in [ex.fig_a1_completeness, ex.fig_a2_distributions, ex.fig_a3_correlation]:
                        _show_saved(fn())
                    st.caption(f"输出目录：{ex.run_dir}")
                except Exception as e:
                    st.info(f"生成失败：{e}")

        with st.expander("B｜模型预测性能"):
            st.caption("严格 OOF / grouped-CV，非训练集拟合值")
            cv = bundle7.get("chain_cv") or {}
            if not (cv.get("metrics") or {}).get("stage1") or not cv.get("oof_predictions"):
                st.info("当前模型缺少逐行 OOF 预测记录，请重新训练一次模型后再生成本节。")
            else:
                b1, b2, b3 = st.columns(3)
                stage_b = b1.selectbox("模型层级", ["stage1", "stage2", "stage3"],
                                       format_func=lambda s: STAGE_FULL_LABELS.get(s, s), key="b_stage")
                targets_b = list((cv.get("metrics") or {}).get(stage_b, {}).keys())
                if not targets_b:
                    st.info("该层级暂无可用目标。")
                else:
                    tgt_b = b2.selectbox("目标指标", targets_b,
                                         format_func=lambda t: zh(t), key="b_tgt")
                    ptype_b = b3.selectbox("图类型", ["Actual vs Predicted", "Residual"], key="b_type")
                    if st.button("生成图表", key="btn_b"):
                        try:
                            ex = _exporter()
                            if ptype_b == "Actual vs Predicted":
                                _show_saved(ex.fig_parity(stage_b, tgt_b))
                            else:
                                _show_saved(ex.fig_residual(stage_b, tgt_b))
                        except Exception as e:
                            st.info(f"生成失败：{e}")
                if st.button("生成 B 模块全部输出（指标表 + 全部 Parity/Residual + R² 总览）", key="btn_b_all"):
                    try:
                        ex = _exporter()
                        p, df_b = ex.table_b1()
                        st.caption(f"Table_B1 已保存：{Path(p).name}")
                        st.dataframe(df_b, width="stretch", hide_index=True)
                        for s in ["stage1", "stage2", "stage3"]:
                            for t in list((cv.get("metrics") or {}).get(s, {}).keys()):
                                _show_saved(ex.fig_parity(s, t))
                                _show_saved(ex.fig_residual(s, t))
                        _show_saved(ex.fig_b_overview())
                    except Exception as e:
                        st.info(f"生成失败：{e}")

        with st.expander("C｜模型可解释性"):
            st.caption("特征重要性 / SHAP / PDP")
            c1, c2, c3, c4 = st.columns(4)
            stage_c = c1.selectbox("模型层级", ["stage1", "stage2"],
                                   format_func=lambda s: STAGE_FULL_LABELS.get(s, s), key="c_stage")
            targets_c = list(bundle7.get(f"{stage_c}_models", {}).keys())
            tgt_c = c2.selectbox("目标指标", targets_c or ["-"], format_func=lambda t: zh(t), key="c_tgt")
            ctype_c = c3.selectbox("分析类型", ["Feature Importance", "SHAP Summary",
                                                "SHAP Dependence", "PDP"], key="c_type")
            feats_c = bundle7.get(f"{stage_c}_features", [])
            feat_c = c4.selectbox("特征（Dependence/PDP 用）", feats_c, format_func=lambda f: zh(f), key="c_feat")
            topn_c = c2.selectbox("重要性 Top N", [10, 15, 20], key="c_topn") if ctype_c == "Feature Importance" else 15
            if st.button("生成图表", key="btn_c") and tgt_c in targets_c:
                try:
                    ex = _exporter()
                    if ctype_c == "Feature Importance":
                        _show_saved(ex.fig_c_importance(stage_c, tgt_c, topn=topn_c))
                    elif ctype_c == "SHAP Summary":
                        _show_saved(ex.fig_c_shap_summary(stage_c, tgt_c))
                    elif ctype_c == "SHAP Dependence":
                        _show_saved(ex.fig_c_shap_dependence(stage_c, tgt_c, feat_c))
                    else:
                        _show_saved(ex.fig_c_pdp(stage_c, tgt_c, feat_c))
                except PaperOutputError as e:
                    st.info(str(e))
                except Exception as e:
                    st.info(f"生成失败：{e}")

        with st.expander("D｜全链预测结果"):
            st.caption("使用中值输入条件生成全链预测示意图（含三级不确定性）。")
            if st.button("生成 D 模块输出", key="btn_d"):
                try:
                    ex = _exporter()
                    _show_saved(ex.fig_d_chain())
                except Exception as e:
                    st.info(f"生成失败：{e}")

        with st.expander("E｜Pareto 优化结果"):
            e1, e2, e3 = st.columns(3)
            x_e = e1.selectbox("X 轴目标", ["porosity_pct"], format_func=lambda t: zh(t), key="e_x")
            y_e = e2.selectbox("Y 轴目标", ["bond_strength_MPa"], format_func=lambda t: zh(t), key="e_y")
            c_e = e3.selectbox("颜色目标", ["coupled_damage_rate"], format_func=lambda t: zh(t), key="e_c")
            if st.button("生成 Pareto 前沿", key="btn_e"):
                try:
                    ex = _exporter()
                    out = ex.fig_e_pareto(n_candidates=2500)
                    _show_saved(out[0:2])
                    st.caption(f"Pareto 解数量：{out[2]['n_pareto']}（候选 {out[2]['n_candidates']}，"
                               f"满足约束 {out[2]['n_feasible']}）")
                except Exception as e:
                    st.info(f"生成失败：{e}")
            if st.button("生成 Pareto 方案表 + 参数范围", key="btn_e2"):
                try:
                    ex = _exporter()
                    p, df_e, _ = ex.table_e1(2500)
                    st.caption(f"Table_E1 已保存：{Path(p).name}")
                    st.dataframe(df_e.head(20), width="stretch", hide_index=True)
                    p2, _ = ex.table_e2_ranges(2500)
                    st.caption(f"Table_E2 已保存：{Path(p2).name}")
                    _show_saved(ex.fig_e2_ranges(2500))
                except Exception as e:
                    st.info(f"生成失败：{e}")

        with st.expander("F｜预测不确定性"):
            st.caption("GP 标准差 / 预测区间")
            tgt_f = st.selectbox("目标指标", list(bundle7.get("stage3_models", {}).keys()) or ["-"],
                                 format_func=lambda t: zh(t), key="f_tgt")
            if st.button("生成不确定性图", key="btn_f"):
                try:
                    ex = _exporter()
                    _show_saved(ex.fig_f_uncertainty(tgt_f))
                except PaperOutputError as e:
                    st.info(str(e))
                except Exception as e:
                    st.info(f"生成失败：{e}")

        with st.expander("G｜实验验证"):
            st.caption("接口预留")
            st.caption("当前阶段仅提供数据模板接口；平台不生成虚假实验验证数据。"
                       "Demo 模式下禁用正式实验验证结论。")
            if st.button("下载验证数据模板（Table_G1）", key="btn_g"):
                try:
                    ex = _exporter()
                    p = ex.table_g1_template()
                    st.caption(f"模板已保存：{Path(p).name}")
                except Exception as e:
                    st.info(f"生成失败：{e}")

        # V1.4 H｜颗粒熔融与沉积状态（可选模块；不修改已有 A–G）
        with st.expander("H｜颗粒熔融与沉积状态（可选，Stage 1.5）"):
            if not (bundle7.get("stage15_enabled") and bundle7.get("stage15_models")):
                st.info("当前数据缺少颗粒熔融状态标签，Stage 1.5 未启用，无 H 模块输出。")
            else:
                st.caption("输出：Particle T–V Map / 熔融状态分布 / 熔融模型评价 / "
                           "熔融-缺陷关联 / 熔融-性能关联 / 熔融特征重要性；"
                           "文件前缀按模式自动命名（DEMO_/DUAL_/APS_/CASCADE_/LIT_，Fig_Hxx_）。")
                if st.button("生成 H 模块全部输出", key="btn_h_all"):
                    try:
                        ex = _exporter()
                        for fn in [ex.fig_h_tv_map, ex.fig_h_melting_distribution,
                                   ex.fig_h_melting_performance, ex.fig_h_melting_vs_defects,
                                   ex.fig_h_melting_vs_performance, ex.fig_h_melting_importance]:
                            try:
                                _show_saved(fn())
                            except PaperOutputError as e:
                                st.info(str(e))
                    except Exception as e:
                        st.info(f"生成失败：{e}")

        # V1.5 I｜数据洞察与实验反馈
        with st.expander("I｜数据洞察与实验反馈", expanded=False):
            st.caption("数据覆盖 / 相关趋势 / 模型规律 / 值得复核的数据 / 建议补充实验区域。"
                       "三源分级：直接观测 / 模型推断 / 优化建议。")
            if st.button("生成 I 模块（Insight_Report + Evidence）", key="btn_i"):
                try:
                    from src.insights import build_insight_report as _bir
                    _run_dir_i = _run_dir()
                    md_i, xlsx_i = _bir(CTX, current_df(), schema, bundle7,
                                        _run_dir_i / "insights")
                    st.success("Insight_Report 已生成")
                    st.caption(f"报告：{md_i}")
                    st.caption(f"证据表：{xlsx_i}")
                except Exception as e:
                    st.info(f"生成失败：{friendly_error(e)}")

        st.markdown("---")
        if st.button("💾 保存当前研究快照", key="btn_snapshot"):
            try:
                snap = save_snapshot(current_df(), schema, bundle7)
                st.success(f"研究快照已保存：{snap}")
                st.caption("包含 dataset_info / model_info / metrics / Pareto 结果 / config 副本 / 可复现 manifest。")
            except Exception as e:
                st.info(f"快照保存失败：{friendly_error(e)}")

        st.markdown("---")
        if st.button("📦 生成完整科研结果包", key="btn_bundle"):
            _make_bundle()
