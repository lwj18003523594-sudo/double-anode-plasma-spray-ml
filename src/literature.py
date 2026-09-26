# -*- coding: utf-8 -*-
"""V1.4 通用文献数据测试模式：工作表选择、运行时 Schema 构建、熔融字段映射建议。

原则（spec 二十九）：平台只提出"建议映射"，最终由用户确认；
不得完全自动猜测，不得人为制造 batch_id。
"""
import pandas as pd
import yaml

from .config import ROOT
from .paper_labels import unit_from_header

# 文献数据中常见的熔融字段别名（仅用于"建议映射"，最终由用户确认）
MELTING_ALIAS_SUGGESTIONS = {
    "melting_state_class": ["melting_state", "melting_state_class", "particle_melting_state",
                            "splat_type", "molten_state"],
    "melting_fraction_pct": ["melting_fraction", "melt_fraction", "molten_fraction",
                             "melting_degree_pct", "particle_melting_fraction", "melt_ratio_pct"],
    "melting_index": ["melting_index", "melting_degree", "particle_melting", "melt_index",
                      "melting_degree_index"],
    "spread_factor": ["spread_factor", "flattening_ratio", "splat_spread_factor"],
    "splat_state_class": ["splat_type", "splat_state", "splat_morphology"],
    "overheating_risk": ["overheating_risk", "overheat_risk"],
    "thermal_margin_C": ["thermal_margin_C", "superheat_C", "temperature_margin_C"],
}

SKIP_SHEET_KEYWORDS = ["readme", "使用说明", "说明", "usage", "guide", "read_me", "字典", "字段"]


def list_sheets(file_or_path):
    """列出 Excel 全部工作表名（CSV 视为单表）。"""
    name = str(getattr(file_or_path, "name", file_or_path)).lower()
    if name.endswith(".csv"):
        return ["CSV"]
    xl = pd.ExcelFile(file_or_path)
    return list(xl.sheet_names)


def pick_default_sheet(sheet_names):
    """多工作表默认选择：跳过 README/说明类工作表，取第一个数据表；无则第一个。"""
    for sn in sheet_names:
        low = str(sn).lower()
        if not any(k in low for k in SKIP_SHEET_KEYWORDS):
            return sn
    return sheet_names[0] if sheet_names else None


def load_sheet(file_or_path, sheet_name):
    """读取指定工作表（sheet_name=None 时用默认选择逻辑）。"""
    name = str(getattr(file_or_path, "name", file_or_path)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_or_path)
    if sheet_name is None:
        sheet_name = pick_default_sheet(list_sheets(file_or_path))
    return pd.read_excel(file_or_path, sheet_name=sheet_name)


def suggest_melting_mapping(columns):
    """对文献数据列名提出熔融字段"建议映射"（spec 二十九）。

    返回 {melting_field: matched_column}；仅建议，必须经用户确认后生效。
    """
    suggestions = {}
    cols = [str(c) for c in columns]
    low_cols = {c.lower().strip(): c for c in cols}
    for field, aliases in MELTING_ALIAS_SUGGESTIONS.items():
        for a in aliases:
            if a in low_cols:
                suggestions[field] = low_cols[a]
                break
    return suggestions


def build_runtime_schema(df, selection):
    """根据用户变量选择构建运行时 Schema。

    selection: {"x": [...], "states": [...], "melting": [...],
                "defects": [...], "performance": [...],
                "objectives": {col: "minimize"/"maximize"}}
    数值列自动获得数据驱动 min/max（用于界面输入控件与优化范围）；
    分类列进入 schema 但训练时自动剔除；不人为制造 batch_id。
    """
    def _spec_for(col):
        s = df[col]
        if pd.api.types.is_numeric_dtype(pd.to_numeric(s, errors="coerce")):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if v.empty:
                return {"unit": "-", "required": False}
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                hi = lo + 1.0
            pad = (hi - lo) * 0.05
            return {"unit": "-", "min": lo - pad, "max": hi + pad, "optimizable": True, "required": False}
        return {"unit": "-", "categorical": True, "required": False}

    schema = {
        "meta": {"experiment_id": {"unit": "-", "required": False},
                 "batch_id": {"unit": "-", "required": False}},
        "structure_inputs": {},
        "process_inputs": {},
        "process_states": {},
        "defect_network": {},
        "performance_outputs": {},
    }
    for col in selection.get("x", []):
        if col in df.columns:
            schema["process_inputs"][col] = _spec_for(col)
    for col in selection.get("states", []):
        if col in df.columns:
            # V1.8.1：单位尽量从列名括号中提取（如 粒子温度(°C) → °C），供
            # Parity/Residual 图 y 轴与统计框使用；提取失败保持 "-"。
            schema["process_states"][col] = {"unit": unit_from_header(col) or "-",
                                             "required": False}
    for col in selection.get("defects", []):
        if col in df.columns:
            schema["defect_network"][col] = {"unit": unit_from_header(col) or "-",
                                             "required": False}
    for col in selection.get("performance", []):
        if col in df.columns:
            schema["performance_outputs"][col] = {"unit": unit_from_header(col) or "-",
                                                  "required": False}
    # 熔融层：用户确认映射后写入 melting_states
    schema["melting_states"] = {}
    for field, col in (selection.get("melting_map") or {}).items():
        if col and col in df.columns:
            schema["melting_states"][field] = {"unit": "-", "required": False}
    schema["material_inputs"] = {}
    for col in selection.get("material", []):
        if col in df.columns:
            schema["material_inputs"][col] = _spec_for(col)
    # 优化目标
    schema["objectives"] = {col: d for col, d in (selection.get("objectives") or {}).items()
                            if col in df.columns}
    return schema


def default_objectives_for(df, schema):
    """文献模式默认优化目标建议（若用户选了孔隙/结合强度等常见列则自动预选，可修改）。"""
    prefs = [("porosity_pct", "minimize"), ("bond_strength_MPa", "maximize"),
             ("coupled_damage_rate", "minimize"), ("hardness_HV", "maximize")]
    out = {}
    perf = list((schema.get("performance_outputs") or {}).keys())
    for col, d in prefs:
        if col in perf:
            out[col] = d
    if not out and perf:
        out[perf[0]] = "maximize"
    return out
