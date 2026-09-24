# -*- coding: utf-8 -*-
"""V1.4 多喷涂工艺架构：研究模式注册中心

四种研究模式：
  dual_anode        双阳极等离子喷涂（保留现有正式模型）
  conventional_aps  传统大气等离子喷涂（APS）
  cascade           多电极 / 级联等离子喷涂
  literature        通用文献数据测试（任意 Excel/CSV，用户自选 X/Y）

职责：模式注册、数据/模型隔离路径、输出前缀、Schema 加载（合并材料层与
熔融可选层）、CV 方式建议。全部 pathlib，Windows / macOS 通用。

铁律：本模块不修改任何模型逻辑；dual_anode 的 Schema 即原 data_schema.yaml
（合并可选层后 material_inputs / melting_states 为空节，训练路径与 V1.3 完全一致）。
"""
from pathlib import Path

import pandas as pd
import yaml

from .config import ROOT

DATA_ROOT = ROOT / "data"
MODELS_ROOT = ROOT / "models"

# 熔融层的元数据字段（记录来源，不作为预测目标）
MELTING_META_FIELDS = ["melting_data_source"]

MELTING_CLASS_ZH = {
    "insufficiently_molten": "熔融不足",
    "partially_molten": "部分熔融",
    "fully_molten": "充分熔融",
    "overheated": "过热状态",
    "stable_splat": "稳定铺展",
    "splash_fragmentation": "飞溅破碎",
    "incomplete_flattening": "铺展不充分",
    "other": "其他",
}

MODES = {
    "dual_anode": {
        "label": "双阳极等离子喷涂",
        "sources": ["模拟演示数据", "文献数据", "真实实验数据"],
        "source_kind": {"模拟演示数据": "demo", "文献数据": "literature", "真实实验数据": "real"},
        "schema_file": "data_schema.yaml",
        "prefix": {"demo": "DEMO_", "real": "DUAL_", "literature": "LIT_"},
        "has_demo": True,
        "research_line": [
            ("① 结构 / 工艺参数", "喷枪结构<br>双阳极电流<br>气体流量<br>送粉速率<br>喷涂距离"),
            ("② 射流与粒子状态", "电弧电压<br>射流温度<br>射流速度<br>粒子温度<br>粒子速度"),
            ("③ 颗粒熔融与沉积", "熔融状态<br>熔融指数<br>铺展状态<br>过热风险<br>（有数据时启用）"),
            ("④ 缺陷网络", "孔隙<br>层间未结合<br>未熔颗粒<br>裂纹<br>缺陷连通度"),
            ("⑤ 涂层性能", "结合强度<br>硬度<br>磨损<br>腐蚀<br>空蚀 · 耦合损伤"),
            ("⑥ 逆向设计", "多目标优化<br>Pareto 前沿<br>工艺窗口"),
        ],
        "flow": ["1. 选择数据", "2. 检查数据", "3. 训练过程模型",
                 "4. 训练熔融状态模型（数据支持时）", "5. 工艺预测", "6. 模型解析",
                 "7. 逆向设计", "8. 结果输出"],
        "train_desc": "三级代理模型训练（结构/工艺/材料 → 过程状态 → 缺陷 → 性能，熔融层可选）",
    },
    "conventional_aps": {
        "label": "传统大气等离子喷涂（APS）",
        "sources": ["文献数据", "真实实验数据"],
        "source_kind": {"文献数据": "literature", "真实实验数据": "real"},
        "schema_file": "schema_aps.yaml",
        "prefix": {"real": "APS_", "literature": "LIT_"},
        "has_demo": False,
        "research_line": [
            ("① APS 工艺 + 粉末属性", "弧电流/弧电压<br>气体流量<br>送粉<br>喷距<br>粉末粒径/熔点"),
            ("② 粒子状态", "射流温度/速度<br>粒子温度/速度<br>粒子粒径"),
            ("③ 熔融 / 铺展", "熔融状态<br>铺展因子<br>飞溅风险<br>（有数据时启用）"),
            ("④ 缺陷 / 微观结构", "孔隙<br>未熔颗粒<br>裂纹<br>氧化物"),
            ("⑤ 涂层性能", "结合强度<br>硬度<br>磨损<br>腐蚀"),
            ("⑥ 数据驱动优化", "Pareto 前沿<br>工艺窗口"),
        ],
        "flow": ["1. 上传 APS 数据", "2. 检查数据", "3. 训练过程模型",
                 "4. 训练熔融/铺展模型（数据支持时）", "5. 工艺预测", "6. 模型解析",
                 "7. 逆向设计", "8. 结果输出"],
        "train_desc": "APS 三级代理模型训练（弧电流/弧电压等 APS 参数，独立于双阳极体系）",
    },
    "cascade": {
        "label": "多电极 / 级联等离子喷涂",
        "sources": ["文献数据", "真实实验数据"],
        "source_kind": {"文献数据": "literature", "真实实验数据": "real"},
        "schema_file": "schema_cascade.yaml",
        "prefix": {"real": "CASCADE_", "literature": "LIT_"},
        "has_demo": False,
        "research_line": [
            ("① 喷枪结构 / 电极构型 + 工艺", "电极数<br>级联级数<br>级联环<br>弧电流/电压<br>气体流量"),
            ("② 电弧 / 射流稳定性", "电压波动<br>射流温度<br>射流速度"),
            ("③ 粒子状态 → 熔融 / 沉积", "粒子温度/速度<br>粒子通量<br>熔融状态<br>（有数据时启用）"),
            ("④ 缺陷 / 微观结构", "孔隙<br>未熔颗粒<br>裂纹"),
            ("⑤ 涂层性能", "结合强度<br>硬度<br>磨损<br>腐蚀"),
            ("⑥ 工艺优化", "Pareto 前沿<br>工艺窗口"),
        ],
        "flow": ["1. 上传级联喷枪数据", "2. 检查数据", "3. 训练过程模型",
                 "4. 训练熔融/沉积模型（数据支持时）", "5. 工艺预测", "6. 模型解析",
                 "7. 逆向设计", "8. 结果输出"],
        "train_desc": "级联/多电极三级代理模型训练（突出喷枪结构与电极构型输入）",
    },
    "literature": {
        "label": "通用文献数据测试",
        "sources": ["文献数据（固定）"],
        "source_kind": {"文献数据（固定）": "literature"},
        "schema_file": "schema_literature.yaml",
        "prefix": {"literature": "LIT_"},
        "has_demo": False,
        "research_line": [
            ("① 自选输入 X", "用户从文献表选择<br>任意输入特征<br>+ 材料/粉末属性"),
            ("② 过程状态（可选）", "用户指定过程变量<br>没有可跳过"),
            ("③ 熔融状态（可选）", "映射熔融字段<br>没有可跳过"),
            ("④ 缺陷（可选）", "用户指定缺陷变量"),
            ("⑤ 性能 Y", "用户指定预测目标"),
            ("⑥ 快速验证", "LOOCV / KFold<br>小样本适用"),
        ],
        "flow": ["1. 上传文献数据（选工作表）", "2. 选择 X / 熔融 / 缺陷 / Y 变量",
                 "3. 训练（自动建议 CV 方式）", "4. 预测与解析",
                 "5. 导出 LIT_ 结果"],
        "train_desc": "通用文献数据测试（变量由用户选择，链条自动按可用数据裁剪）",
    },
}

DEFAULT_MODE = "dual_anode"


def mode_ids():
    return list(MODES.keys())


def load_schema_for_mode(mode_id):
    """加载模式 Schema 并合并可选层（材料/粉末属性 + 熔融状态）。

    合并策略：可选层的节若与基础 Schema 冲突，以基础 Schema 为准；
    合并后的 material_inputs / melting_states 全部 required: false。
    """
    cfg = MODES[mode_id]
    with open(ROOT / "config" / cfg["schema_file"], "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f) or {}
    with open(ROOT / "config" / "schema_layers.yaml", "r", encoding="utf-8") as f:
        layers = yaml.safe_load(f) or {}
    for sec in ["material_inputs", "melting_states"]:
        merged = dict(layers.get(sec) or {})
        merged.update(schema.get(sec) or {})
        schema[sec] = merged
    for sec in ["meta", "structure_inputs", "process_inputs", "process_states",
                "defect_network", "performance_outputs"]:
        schema.setdefault(sec, {})
    return schema


def data_dir(mode_id, kind):
    """按模式隔离的数据目录。dual 的 Demo 数据保持原位 data/demo/。"""
    if kind == "demo":
        return DATA_ROOT / "demo"
    return DATA_ROOT / mode_id / kind


def data_file(mode_id, kind):
    """当前模式+来源的当前数据文件路径（可能不存在）。"""
    d = data_dir(mode_id, kind)
    if kind == "demo":
        return d / "DEMO_双阳极喷涂数据.xlsx"
    if kind == "real":
        return d / "current_data.xlsx"
    return d / "current_literature.xlsx"


def model_path(mode_id):
    """按模式隔离的部署模型路径。dual_anode 兼容回退旧版根目录模型。"""
    p = MODELS_ROOT / mode_id / "latest_chain.joblib"
    if mode_id == "dual_anode" and not p.exists() and (MODELS_ROOT / "latest_chain.joblib").exists():
        return MODELS_ROOT / "latest_chain.joblib"
    return p


def output_prefix(mode_id, kind):
    return MODES[mode_id]["prefix"].get(kind, "LIT_")


def suggest_cv(n_rows, has_groups):
    """文献小样本 CV 建议（spec 三十八）。

    存在真实独立 batch → GroupKFold（严禁人为制造 batch_id）；
    n <= 25 → LOOCV；26–60 → 5 折 KFold；n > 60 → 5 折 KFold。
    """
    if has_groups:
        return "GroupKFold（按批次分组）", "grouped"
    if n_rows <= 25:
        return "LOOCV（留一交叉验证）", "loo"
    return "5 折 KFold（KFold, shuffle）", "kfold"


def numeric_input_cols(df, schema_section):
    """从 Schema 节中筛选：存在于 df、可数值化、非分类的输入列。"""
    out = []
    for col, spec in (schema_section or {}).items():
        if col not in df.columns or spec.get("categorical"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            out.append(col)
    return out


def melting_target_cols(df, schema):
    """解析熔融层可用目标：{"cont": [...], "cat": [...], "meta": [...]}。

    - 连续目标：数值列且有效值 >= 20 才纳入（与其他层级同一阈值）；
    - 分类目标：非数值列且有效值 >= 20；
    - melting_data_source 等元数据字段不作为目标。
    启用规则（spec 十）：任一目标可用 → Stage 1.5 启用；否则整层自动跳过。
    """
    cont, cat, meta_cols = [], [], []
    for col, spec in (schema.get("melting_states") or {}).items():
        if col not in df.columns or spec.get("role") == "metadata" or col in MELTING_META_FIELDS:
            meta_cols.append(col) if col in df.columns else None
            continue
        s = df[col]
        if s.notna().sum() < 20:
            continue
        if spec.get("categorical") or not pd.api.types.is_numeric_dtype(s):
            cat.append(col)
        else:
            cont.append(col)
    return {"cont": cont, "cat": cat, "meta": meta_cols}
