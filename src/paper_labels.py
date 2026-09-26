# -*- coding: utf-8 -*-
"""论文图表统一标签层（V1.3）

同时维护：中文名称 / 英文论文名称 / 单位 / 缩写。
绘图代码不得散落硬编码，一律通过 paper_label() 获取。
只改变图中显示文字，不修改内部字段与数据。
"""

PAPER_LABELS = {
    # 基础信息
    "experiment_id":   {"zh": "实验编号", "en": "Experiment ID", "unit": "-", "abbr": "Exp ID"},
    "batch_id":        {"zh": "喷涂批次", "en": "Spray batch", "unit": "-", "abbr": "Batch"},
    "sample_id":       {"zh": "试样编号", "en": "Sample ID", "unit": "-", "abbr": "Sample"},
    "replicate_id":    {"zh": "重复测量编号", "en": "Replicate ID", "unit": "-", "abbr": "Rep"},

    # 双阳极结构参数
    "nozzle_throat_mm":            {"zh": "喷嘴喉部直径", "en": "Nozzle throat diameter", "unit": "mm", "abbr": "Throat D"},
    "nozzle_exit_mm":              {"zh": "喷嘴出口直径", "en": "Nozzle exit diameter", "unit": "mm", "abbr": "Exit D"},
    "secondary_laval_length_mm":   {"zh": "二级拉瓦尔段长度", "en": "Secondary Laval length", "unit": "mm", "abbr": "Laval L"},
    "anode_spacing_mm":            {"zh": "双阳极间距", "en": "Anode spacing", "unit": "mm", "abbr": "Spacing"},
    "powder_injection_angle_deg":  {"zh": "送粉注入角", "en": "Powder injection angle", "unit": "°", "abbr": "Inject θ"},

    # 工艺参数
    "anode1_current_A":   {"zh": "阳极1电流", "en": "Anode 1 current", "unit": "A", "abbr": "I1"},
    "anode2_current_A":   {"zh": "阳极2电流", "en": "Anode 2 current", "unit": "A", "abbr": "I2"},
    "Ar_flow_slpm":       {"zh": "Ar 气体流量", "en": "Ar flow rate", "unit": "L/min", "abbr": "Q.Ar"},
    "H2_flow_slpm":       {"zh": "H₂ 气体流量", "en": "H2 flow rate", "unit": "L/min", "abbr": "Q.H2"},
    "powder_feed_g_min":  {"zh": "送粉速率", "en": "Powder feed rate", "unit": "g/min", "abbr": "Feed"},
    "spray_distance_mm":  {"zh": "喷涂距离", "en": "Spray distance", "unit": "mm", "abbr": "SoD"},
    "traverse_speed_mm_s": {"zh": "喷枪移动速度", "en": "Traverse speed", "unit": "mm/s", "abbr": "Vt"},

    # 射流与粒子状态
    "arc_voltage_V":         {"zh": "电弧电压", "en": "Arc voltage", "unit": "V", "abbr": "U"},
    "jet_temperature_K":     {"zh": "射流温度", "en": "Jet temperature", "unit": "K", "abbr": "T.jet"},
    "jet_velocity_m_s":      {"zh": "射流速度", "en": "Jet velocity", "unit": "m/s", "abbr": "v.jet"},
    "particle_temperature_C": {"zh": "粒子温度", "en": "Particle temperature", "unit": "℃", "abbr": "T.p"},
    "particle_velocity_m_s": {"zh": "粒子速度", "en": "Particle velocity", "unit": "m/s", "abbr": "v.p"},

    # 涂层缺陷网络
    "porosity_pct":             {"zh": "孔隙率", "en": "Porosity", "unit": "%", "abbr": "P"},
    "lamellar_gap_pct":         {"zh": "层间未结合率", "en": "Inter-lamellar gap ratio", "unit": "%", "abbr": "ILG"},
    "unmelted_particle_pct":    {"zh": "未熔颗粒比例", "en": "Unmelted particle fraction", "unit": "%", "abbr": "UMP"},
    "crack_density_mm_mm2":     {"zh": "裂纹密度", "en": "Crack density", "unit": "mm/mm²", "abbr": "Cd"},
    "defect_connectivity_index": {"zh": "缺陷连通度指数", "en": "Defect connectivity index", "unit": "-", "abbr": "DCI"},

    # 涂层性能
    "bond_strength_MPa":         {"zh": "结合强度", "en": "Bond strength", "unit": "MPa", "abbr": "BS"},
    "hardness_HV":               {"zh": "显微硬度", "en": "Microhardness", "unit": "HV", "abbr": "HV"},
    "wear_rate_mg_m":            {"zh": "磨损率", "en": "Wear rate", "unit": "mg/m", "abbr": "Wr"},
    "corrosion_current_uA_cm2":  {"zh": "腐蚀电流密度", "en": "Corrosion current density", "unit": "μA/cm²", "abbr": "icorr"},
    "cavitation_mass_loss_mg":   {"zh": "空蚀失重", "en": "Cavitation mass loss", "unit": "mg", "abbr": "CML"},
    "coupled_damage_rate":       {"zh": "耦合损伤指标", "en": "Coupled damage index", "unit": "a.u.", "abbr": "CDI"},
    "deposition_efficiency_pct": {"zh": "沉积效率", "en": "Deposition efficiency", "unit": "%", "abbr": "DE"},

    # 物理派生变量
    "total_current_A":               {"zh": "总电流", "en": "Total current", "unit": "A", "abbr": "Itot"},
    "current_difference_A":          {"zh": "双阳极电流差", "en": "Current difference", "unit": "A", "abbr": "ΔI"},
    "current_imbalance_ratio":       {"zh": "双阳极电流不平衡系数", "en": "Current imbalance ratio", "unit": "-", "abbr": "CIR"},
    "total_gas_flow_slpm":           {"zh": "总气体流量", "en": "Total gas flow", "unit": "L/min", "abbr": "Q.tot"},
    "H2_Ar_ratio":                   {"zh": "H₂/Ar 气体流量比", "en": "H2/Ar flow ratio", "unit": "-", "abbr": "H2/Ar"},
    "electrical_power_W":            {"zh": "电功率", "en": "Electrical power", "unit": "W", "abbr": "Pel"},
    "specific_power_W_per_g_min":    {"zh": "单位送粉功率", "en": "Specific power", "unit": "W/(g/min)", "abbr": "Psp"},
    "particle_flight_time_ms_proxy": {"zh": "粒子飞行时间代理量", "en": "Particle flight time proxy", "unit": "ms", "abbr": "t.flight"},
}

# 源记录标识列：只用于定位原始记录，绝不能进入模型输入。
# 命中这些列名（小写比较）的列将作为 source_experiment_no 贯穿 OOF/Parity/证据表。
IDENTIFIER_COLUMNS = {"喷涂序号", "spray_run_id", "experiment_id", "batch_id",
                      "sample_id", "试样编号", "实验编号"}


def find_source_id_column(df):
    """返回 (列名, 质量)；质量: unique(唯一且无缺失) / duplicated(有重复) / missing(缺失)。
    df 无标识列时返回 (None, None)。不改写原值。"""
    import pandas as pd
    for col in df.columns:
        if str(col).strip().lower() in IDENTIFIER_COLUMNS:
            s = df[col]
            if s.isna().any():
                return col, "missing"
            if s.duplicated().any():
                return col, "duplicated"
            return col, "unique"
    return None, None


STAGE_LABELS_PAPER = {
    "stage1": {"zh": "一级模型", "en": "Stage 1 model"},
    "stage2": {"zh": "二级模型", "en": "Stage 2 model"},
    "stage3": {"zh": "三级模型", "en": "Stage 3 model"},
}

AXIS_LABELS = {
    "actual":     {"zh": "实验值", "en": "Experimental value"},
    "predicted":  {"zh": "预测值", "en": "Predicted value"},
    "residual":   {"zh": "残差", "en": "Residual"},
    "importance": {"zh": "特征重要性", "en": "Feature importance"},
    "feature":    {"zh": "影响因素", "en": "Feature"},
    "sample":     {"zh": "样本编号", "en": "Sample index"},
    "completeness": {"zh": "数据完整率", "en": "Completeness"},
    "SHAP值":     {"zh": "SHAP 值（对预测的贡献）", "en": "SHAP value (impact on prediction)"},
    "PDP响应":    {"zh": "模型平均预测响应", "en": "Mean model response"},
    "相关性":     {"zh": "皮尔逊相关系数（仅表征相关性，不代表因果关系）", "en": "Pearson correlation (not causation)"},
    "R2overview": {"zh": "决定系数 R²", "en": "R²"},
    "预测不确定性": {"zh": "预测值 ± 标准差（部署模型）", "en": "Prediction ± std (deployment model)"},
}

MISC_LABELS = {
    "title_parity":  {"zh": "实测值与预测值对比", "en": "Actual vs. Predicted"},
    "title_residual": {"zh": "残差图", "en": "Residual plot"},
    "porosity": {"zh": "预测孔隙率", "en": "Predicted porosity"},
    "bond": {"zh": "预测结合强度", "en": "Predicted bond strength"},
    "coupled_color": {"zh": "预测耦合损伤", "en": "Predicted coupled damage"},
    "pareto_front": {"zh": "Pareto 非支配解", "en": "Pareto non-dominated solutions"},
    "candidates": {"zh": "候选工艺点", "en": "Candidate solutions"},
    "demo_mark": {"zh": "DEMO 数据", "en": "DEMO DATA"},
    # ---- V1.6：统计诚信注脚 / 版式相关英文标签（P0-5 验收 2） ----
    "identity_line": {"zh": "y = x 参考线", "en": "y = x reference line"},
    "zero_line": {"zh": "零参考线", "en": "Zero reference line"},
    "stat_note_prefix": {"zh": "统计标注", "en": "Statistical annotation"},
    # 注意：CV 显示文案一律由 cv_display_method() 按实际执行方法动态生成，
    # 本静态标签只保留通用兜底文案，禁止在此硬编码"全链分组"等具体方法。
    "cv_note": {"zh": "评价方式：折外预测（非训练集拟合值）",
                "en": "Evaluation: out-of-fold predictions (not training-fit values)"},
    "errbar_note": {"zh": "误差棒为预测标准差（部署模型）",
                    "en": "Error bars: prediction std (deployment model)"},
    "single_col": {"zh": "单栏图（89 mm）", "en": "Single column (89 mm)"},
    "double_col": {"zh": "双栏图（183 mm）", "en": "Double column (183 mm)"},
    "evidence_note": {"zh": "附：证据溯源", "en": "Appendix: Evidence traceability"},
    "sample_count": {"zh": "样本数", "en": "Sample count"},
}


def paper_label(name, lang="zh", with_unit=True):
    """内部字段名 -> 论文图标签。lang: 'zh' | 'en'。"""
    info = PAPER_LABELS.get(name)
    if info is None:
        base = name
        unit = ""
    else:
        base = info["zh"] if lang == "zh" else info["en"]
        unit = info["unit"]
    if with_unit and unit and unit != "-":
        return f"{base} ({unit})"
    return base


def axis_label(key, lang="zh"):
    return AXIS_LABELS.get(key, {}).get(lang, key)


def stage_paper_label(stage, lang="zh"):
    return STAGE_LABELS_PAPER.get(stage, {}).get(lang, stage)


def misc_label(key, lang="zh"):
    return MISC_LABELS.get(key, {}).get(lang, key)


def target_slug(name, lang="en"):
    """用于文件名的目标短名，如 porosity_pct -> Porosity。"""
    info = PAPER_LABELS.get(name)
    if info is None:
        # Literature headers may contain '/', degree signs, parentheses or path
        # separators. Keep readable Unicode while preventing nested/invalid paths.
        import hashlib
        import re
        slug = re.sub(r"[^\w]+", "_", str(name), flags=re.UNICODE).strip("_")[:70]
        return slug or "field_" + hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8]
    base = info["abbr"] if lang == "zh" else info["en"]
    return "".join(ch for ch in base if ch.isalnum())


# ---------------------------------------------------------------------------
# CV / Stage 层级显示文案（真实方法字符串 → 用户可读文案）
#
# 原则（spec 三十一：自动识别但绝不盲目猜测）：
# 显示文案必须反映实际执行的验证方式——
#   - 无批次数据（batch_count 为 None/0）时实际执行的是随机 KFold，严禁显示
#     「分组交叉验证」；
#   - 只有 bundle 中确实训练成功了 Stage 2/3，才允许出现「全链」字样。
# ---------------------------------------------------------------------------

_MODEL_BEHAVIOUR_NOTE = {
    "zh": "该分析解释的是训练后模型如何使用输入变量，属于模型行为描述，不构成物理因果结论。",
    "en": ("This analysis describes how the trained model uses the input variables "
           "(model behaviour), not a physical causal conclusion."),
}


def model_behaviour_note(lang="zh"):
    """SHAP / 特征重要性图注与 Analysis.md 的统一免责声明（要求 D）。"""
    return _MODEL_BEHAVIOUR_NOTE.get(lang, _MODEL_BEHAVIOUR_NOTE["zh"])


def unit_from_header(name):
    """从数据列名末尾的括号中提取物理单位（要求 C）。

    文献表常把单位写进列名（如「粒子温度(°C)」「粒子速度(m/s)」「孔隙率(%)」）。
    仅当最后一个括号内的内容看起来像单位（≤12 字符，且由 °/字母/%/μ/·/斜杠/
    上标/数字组成、不是纯数字）时返回；否则返回 ""。
    """
    import re
    groups = re.findall(r"[（(]([^()（）]+)[)）]", str(name))
    if not groups:
        return ""
    cand = groups[-1].strip()
    if not cand or len(cand) > 12:
        return ""
    if not re.fullmatch(r"[°*a-zA-Zμ%/·⁻²³0-9.\-]+", cand):
        return ""
    if re.fullmatch(r"[\d.\-]+", cand):
        return ""
    if cand.lower() in ("c", "°c", "℃"):
        return "°C"
    return cand


def cv_display_method(method_str, batch_count=None, lang="zh"):
    """把 chain_cv.method 的真实方法字符串转为用户可读显示文案。

    - "GroupKFold(...)"（确实执行了分组切分）→ 「分组交叉验证（折外预测）」
    - "LOOCV (...)"                          → 「留一法交叉验证（折外预测）」
    - "KFold(n_splits=5, shuffle)"           → 「随机五折交叉验证（折外预测）」
    - method_str 缺失时按 batch_count 兜底：无批次 → 随机五折；有批次 → 分组。

    只改显示文字，不修改内部字段、数据与训练逻辑。
    """
    s = str(method_str or "").lower()
    if "loocv" in s or "leaveoneout" in s:
        return ("留一法交叉验证（折外预测）" if lang == "zh"
                else "Leave-one-out CV (out-of-fold predictions)")
    if "groupkfold" in s:
        return ("分组交叉验证（折外预测）" if lang == "zh"
                else "Grouped CV (out-of-fold predictions)")
    if "kfold" in s:
        return ("随机五折交叉验证（折外预测）" if lang == "zh"
                else "Randomized 5-fold CV (out-of-fold predictions)")
    try:
        n_batches = int(batch_count) if batch_count is not None else 0
    except (TypeError, ValueError):
        n_batches = 0
    if n_batches == 0:
        return ("随机五折交叉验证（折外预测）" if lang == "zh"
                else "Randomized 5-fold CV (out-of-fold predictions)")
    return ("分组交叉验证（折外预测）" if lang == "zh"
            else "Grouped CV (out-of-fold predictions)")


def stage_scope_label(trained_stages, lang="zh"):
    """按实际训练成功的层级生成预测范围标注。

    - 只训练了 Stage 1（可选 Stage 1.5）→ 「Stage 1 过程预测（折外验证）」
    - Stage 3 也训练成功                  → 才允许写「全链」
    - 只有 Stage 1–2                      → 「Stage 1–2 过程—缺陷预测（折外验证）」
    """
    stages = {str(s) for s in (trained_stages or [])}
    if not stages:
        return ("模型预测（折外验证）" if lang == "zh"
                else "Model prediction (out-of-fold validated)")
    if stages <= {"stage1", "stage15"}:
        return ("Stage 1 过程预测（折外验证）" if lang == "zh"
                else "Stage 1 process prediction (out-of-fold validated)")
    if "stage3" in stages:
        return ("全链预测（Stage 1→2→3，折外验证）" if lang == "zh"
                else "Full-chain prediction (Stage 1→2→3, out-of-fold validated)")
    return ("Stage 1–2 过程—缺陷预测（折外验证）" if lang == "zh"
            else "Stage 1–2 process–defect prediction (out-of-fold validated)")
