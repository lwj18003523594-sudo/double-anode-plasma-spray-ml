# -*- coding: utf-8 -*-
"""统一中文显示映射层（V1.1）

底层 DataFrame / 模型接口一律保持英文标准字段名；
所有网页、表格、图表、下拉菜单、按钮的中文显示统一从本文件调用。
禁止在 app.py 中手写零散映射。
"""

# ---- 字段级映射：内部英文名 -> 界面中文名（含单位） ----
COLUMN_LABELS = {
    # 基础信息
    "experiment_id": "实验编号",
    "batch_id": "喷涂批次",

    # 双阳极结构参数
    "nozzle_throat_mm": "喷嘴喉部直径（mm）",
    "nozzle_exit_mm": "喷嘴出口直径（mm）",
    "secondary_laval_length_mm": "二级拉瓦尔段长度（mm）",
    "anode_spacing_mm": "双阳极间距（mm）",
    "powder_injection_angle_deg": "送粉注入角（°）",

    # 工艺参数
    "anode1_current_A": "阳极1电流（A）",
    "anode2_current_A": "阳极2电流（A）",
    "Ar_flow_slpm": "Ar 气体流量（L/min）",
    "H2_flow_slpm": "H₂ 气体流量（L/min）",
    "powder_feed_g_min": "送粉速率（g/min）",
    "spray_distance_mm": "喷涂距离（mm）",
    "traverse_speed_mm_s": "喷枪移动速度（mm/s）",

    # 射流与粒子状态
    "arc_voltage_V": "电弧电压（V）",
    "jet_temperature_K": "射流温度（K）",
    "jet_velocity_m_s": "射流速度（m/s）",
    "particle_temperature_C": "粒子温度（℃）",
    "particle_velocity_m_s": "粒子速度（m/s）",

    # 涂层缺陷网络
    "porosity_pct": "孔隙率（%）",
    "lamellar_gap_pct": "层间未结合率（%）",
    "unmelted_particle_pct": "未熔颗粒比例（%）",
    "crack_density_mm_mm2": "裂纹密度（mm/mm²）",
    "defect_connectivity_index": "缺陷连通度指数",

    # 涂层性能
    "bond_strength_MPa": "结合强度（MPa）",
    "hardness_HV": "显微硬度（HV）",
    "wear_rate_mg_m": "磨损率（mg/m）",
    "corrosion_current_uA_cm2": "腐蚀电流密度（μA/cm²）",
    "cavitation_mass_loss_mg": "空蚀失重（mg）",
    "coupled_damage_rate": "耦合损伤指标",
    "deposition_efficiency_pct": "沉积效率（%）",

    # 物理派生变量
    "total_current_A": "总电流（A）",
    "current_difference_A": "双阳极电流差（A）",
    "current_imbalance_ratio": "双阳极电流不平衡系数",
    "total_gas_flow_slpm": "总气体流量（L/min）",
    "H2_Ar_ratio": "H₂/Ar 气体流量比",
    "electrical_power_W": "电功率（W）",
    "specific_power_W_per_g_min": "单位送粉功率",
    "particle_flight_time_ms_proxy": "粒子飞行时间代理量",

    # ---- V1.4 材料 / 粉末属性层 ----
    "material_name": "材料名称",
    "material_category": "材料类别",
    "powder_morphology": "粉末形貌",
    "powder_d10_um": "粉末 D10（μm）",
    "powder_d50_um": "粉末 D50（μm）",
    "powder_d90_um": "粉末 D90（μm）",
    "melting_temperature_C": "材料熔点（℃）",
    "density_kg_m3": "密度（kg/m³）",
    "heat_capacity_J_kgK": "比热容（J/(kg·K)）",
    "thermal_conductivity_W_mK": "导热系数（W/(m·K)）",
    "latent_heat_J_kg": "熔化潜热（J/kg）",
    "feedstock_type": "原料类型",

    # ---- V1.4 颗粒熔融与沉积状态层（Stage 1.5） ----
    "melting_state_class": "颗粒熔融状态",
    "melting_fraction_pct": "熔融比例（%）",
    "melting_index": "熔融指数",
    "thermal_margin_C": "粒子温度相对熔点差值（℃）",
    "overheating_risk": "过热风险",
    "splat_state_class": "铺展状态",
    "spread_factor": "铺展因子",
    "splash_fragmentation_risk": "飞溅破碎风险",
    "melting_data_source": "熔融数据来源",

    # ---- V1.4 传统 APS / 多电极级联模式 ----
    "torch_type": "喷枪类型",
    "arc_current_A": "弧电流（A）",
    "arc_voltage_fluctuation": "电弧电压波动",
    "spray_power_kW": "喷涂功率（kW）",
    "N2_flow_slpm": "N₂ 气体流量（L/min）",
    "He_flow_slpm": "He 气体流量（L/min）",
    "carrier_gas_flow_slpm": "送粉载气流量（L/min）",
    "substrate_rpm": "基体转速（rpm）",
    "spray_angle_deg": "喷涂角度（°）",
    "particle_diameter_um": "粒子直径（μm）",
    "oxide_content_pct": "氧化物含量（%）",
    "thermal_conductivity_coating_W_mK": "涂层导热系数（W/(m·K)）",
    "electrode_count": "电极数量",
    "cathode_count": "阴极数量",
    "anode_count": "阳极数量",
    "cascade_stage_count": "级联级数",
    "cascade_ring_count": "级联环数量",
    "cascade_length_mm": "级联段长度（mm）",
    "anode_cathode_distance_mm": "阴阳极间距（mm）",
    "powder_injection_position_mm": "送粉位置（mm）",
    "particle_flux": "粒子通量",
    "electrode_pair_count": "电极对数量",
}

# ---- 模型层级映射 ----
STAGE_LABELS = {
    "stage1": "一级模型",
    "stage2": "二级模型",
    "stage3": "三级模型",
}

STAGE_FULL_LABELS = {
    "stage1": "一级模型：射流与粒子状态",
    "stage2": "二级模型：涂层缺陷网络",
    "stage3": "三级模型：涂层性能",
}

# ---- 模型类型映射 ----
MODEL_KIND_LABELS = {
    "XGBoost": "XGBoost",
    "GaussianProcess": "高斯过程回归（GP）",
    "RandomForest": "随机森林（RF）",
}

# ---- 评价指标表头映射 ----
METRIC_LABELS = {
    "stage": "模型层级",
    "target": "预测指标",
    "R2": "决定系数 R²",
    "RMSE": "均方根误差 RMSE",
    "MAE": "平均绝对误差 MAE",
    "model": "模型类型",
}

# ---- 质量检查表头映射 ----
QUALITY_LABELS = {
    "column": "数据字段",
    "n": "总数据量",
    "valid": "有效数据",
    "missing": "缺失数据",
    "missing_pct": "缺失比例（%）",
}

# ---- 带单位的简称（用于结果卡片等紧凑场合） ----
COLUMN_SHORT = {
    "porosity_pct": "孔隙率",
    "bond_strength_MPa": "结合强度",
    "coupled_damage_rate": "耦合损伤",
    "deposition_efficiency_pct": "沉积效率",
}


def zh(name):
    """内部英文名 -> 中文名；未登记的原样返回。"""
    return COLUMN_LABELS.get(name, name)


def zh_short(name):
    """不带单位的中文简称。"""
    return COLUMN_SHORT.get(name, COLUMN_LABELS.get(name, name))


def zh_df(df, columns=None):
    """仅用于显示的表头中文转换；不修改底层 DataFrame。"""
    cols = list(columns) if columns is not None else list(df.columns)
    rename = {c: zh(c) for c in cols if c in COLUMN_LABELS}
    return df.rename(columns=rename)
