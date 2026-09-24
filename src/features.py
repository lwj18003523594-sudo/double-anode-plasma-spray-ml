import numpy as np
import pandas as pd

def add_physics_features(df):
    x = df.copy()
    eps = 1e-9
    required = set(x.columns)

    if {"anode1_current_A","anode2_current_A"}.issubset(required):
        x["total_current_A"] = x["anode1_current_A"] + x["anode2_current_A"]
        x["current_difference_A"] = (x["anode1_current_A"] - x["anode2_current_A"]).abs()
        x["current_imbalance_ratio"] = x["current_difference_A"] / (x["total_current_A"] + eps)

    # V1.4：APS / 级联模式的电功率派生（单阴极弧电流体系，不与双阳极互映射）
    if {"arc_current_A","arc_voltage_V"}.issubset(required) and "electrical_power_W" not in x.columns:
        x["electrical_power_W"] = x["arc_voltage_V"] * x["arc_current_A"]

    if {"Ar_flow_slpm","H2_flow_slpm"}.issubset(required):
        x["total_gas_flow_slpm"] = x["Ar_flow_slpm"] + x["H2_flow_slpm"]
        x["H2_Ar_ratio"] = x["H2_flow_slpm"] / (x["Ar_flow_slpm"] + eps)

    if {"arc_voltage_V","total_current_A"}.issubset(set(x.columns)):
        x["electrical_power_W"] = x["arc_voltage_V"] * x["total_current_A"]

    if {"electrical_power_W","powder_feed_g_min"}.issubset(set(x.columns)):
        x["specific_power_W_per_g_min"] = x["electrical_power_W"] / (x["powder_feed_g_min"] + eps)

    if {"spray_distance_mm","particle_velocity_m_s"}.issubset(set(x.columns)):
        x["particle_flight_time_ms_proxy"] = x["spray_distance_mm"] / (x["particle_velocity_m_s"] + eps)

    return x
