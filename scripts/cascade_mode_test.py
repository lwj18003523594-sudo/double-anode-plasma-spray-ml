# -*- coding: utf-8 -*-
"""V1.4 多电极 / 级联模式测试（spec 六/三十五）

检查：独立 Schema（全部字段可缺省）、合成级联数据端到端训练/预测、
可选熔融层在级联模式下同样可用。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.modes import load_schema_for_mode
from src.model_chain import train_chain, predict_chain, cross_validate_chain

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


schema_c = load_schema_for_mode("cascade")

# 1. 独立 Schema：全部字段可缺省；关键级联结构字段存在
all_optional = all(sp.get("required", False) is False
                   for sec in ["meta", "structure_inputs", "process_inputs", "process_states",
                               "defect_network", "performance_outputs"]
                   for sp in (schema_c.get(sec) or {}).values())
struct_cols = list(schema_c["structure_inputs"].keys())
check("1. cascade Schema 独立且全部字段可缺省", all_optional,
      f"结构 {len(struct_cols)} / 工艺 {len(schema_c['process_inputs'])} 项")
check("1b. 含电极构型字段",
      all(c in struct_cols for c in ["electrode_count", "cathode_count", "cascade_stage_count",
                                     "cascade_length_mm", "anode_cathode_distance_mm"]))
check("1c. 与双阳极字段互不映射（无 anode1_current_A）",
      "anode1_current_A" not in schema_c["process_inputs"]
      and "arc_current_A" in schema_c["process_inputs"])

# 2. 合成级联数据端到端（仅测试夹具）
rng = np.random.default_rng(33)
n = 120
df_c = pd.DataFrame({
    "experiment_id": [f"C{i:03d}" for i in range(n)],
    "batch_id": [f"B{i % 12:02d}" for i in range(n)],
    "electrode_count": rng.integers(2, 5, n).astype(float),
    "cathode_count": rng.integers(1, 3, n).astype(float),
    "cascade_stage_count": rng.integers(2, 5, n).astype(float),
    "cascade_length_mm": rng.uniform(80, 240, n),
    "arc_current_A": rng.uniform(300, 700, n),
    "arc_voltage_V": rng.uniform(50, 90, n),
    "Ar_flow_slpm": rng.uniform(40, 90, n),
    "powder_feed_g_min": rng.uniform(15, 60, n),
    "spray_distance_mm": rng.uniform(90, 160, n),
})
df_c["jet_temperature_K"] = 9000 + 3 * df_c["arc_current_A"] / 10 + rng.normal(0, 120, n)
df_c["particle_temperature_C"] = 2200 + 0.4 * df_c["arc_current_A"] + rng.normal(0, 60, n)
df_c["particle_velocity_m_s"] = 380 + 0.2 * df_c["arc_current_A"] + rng.normal(0, 25, n)
df_c["porosity_pct"] = np.clip(6 - 0.004 * df_c["particle_temperature_C"] + rng.normal(0, 0.3, n), 0.2, 8)
df_c["bond_strength_MPa"] = np.clip(30 + 2.2 * df_c["porosity_pct"] * -1 + 40 + rng.normal(0, 2, n), 20, 90)

b0 = train_chain(df_c, schema_c, model_path=None, seed=42)
check("2. 级联模式三级链端到端训练", b0["stage15_enabled"] is False
      and b0["stage1_models"] and b0["stage2_models"] and b0["stage3_models"])
s, d, p, u = predict_chain(df_c.head(4), b0)
check("2b. 级联模式预测正常（3 个过程状态 + 1 个性能目标）",
      s.shape == (4, 3) and p.shape == (4, 1))
cv = cross_validate_chain(df_c, schema_c, seed=42, n_splits=3)
check("2c. 级联模式分组 CV 正常（batch 分组防泄漏）",
      "GroupKFold" in cv["method"] and cv["metrics"]["stage1"])

# 3. 级联 + 熔融层
df_m = df_c.copy()
mi = rng.uniform(0.3, 1.1, n)
df_m["melting_index"] = mi
df_m["melting_state_class"] = pd.qcut(pd.Series(mi).rank(method="first"), 3,
                                      labels=["insufficiently_molten", "partially_molten",
                                              "fully_molten"]).astype(str)
b1 = train_chain(df_m, schema_c, model_path=None, seed=42)
check("3. 级联模式熔融层可用", b1["stage15_enabled"] is True
      and "pred_melting_index" in b1["stage2_features"])

print()
if failures:
    print(f"CASCADE MODE TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("CASCADE MODE TEST PASSED")
