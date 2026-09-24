# -*- coding: utf-8 -*-
"""V1.4 通用文献模式测试（spec 四/二十九/三十六/三十七/三十八）

检查：多工作表选择（禁止默认 README 说明表）、文献库读取、
熔融字段"建议映射"、运行时 Schema 构建、小样本端到端训练、CV 建议。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.literature import (list_sheets, pick_default_sheet, load_sheet,
                            suggest_melting_mapping, build_runtime_schema,
                            default_objectives_for)
from src.model_chain import train_chain, predict_chain, cross_validate_chain
from src.data_utils import load_table

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


TMP = Path(__file__).resolve().parents[1] / "outputs" / "v14_test_tmp"
TMP.mkdir(parents=True, exist_ok=True)
XLSX = TMP / "Literature_Test_Database_PlasmaSpray_ML.xlsx"

# 构造文献数据库夹具（README + 数据 + 字典 三个工作表）
rng = np.random.default_rng(55)
n = 40
data = pd.DataFrame({
    "experiment_id": [f"L{i:03d}" for i in range(n)],
    "arc_current_A": rng.uniform(300, 700, n),
    "spray_distance_mm": rng.uniform(80, 160, n),
    "powder_d50_um": rng.uniform(30, 90, n),
    "melting_degree": rng.uniform(0.2, 0.95, n),          # 文献自定义熔融指标（别名）
    "porosity_pct": rng.uniform(0.5, 6, n),
    "bond_strength_MPa": rng.uniform(25, 80, n),
})
readme = pd.DataFrame({"说明": ["本工作表为使用说明，不是训练数据", "字段见 字典 工作表"]})
dict_sheet = pd.DataFrame({"字段": list(data.columns), "含义": ["编号", "弧电流", "喷距", "粒径", "熔融程度", "孔隙率", "结合强度"]})
with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
    readme.to_excel(w, sheet_name="README_使用说明", index=False)
    data.to_excel(w, sheet_name="Data", index=False)
    dict_sheet.to_excel(w, sheet_name="字段说明", index=False)

# 1. 多工作表列出与默认选择（禁止默认 README）
sheets = list_sheets(XLSX)
check("1. 多工作表可列出", len(sheets) == 3, str(sheets))
check("1b. 默认选择跳过 README/说明表", pick_default_sheet(sheets) == "Data",
      f"默认: {pick_default_sheet(sheets)}")

# 2. 文献库读取（若用户真实库存在也一并验证）
lit_df = load_sheet(XLSX, "Data")
check("2. 指定工作表读取正常", lit_df.shape == (n, 7))
existing_lit = list(Path(__file__).resolve().parents[1].glob("data/**/Literature_Test_Database_PlasmaSpray_ML.xlsx"))
if existing_lit:
    try:
        load_table(existing_lit[0])
        check("2b. 用户真实 Literature_Test_Database 可正常读取", True, existing_lit[0].name)
    except Exception as e:
        check("2b. 用户真实 Literature_Test_Database 可正常读取", False, repr(e))
else:
    check("2b. 用户真实库未放置（跳过，不判失败）", True)

# 3. 熔融字段建议映射（spec 29：仅建议，用户确认）
sugg = suggest_melting_mapping(lit_df.columns)
check("3. 建议映射识别 melting_degree → melting_index",
      sugg.get("melting_index") == "melting_degree", str(sugg))

# 4. 运行时 Schema 构建与端到端训练
sel = {"x": ["arc_current_A", "spray_distance_mm", "powder_d50_um"],
       "states": [], "defects": ["porosity_pct"], "performance": ["bond_strength_MPa"],
       "material": ["powder_d50_um"], "melting_map": {"melting_index": "melting_degree"}}
schema_l = build_runtime_schema(lit_df, sel)
check("4. 运行时 Schema 构建（X/缺陷/性能/熔融）",
      len(schema_l["process_inputs"]) == 3 and "porosity_pct" in schema_l["defect_network"]
      and schema_l["melting_states"].get("melting_index") is not None)
# 与 app 行为一致：熔融映射列重命名为规范字段名
lit_renamed = lit_df.rename(columns={"melting_degree": "melting_index"})
b = train_chain(lit_renamed, schema_l, model_path=None, seed=42)
check("4b. 文献模式端到端训练（熔融层经用户映射后启用）",
      b["stage15_enabled"] is True and "melting_index" in b["stage15_models"])
s, d, p, u, m = predict_chain(lit_renamed.head(3), b, return_melting=True)
check("4c. 文献模式预测正常（含熔融输出）",
      "melting_index" in m["continuous"].columns and p.shape[1] == 1)
cv = cross_validate_chain(lit_df, schema_l, seed=42)
check("4d. 无 batch 时 CV 自动回退（n=40 → KFold）", "KFold" in cv["method"], cv["method"])

# 5. 小样本 LOOCV 建议
small = lit_df.head(18)
bs = train_chain(small, schema_l, model_path=None, seed=42)
cvs = cross_validate_chain(small, schema_l, seed=42)
check("5. n≤25 无批次 → LOOCV 全链交叉验证", "LOOCV" in cvs["method"], cvs["method"])

print()
if failures:
    print(f"LITERATURE MODE TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("LITERATURE MODE TEST PASSED")
