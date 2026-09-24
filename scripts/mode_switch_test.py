# -*- coding: utf-8 -*-
"""V1.4 模式切换测试（spec 一/三/四/三十二/三十三/三十四/三十五/四十）

检查：四种研究模式注册、按模式隔离的数据/模型路径、输出前缀、
Schema 加载与可选层合并、模型状态不跨模式继承、CV 方式建议。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modes import (MODES, DEFAULT_MODE, load_schema_for_mode, data_dir, data_file,
                       model_path as mode_model_path, output_prefix, suggest_cv)

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# 1. 四种模式
check("1. 存在四种喷涂/研究模式",
      list(MODES.keys()) == ["dual_anode", "conventional_aps", "cascade", "literature"],
      "、".join(MODES[m]["label"] for m in MODES))
check("1b. 默认模式为双阳极", DEFAULT_MODE == "dual_anode")

# 2. 数据来源动态变化
check("2. 双阳极有 模拟/文献/真实 三种来源",
      MODES["dual_anode"]["sources"] == ["模拟演示数据", "文献数据", "真实实验数据"])
check("2b. APS 与级联只有 文献/真实 两种来源",
      MODES["conventional_aps"]["sources"] == ["文献数据", "真实实验数据"]
      and MODES["cascade"]["sources"] == ["文献数据", "真实实验数据"])
check("2c. 通用文献模式固定文献数据",
      MODES["literature"]["source_kind"] == {"文献数据（固定）": "literature"})

# 3. 数据/模型目录按模式隔离
dirs = {m: data_dir(m, "real") for m in MODES if m != "dual_anode"}
dirs["dual_anode"] = data_dir("dual_anode", "real")
check("3. 数据目录按模式完全隔离", len(set(dirs.values())) == 4,
      "；".join(str(v) for v in dirs.values()))
mpaths = {m: mode_model_path(m) for m in MODES}
check("3b. 模型路径按模式隔离（dual 回退兼容旧根目录模型）",
      len({str(p) for p in mpaths.values()}) >= 3,
      "；".join(str(v) for v in mpaths.values()))

# 4. 输出前缀（spec 四十）
check("4. 输出前缀正确",
      output_prefix("dual_anode", "demo") == "DEMO_"
      and output_prefix("dual_anode", "real") == "DUAL_"
      and output_prefix("conventional_aps", "real") == "APS_"
      and output_prefix("cascade", "real") == "CASCADE_"
      and output_prefix("cascade", "literature") == "LIT_"
      and output_prefix("literature", "literature") == "LIT_")

# 5. 各模式 Schema 可加载且可选层全部允许缺省
for m in MODES:
    sch = load_schema_for_mode(m)
    layers_ok = all(sp.get("required", False) is False
                    for sec in ["material_inputs", "melting_states"]
                    for sp in (sch.get(sec) or {}).values())
    check(f"5. Schema 可加载且可选层全部可缺省：{m}", layers_ok,
          f"材料 {len(sch.get('material_inputs') or {})} 项 / 熔融 {len(sch.get('melting_states') or {})} 项")

# 6. 模型状态不跨模式继承（路径层面：未训练模式下无模型文件 → get_bundle 等价 None）
aps_mp = mode_model_path("conventional_aps")
check("6. 未训练模式不继承他模式模型（APS 独立路径按存在性判定）",
      str(aps_mp).endswith(str(Path("models") / "conventional_aps" / "latest_chain.joblib")))

# 7. CV 方式建议（spec 三十八）
t1, _ = suggest_cv(20, True)
t2, _ = suggest_cv(20, False)
t3, _ = suggest_cv(40, False)
check("7. CV 建议：有批次→GroupKFold；n≤25→LOOCV；26–60→5折KFold",
      "GroupKFold" in t1 and "LOOCV" in t2 and "KFold" in t3, f"{t1} / {t2} / {t3}")

print()
if failures:
    print(f"MODE SWITCH TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("MODE SWITCH TEST PASSED")
