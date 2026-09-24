# -*- coding: utf-8 -*-
"""V1.3.2 变更专项检查（不修改模型逻辑，只做检查）

子命令：
  python scripts/v132_check.py snapshot   改动前：对当前 latest_chain.joblib 固定输入快照全链预测
  python scripts/v132_check.py compare    改动后：重算并与基线逐值对比（证明模型预测结果不变）
  python scripts/v132_check.py            默认：V1.3.2 功能检查
        1) 恢复默认参数全部位于当前训练数据支持域内（双阳极间距不再默认 10.5 外推）；
        2) 训练数据支持域模式下 Pareto 不产生越界候选点，且不新增外推标记列（论文输出结构不变）；
        3) 设备允许可行域模式下，越界候选全部标记"存在外推风险"。

检查会训练新模型版本，照常写入 models/history/（旧版本不覆盖）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import ROOT, load_schema, load_objectives
from src.model_chain import train_chain_full, predict_chain
from src.optimize import random_pareto_search

DEMO = ROOT / "data" / "demo" / "DEMO_双阳极喷涂数据.xlsx"
MODEL = ROOT / "models" / "latest_chain.joblib"
BASE = ROOT / "outputs" / "v132_baseline_predictions.json"

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def _fixed_inputs():
    schema = load_schema()
    row_mid = {}
    for sec in ["structure_inputs", "process_inputs"]:
        for col, spec in schema[sec].items():
            row_mid[col] = (float(spec["min"]) + float(spec["max"])) / 2
    row_alt = dict(row_mid)
    row_alt.update({"anode_spacing_mm": 6.0, "nozzle_throat_mm": 8.0, "nozzle_exit_mm": 10.0,
                    "secondary_laval_length_mm": 30.0, "powder_injection_angle_deg": 60.0,
                    "anode1_current_A": 400.0, "anode2_current_A": 380.0, "Ar_flow_slpm": 50.0,
                    "H2_flow_slpm": 8.0, "powder_feed_g_min": 25.0, "spray_distance_mm": 100.0,
                    "traverse_speed_mm_s": 200.0})
    return pd.DataFrame([row_mid, row_alt])


def _predict_dict(bundle):
    X = _fixed_inputs()
    states, defects, perf, unc = predict_chain(X, bundle)
    return {"inputs": X.to_dict(orient="list"),
            "states": states.to_dict(orient="list"),
            "defects": defects.to_dict(orient="list"),
            "perf": perf.to_dict(orient="list"),
            "unc": unc.to_dict(orient="list")}


def _diffs(a, b, path="root"):
    out = []
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ"]
        for k in a:
            out += _diffs(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length differ"]
        for i, (u, v) in enumerate(zip(a, b)):
            if isinstance(u, float) or isinstance(v, float):
                fu, fv = float(u), float(v)
                if fu != fv and not (np.isnan(fu) and np.isnan(fv)):
                    out.append(f"{path}[{i}]: {u} != {v}")
            elif u != v:
                out.append(f"{path}[{i}]: {u} != {v}")
    elif a != b:
        out.append(f"{path}: {a} != {b}")
    return out


if len(sys.argv) > 1 and sys.argv[1] == "snapshot":
    import joblib
    bundle = joblib.load(MODEL)
    BASE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASE, "w", encoding="utf-8") as f:
        json.dump(_predict_dict(bundle), f, ensure_ascii=False, indent=1)
    print(f"BASELINE SAVED — model_version={bundle.get('model_version')} file={BASE.name}")
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "compare":
    import joblib
    bundle = joblib.load(MODEL)
    with open(BASE, encoding="utf-8") as f:
        base = json.load(f)
    diffs = _diffs(base, _predict_dict(bundle))
    if diffs:
        print(f"[FAIL] 模型预测结果与基线不一致（{len(diffs)} 处），示例：")
        for d in diffs[:10]:
            print("   ", d)
        sys.exit(1)
    print("[PASS] 模型预测结果与改动前基线完全一致（固定输入 2 行 × 全链三级输出逐值相等）")
    sys.exit(0)

# ---------------- 默认：V1.3.2 功能检查 ----------------
from src.data_utils import load_table, compute_default_inputs  # noqa: E402

schema = load_schema()
obj_cfg = load_objectives()
df = load_table(DEMO)

print("== 训练（n_splits=3，用于取得训练域与 Pareto）==")
bundle = train_chain_full(df, schema, MODEL, n_splits=3)
domain = bundle.get("training_domain") or {}
schema_inputs = {**schema["structure_inputs"], **schema["process_inputs"]}
opt_cols = [c for c, s in schema["process_inputs"].items() if s.get("optimizable")]

# ---- 1. 默认参数：训练数据中位数优先，且全部落在训练数据支持域内 ----
defaults = compute_default_inputs(schema, df)
bad = [c for c in defaults if c in domain and not (domain[c]["min"] <= defaults[c] <= domain[c]["max"])]
check("默认参数全部位于当前训练数据支持域内", not bad,
      f"越界: {bad}" if bad else f"{len(defaults)} 项全部在支持域内")
v = defaults.get("anode_spacing_mm")
dv = domain.get("anode_spacing_mm")
check("双阳极间距默认值落在训练数据范围内（不再默认 10.5 中点外推）",
      v is not None and dv and dv["min"] <= v <= dv["max"] and abs(v - 10.5) > 1e-9,
      f"默认 {v:.2f}，训练域 {dv['min']:.1f}–{dv['max']:.1f}" if dv else "无训练域")

# 无数据回退链
d_nodata = compute_default_inputs(schema, None)
check("无训练数据且无 default → 回退配置范围中点",
      all(abs(d_nodata[c] - (float(s["min"]) + float(s["max"])) / 2) < 1e-9
          for c, s in schema_inputs.items()))
schema_default = {"structure_inputs": {"anode_spacing_mm": {"min": 1.0, "max": 20.0, "default": 6.5}},
                  "process_inputs": {}}
check("无训练数据但有 config.default → 使用 default",
      abs(compute_default_inputs(schema_default, None)["anode_spacing_mm"] - 6.5) < 1e-9)

# ---- 2. 训练数据支持域模式（默认）：Pareto 不允许产生越界候选点 ----
fixed = {c: domain[c]["median"] for c in schema["structure_inputs"] if c in domain}
front_t, stats_t = random_pareto_search(bundle, schema, obj_cfg, fixed_values=fixed,
                                        n_candidates=1500, return_stats=True)
if front_t.empty:
    check("训练支持域模式：Pareto 存在可行解", False, "无可行解（需检查约束）")
else:
    n_out = 0
    for c in opt_cols:
        if c in domain:
            n_out += int(((front_t[c] < domain[c]["min"] - 1e-9) |
                          (front_t[c] > domain[c]["max"] + 1e-9)).sum())
    check("训练支持域模式：Pareto 可优化变量无越界候选点",
          n_out == 0, f"Pareto 解 {len(front_t)} 个，越界点 {n_out} 个")
    check("训练支持域模式：不新增外推标记列（论文输出结构不变）",
          "extrapolation_risk" not in front_t.columns)
    check("训练支持域模式：stats 记录搜索域", stats_t.get("search_domain") == "training")

# ---- 3. 设备允许可行域模式：越界候选必须标记"存在外推风险" ----
front_d, stats_d = random_pareto_search(bundle, schema, obj_cfg, fixed_values=fixed,
                                        n_candidates=1500, return_stats=True, search_domain="device")
if front_d.empty:
    check("设备允许域模式：存在可行解", False, "无可行解（需检查约束）")
else:
    check("设备允许域模式：输出含 extrapolation_risk 标记列", "extrapolation_risk" in front_d.columns)
    if "extrapolation_risk" in front_d.columns:
        consist, n_risk = True, 0
        for _, r in front_d.iterrows():
            is_out = any(c in domain and not (domain[c]["min"] - 1e-9 <= r[c] <= domain[c]["max"] + 1e-9)
                         for c in opt_cols)
            marked = r["extrapolation_risk"] == "存在外推风险"
            n_risk += int(marked)
            if is_out != marked:
                consist = False
        check("设备允许域模式：外推标记与训练域判定逐行一致", consist,
              f"Pareto 解 {len(front_d)} 个，标记外推 {n_risk} 个")
    check("设备允许域模式：stats 记录搜索域与外推方案数",
          stats_d.get("search_domain") == "device" and "n_extrap" in stats_d)

# ---- 4. 预测一致性（若存在改动前基线则自动对比） ----
if BASE.exists():
    import joblib
    with open(BASE, encoding="utf-8") as f:
        base = json.load(f)
    diffs = _diffs(base, _predict_dict(bundle))
    check("模型预测结果与改动前基线完全一致", not diffs,
          f"{len(diffs)} 处差异" if diffs else "固定输入 2 行 × 全链输出逐值相等")
else:
    print("[SKIP] 未找到基线文件 outputs/v132_baseline_predictions.json（先运行 snapshot 子命令）")

print()
if failures:
    print(f"V1.3.2 CHECK FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("V1.3.2 CHECK PASSED")
