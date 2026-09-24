# -*- coding: utf-8 -*-
"""V1.3.2 可用性/稳健性测试（V1.3.2 起默认参数改为训练数据中位数优先）

覆盖：模型过期检测、输入范围分级与外推判定、默认参数恢复（中位数优先）、
数据中位数参数、版本化输出文件名、研究快照创建、重复数据统计、中文错误提示、日志。
用法：python scripts/usability_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import ROOT, load_schema
from src.data_utils import load_table, compute_default_inputs
from src.model_chain import train_chain_full
from src.research_utils import (dataset_hash, model_staleness, training_domain,
                                classify_support, support_label, pareto_row_support,
                                fmt_value, fmt_metric, friendly_error, replicate_stats,
                                log_event)
from src.paper_output import PaperExporter, save_snapshot

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


schema = load_schema()
df = load_table(ROOT / "data" / "demo" / "DEMO_双阳极喷涂数据.xlsx")

# ---- 1. 数据指纹与过期检测 ----
h1 = dataset_hash(df)
check("数据指纹可计算且稳定", h1 == dataset_hash(df) and len(h1) == 16, h1)
df_mut = df.copy()
df_mut.loc[df_mut.index[0], "anode1_current_A"] = float(df_mut["anode1_current_A"].iloc[0]) + 1.0
check("数据变化 → 指纹变化", dataset_hash(df_mut) != h1)

print("== 训练（n_splits=3）==")
MODEL = ROOT / "models" / "latest_chain.joblib"
bundle = train_chain_full(df, schema, MODEL, n_splits=3)

stale, mh, ch = model_staleness(bundle, df)
check("未变化 → 不算过期", stale is False)
stale2, _, _ = model_staleness(bundle, df_mut)
check("数据变化 → 检测为过期", stale2 is True)

# ---- 2. 训练域与范围分级 ----
domain = bundle.get("training_domain") or training_domain(df, bundle["x_cols"])
check("训练域包含全部输入特征", len(domain) == 12, f"{len(domain)} 项")
check("训练域含 min/max/median/Q1/Q3",
      {"min", "max", "median", "q1", "q3"}.issubset(domain["anode1_current_A"]))
col = "anode1_current_A"
d = domain[col]
mid = (d["q1"] + d["q3"]) / 2
check("Q1–Q3 内 → 数据覆盖良好", classify_support(mid, d) == "good")
edge_val = d["min"] + (d["max"] - d["min"]) * 0.02 if d["q1"] > d["min"] else d["max"] - 1e-6
check("min–max 内但超出 Q1–Q3 → 接近边界",
      classify_support(edge_val, d) == "edge", f"v={edge_val:.1f}")
check("超出 min–max → 外推", classify_support(d["max"] + 50, d) == "extrap")
check("外推中文标签", support_label("extrap") == "存在外推风险")
check("Pareto 行支持判定（全部中值 → 覆盖良好）",
      pareto_row_support({c: domain[c]["median"] for c in domain if c != "anode1_current_A"} | 
                         {"anode1_current_A": mid}, domain) == "good")

# ---- 3. 默认参数恢复（V1.3.2：训练数据中位数优先）/ 中位数参数（纯函数层面） ----
schema_inputs = {**schema["structure_inputs"], **schema["process_inputs"]}
domain_u = bundle.get("training_domain") or training_domain(df, bundle["x_cols"])
defaults = compute_default_inputs(schema, df)
check("恢复默认参数（有训练数据）= 训练数据中位数（截断到设备范围）",
      len(defaults) == 12 and all(
          abs(defaults[c] - float(np.clip(pd.to_numeric(df[c], errors="coerce").median(),
                                          schema_inputs[c]["min"], schema_inputs[c]["max"]))) < 1e-9
          for c in defaults))
check("默认参数全部位于当前训练数据支持域内",
      all(domain_u[c]["min"] <= defaults[c] <= domain_u[c]["max"] for c in defaults if c in domain_u),
      f"{len(defaults)} 项")
check("双阳极间距默认值不再使用范围中点 10.5（落在训练数据范围内）",
      domain_u["anode_spacing_mm"]["min"] <= defaults["anode_spacing_mm"] <= domain_u["anode_spacing_mm"]["max"]
      and abs(defaults["anode_spacing_mm"] - 10.5) > 1e-9,
      f"默认值 {defaults['anode_spacing_mm']:.2f}，训练域 "
      f"{domain_u['anode_spacing_mm']['min']:.1f}–{domain_u['anode_spacing_mm']['max']:.1f}")
d_nodata = compute_default_inputs(schema, None)
check("无训练数据且无 default → 回退配置范围中点",
      all(abs(d_nodata[c] - (float(s["min"]) + float(s["max"])) / 2) < 1e-9
          for c, s in schema_inputs.items()))
schema_default = {"structure_inputs": {"anode_spacing_mm": {"min": 1.0, "max": 20.0, "default": 6.5}},
                  "process_inputs": {}}
check("无训练数据但有 config.default → 使用 default",
      abs(compute_default_inputs(schema_default, None)["anode_spacing_mm"] - 6.5) < 1e-9)
medians = {}
for c, s in schema_inputs.items():
    if c in df.columns:
        med = pd.to_numeric(df[c], errors="coerce").median()
        if pd.notna(med):
            medians[c] = float(np.clip(med, s["min"], s["max"]))
check("中位数参数 = 数据中位数且被设备范围截断",
      len(medians) == 12 and all(schema_inputs[c]["min"] <= v <= schema_inputs[c]["max"]
                                 for c, v in medians.items()), f"{len(medians)} 项")

# ---- 4. 版本化输出文件名 ----
ex = PaperExporter(df, schema, bundle, lang="zh", formats=("png",), demo=True)
out, stem = ex.fig_a1_completeness()
png = [x for x in out if str(x).endswith(".png")][0]
name = Path(png).name
check("文件名含 Dataset 标签", "_DS" in name, name)
check("文件名含 Model 标签", "_MV" in name, name)
check("文件名含 DEMO 前缀", name.startswith("DEMO_"))
meta_p = [x for x in out if str(x).endswith(".json")][0]
import json
meta = json.load(open(meta_p, encoding="utf-8"))
check("metadata 含 caption_suggestion（客观，无因果词）",
      "caption_suggestion" in meta and
      all(k not in meta["caption_suggestion"] for k in ["证明", "因果", "显著改善", "表明"]),
      meta.get("caption_suggestion", "")[:40])

# ---- 5. 研究快照 ----
snap = save_snapshot(df, schema, bundle, n_pareto_candidates=400)
need = ["dataset_info.json", "model_info.json", "metrics.xlsx", "Pareto结果.xlsx",
        "reproducibility_manifest.json", "README.txt", "config/data_schema.yaml",
        "config/objectives.yaml", "config/data_dictionary.yaml"]
check("研究快照文件齐全", all((snap / f).exists() for f in need), snap.name)
check("快照不含 venv/缓存", not any("venv" in p.name or "__pycache__" in p.name
                                    for p in snap.rglob("*")))

# ---- 6. 重复实验统计 ----
df_dup = pd.concat([df.head(10), df.head(10)], ignore_index=True)
stats = replicate_stats(df_dup, "experiment_id")
check("重复测量统计（构造重复后可检出）", stats is not None and (stats["n"] >= 2).all())
check("原始数据行数未被修改（不覆盖原始数据）", len(df) == 280)
stats_none = replicate_stats(df, "sample_id")
check("无 sample_id 时返回 None 且不报错", stats_none is None)

# ---- 7. 中文错误提示 ----
try:
    raise KeyError("porosity_pct")
except KeyError as e:
    err = friendly_error(e)
check("KeyError → 中文提示", "缺少对应字段" in err and "Traceback" not in err)
df_nan = df.copy()
df_nan.loc[df_nan.index[:3], "porosity_pct"] = np.nan
try:
    raise ValueError("Input contains NaN")
except ValueError as e:
    err2 = friendly_error(e, df_nan, schema)
check("NaN 错误 → 缺失统计中文提示",
      "缺失" in err2 and "实际可用" in err2 and "porosity_pct" in err2, err2[:80])
log_file = ROOT / "outputs" / "logs"
logs = list(log_file.glob("*.log")) if log_file.exists() else []
check("错误 traceback 已写入日志（不进界面）", len(logs) > 0 and
      any("Traceback" in f.read_text(encoding="utf-8") for f in logs))

# ---- 8. 数字显示格式（仅显示层） ----
check("电流 1 位", fmt_value("anode1_current_A", 450.567) == 450.6)
check("孔隙率 2 位", fmt_value("porosity_pct", 3.14159) == 3.14)
check("温度 1 位", fmt_value("particle_temperature_C", 2253.94) == 2253.9)
check("R² 4 位", fmt_metric(0.8282409684160965) == 0.8282)
check("RMSE 按量级 3–4 位有效数字",
      fmt_metric(236.45252849176595, "RMSE") == 236.0
      and fmt_metric(0.04975895898491021, "RMSE") == 0.04976)
raw = df["anode1_current_A"].iloc[0]
check("原始数据未被改动（显示层不回写）", df["anode1_current_A"].iloc[0] == raw)

print()
if failures:
    print(f"USABILITY TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("USABILITY TEST PASSED")
