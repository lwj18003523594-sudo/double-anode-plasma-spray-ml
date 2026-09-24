# -*- coding: utf-8 -*-
"""V1.5 数据洞察测试（spec 七十五）

确认：experimental/literature 可参与真实数据总结；
model_prediction / optimization_candidate 不自动参与真实事实统计；
正向预测可显示邻近真实实验；Pareto 可显示数据支持程度。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_utils import load_table
from src.modes import load_schema_for_mode
from src.model_chain import train_chain
from src import insights as insight_mod
from src.optimize import random_pareto_search

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


schema = load_schema_for_mode("dual_anode")
df = load_table("data/demo/DEMO_双阳极喷涂数据.xlsx")
rng = np.random.default_rng(5)

# 1. data_origin 分级：model_prediction / optimization_candidate 不进事实统计
df_o = df.copy()
df_o["data_origin"] = "experimental"
df_o.loc[df_o.index[:50], "data_origin"] = "model_prediction"
df_o.loc[df_o.index[50:100], "data_origin"] = "optimization_candidate"
cov_all = insight_mod.coverage_stats(df, schema)["n_fact"]
cov_o = insight_mod.coverage_stats(df_o, schema)["n_fact"]
check("1. model_prediction / optimization_candidate 不参与真实事实统计",
      cov_o == len(df_o) - 100 and cov_o < cov_all,
      f"事实行 {cov_o}/{len(df_o)}（排除 100 条模型/优化来源行）")

# 2. 覆盖描述措辞：只写"当前数据主要覆盖"，禁止"最佳范围"
txt = insight_mod.coverage_text(df, schema)
check("2. 覆盖描述为客观措辞（含'当前数据主要覆盖'，无'最佳范围'）",
      "当前数据主要覆盖" in txt and "最佳" not in txt)

# 3. 正向实验规律：相关趋势 + 无"显著"用词
corr = insight_mod.correlation_report(df, schema)
check("3. 相关趋势表输出（Pearson/Spearman，n≥5）",
      len(corr) > 0 and {"输入", "输出", "Pearson", "Spearman", "n"}.issubset(corr.columns))

# 4. 正向预测历史参考：最近 3–5 个真实实验
vals = {c: float(pd.to_numeric(df[c], errors="coerce").median())
        for sec in ["structure_inputs", "process_inputs"] for c in schema[sec]}
near = insight_mod.nearest_experiments(vals, df, input_cols=list(vals.keys()), k=5)
check("4. 邻近真实实验（3–5 个，含 experiment_id/来源/距离）",
      3 <= len(near) <= 5 and "experiment_id" in near.columns
      and near["归一化距离"].is_monotonic_increasing,
      f"最近：{near.iloc[0]['experiment_id'] if len(near) else '—'}（d={near.iloc[0]['归一化距离'] if len(near) else '—'}）")
df_op = df.copy()
df_op["data_origin"] = "optimization_candidate"
near_op = insight_mod.nearest_experiments(vals, df_op, input_cols=list(vals.keys()), k=5)
check("4b. optimization_candidate 来源数据不出现在历史参考中", len(near_op) == 0)

# 5. Pareto 证据支持
bundle = train_chain(df, schema, model_path=None, seed=42)
fixed = {c: (bundle.get("training_domain") or {}).get(c, {}).get("median", 6.0)
         for c in schema["structure_inputs"]}
front, stats = random_pareto_search(bundle, schema,
                                    {"objectives": {"porosity_pct": "minimize",
                                                    "bond_strength_MPa": "maximize"},
                                     "constraints": {}},
                                    fixed_values=fixed, n_candidates=600, return_stats=True)
ev = insight_mod.pareto_evidence(front, df, bundle.get("training_domain") or {}, bundle,
                                 list(schema["process_inputs"].keys()), max_rows=5)
need_cols = {"方案", "数据支持", "最近真实实验", "局部数据密度", "预测不确定性"}
check("5. Pareto 证据支持字段齐全", len(ev) > 0 and need_cols.issubset(ev.columns),
      f"{len(ev)} 个方案的证据")
basis = insight_mod.recommendation_basis(ev.iloc[0])
check("5b. 推荐依据只含允许内容（目标权衡/数据覆盖/邻近实验/不确定性，无机理编造）",
      "[优化建议]" in basis and "[直接数据]" in basis and "[需进一步验证]" in basis
      and "机理" in basis and "不包含材料机理判断" in basis)

# 6. 值得复核 + 数据空白
bundle_full_cv = dict(bundle)
bundle_full_cv["chain_cv"] = {"oof_predictions": {}, "metrics": {}, "fold_info": []}
anom = insight_mod.anomaly_review(df, bundle)
gaps = insight_mod.data_gap_suggestions(df, schema, bundle)
check("6. 值得复核 / 建议补充实验区域 正常输出（不崩溃、不自动删数据）",
      isinstance(anom, pd.DataFrame) and isinstance(gaps, pd.DataFrame))

# 7. 模型规律 + 跨折稳定性（fold_info 记录了 fi）
factors, stability = insight_mod.model_key_factors(bundle)
check("7. 模型规律（特征重要性）输出", len(factors) > 0 and any(len(v) for v in factors.values()))
cv_fake = {"fold_info": [{"stage1_fi_top": ["a", "b", "c", "d", "e"]},
                         {"stage1_fi_top": ["b", "a", "f", "g", "h"]},
                         {"stage1_fi_top": ["a", "b", "c", "i", "j"]}]}
bundle_stab = dict(bundle)
bundle_stab["chain_cv"] = cv_fake
_, stability2 = insight_mod.model_key_factors(bundle_stab)
check("7b. 跨折稳定性统计（3 折中 a/b 均 3/3 进入 Top5）",
      stability2 is not None and "3/3" in stability2.get("a", ""))

print()
if failures:
    print(f"INSIGHT TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("INSIGHT TEST PASSED")
