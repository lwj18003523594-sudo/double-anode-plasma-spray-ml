# -*- coding: utf-8 -*-
"""V1.5 图表科研解读测试（spec 七十六）

1. 能根据图表数据生成现象描述；
2. 保留原始数字；
3. 正确区分直接数据/模型推断；
4. 无文献时不生成虚假引用；
5. 无机理证据时自动使用谨慎措辞；
6. 外推情况下自动生成局限性提醒；
7. Demo 明确标记不可用于正式科研结论；
8. Analysis.md 正常导出。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.figure_analysis import (analyze_figure, describe_trend, write_analysis_md,
                                 summarize_analyses)

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


TMP = Path(__file__).resolve().parents[1] / "outputs" / "v15_test_tmp"
TMP.mkdir(parents=True, exist_ok=True)

# 1. 现象描述（真实序列 → 可观测模式）
x = np.linspace(0, 10, 30)
t_up = describe_trend(x, 2 * x + 1)
t_peak = describe_trend(x, -(x - 5) ** 2 + 10)
t_flat = describe_trend(x, np.full_like(x, 3.0))
check("1. 现象描述：上升/峰值/平台可识别",
      "上升" in t_up and "峰值" in t_peak and "平台" in t_flat,
      f"上升={'上升' in t_up} 峰值={'峰值' in t_peak} 平台={'平台' in t_flat}")

# 2-7. analyze_figure 生成与证据约束
md = analyze_figure("parity", target="结合强度", n=280, r2=0.8421, rmse=3.0144, mae=2.4687,
                    model_type="XGBoost", cv_method="GroupKFold OOF",
                    trend_desc="散点围绕 y = x 参考线分布。")
check("2. 保留原始数字（R²=0.842 / RMSE=3.01 原样呈现）",
      "0.842" in md and "3.01" in md and "n=280" in md)
check("3. 区分 [直接数据] 与 [模型推断]",
      "[直接数据]" in md and "[模型推断]" in md)
check("4. 无文献时不生成虚假引用（明示不作引用，无 DOI 编造）",
      "不作文献引用" in md and "doi" not in md.lower())
check("5. 无机理证据时使用谨慎措辞（推测/仍需进一步验证，不写'证明了'）",
      "仍需进一步验证" in md and "推测" in md and "证明了" not in md)
check("5b. 无显著性检验时禁用统计意义上的'显著'",
      "显著" not in md)
md_ext = analyze_figure("tv_map", target="粒子状态", n=12, extrapolation=True,
                        melting_from_model=True)
check("6. 外推 + 模型预测熔融 → 局限性自动提醒",
      "外推" in md_ext and "熔融状态来自模型预测" in md_ext)
md_demo = analyze_figure("parity", target="孔隙率", n=280, demo=True)
check("7. Demo 明确标记不可用于正式科研结论",
      "不可用于正式科研结论" in md_demo and "模拟演示数据" in md_demo)

# 措辞强度：绝对化词汇被替换
md_abs = analyze_figure("pareto", target="Pareto", pareto_count=4)
check("7b. 无绝对化结论（无'绝对最优/完全证明/必然导致'）",
      all(w not in md_abs for w in ["绝对最优", "完全证明", "必然导致"]))

# 8. Analysis.md 导出 + 汇总
p = write_analysis_md(TMP / "Fig_X01_Demo", md)
check("8. Analysis.md 正常导出", p.exists() and p.name == "Fig_X01_Demo_Analysis.md"
      and "## 一、现象描述" in p.read_text(encoding="utf-8"))
p2 = write_analysis_md(TMP / "Fig_X02_Demo", md_demo)
summ = summarize_analyses(TMP)
check("8b. Figure_Analysis_Summary.md 汇总",
      summ.exists() and "Figure Analysis Summary" in summ.read_text(encoding="utf-8")
      and "Fig_X01_Demo" in summ.read_text(encoding="utf-8"))

print()
if failures:
    print(f"FIGURE ANALYSIS TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("FIGURE ANALYSIS TEST PASSED")
