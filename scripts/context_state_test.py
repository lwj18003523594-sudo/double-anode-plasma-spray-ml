# -*- coding: utf-8 -*-
"""V1.5 CurrentContext 状态测试（spec 七十三）

依次切换六种 模式×数据来源 组合，检查状态字段一致、无跨模式串用：
双阳极 Demo / 双阳极文献 / 双阳极真实 / 传统 APS / 多电极级联 / 通用文献。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.context import build_context, context_status_items, context_detail_lines
from src.data_utils import load_table
from src.model_chain import train_chain
from src.modes import load_schema_for_mode

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


DEMO = load_table("data/demo/DEMO_双阳极喷涂数据.xlsx")
# 轻量 bundle（含 CV 方式与模型版本字段；context 只读这些键，不触发训练）
bundle_d = {"model_version": "v20260922_000000",
            "chain_cv": {"method": "GroupKFold(n_splits=3) by batch_id, full-chain",
                         "metrics": {}, "fold_info": []},
            "stage15_enabled": False,
            "dataset_hash": "0" * 64}

cases = [
    ("dual_anode", "模拟演示数据", DEMO, bundle_d, True),
    ("dual_anode", "文献数据", None, None, False),
    ("dual_anode", "真实实验数据", None, None, False),
    ("conventional_aps", "文献数据", None, None, False),
    ("cascade", "文献数据", None, None, False),
    ("literature", "文献数据（固定）", None, None, False),
]

expected = {
    "dual_anode": "双阳极等离子喷涂",
    "conventional_aps": "传统大气等离子喷涂（APS）",
    "cascade": "多电极 / 级联等离子喷涂",
    "literature": "通用文献数据测试",
}

for mode_id, source, df, bundle, trained in cases:
    ctx = build_context(mode_id, source, df=df, bundle=bundle, model_file_exists=False)
    tag = f"{expected[mode_id]}·{source}"
    ok = (ctx["research_mode"] == mode_id
          and ctx["data_source"] == source
          and ctx["dataset_loaded"] == (df is not None)
          and ctx["model_trained"] == trained
          and expected[mode_id] in ctx["research_mode_label"])
    check(f"状态一致且无串用：{tag}", ok,
          f"loaded={ctx['dataset_loaded']} trained={ctx['model_trained']}")

# 1. 双阳极 Demo 完整状态
ctx1 = build_context("dual_anode", "模拟演示数据", df=DEMO, bundle=bundle_d)
check("1. Demo：记录数/批次/验证方式/Stage1.5 字段齐全",
      ctx1["record_count"] == len(DEMO)
      and ctx1["batch_count"] == DEMO["batch_id"].nunique()
      and ctx1["validation_method"] and "GroupKFold" in ctx1["validation_method"]
      and ctx1["stage15_enabled"] is False
      and ctx1["dataset_version"].startswith("auto_"))
items = context_status_items(ctx1)
check("1b. 状态栏 7 项（研究模式/数据来源/数据状态/记录数/批次/模型状态/验证方式）",
      len(items) == 7 and "研究模式" in items[0] and "验证方式" in items[6]
      and "已加载" in items[2] and "已训练" in items[5])
check("1c. 普通状态栏不显示长 hash/完整版本号",
      not any("auto_" in i for i in items) and not any("v2026" in i for i in items))
det = context_detail_lines(ctx1)
check("1d. 【详细信息】含 dataset_hash 与模型版本",
      any("dataset_hash" in d for d in det)
      and any("模型版本" in d and "v2026" in d for d in det))

# 2. 无数据模式：等待数据 / 未加载（spec 三十二）
ctx2 = build_context("conventional_aps", "文献数据", df=None, bundle=None)
items2 = context_status_items(ctx2)
check("2. 无数据状态：未加载 / 记录数— / 等待数据 / 验证方式—",
      "未加载" in items2[2] and "—" in items2[3] and "等待数据" in items2[5]
      and "—" in items2[6])
check("2b. 双阳极已训练不影响 APS 状态（不跨模式继承）",
      ctx2["model_trained"] is False and ctx2["dataset_loaded"] is False
      and ctx2["model_version"] is None)

# 3. 模式标签互不相同（无串扰）
labels = [build_context(m, "文献数据")["research_mode_label"] for m in expected]
check("3. 四模式标签互不相同", len(set(labels)) == 4, str(labels))

# 4. sample_id 支持
df_s = DEMO.copy()
df_s["sample_id"] = [f"S{i//4}" for i in range(len(df_s))]
ctx4 = build_context("dual_anode", "真实实验数据", df=df_s, bundle=None)
check("4. sample_id 存在时统计独立试样数", ctx4["sample_count"] == len(df_s) // 4)

print()
if failures:
    print(f"CONTEXT STATE TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("CONTEXT STATE TEST PASSED")
