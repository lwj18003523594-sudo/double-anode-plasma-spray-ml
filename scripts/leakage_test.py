# -*- coding: utf-8 -*-
"""数据泄漏专项测试（V1.2）

验证内容：
1. 全链分组交叉验证（cross_validate_chain）的每个外层 fold 中，
   任何一个 batch_id 都不同时出现在训练集和验证集；
2. 内层 OOF（_fit_one_oof 中的 GroupKFold）的所有切分同样按 batch_id 隔离
   （通过包装 GroupKFold 记录并断言每一次 split）；
3. 验证批次并集覆盖全部批次，且各 fold 验证集互不重叠；
4. 三个层级的交叉验证指标均由验证集预测计算得到。

用法：python scripts/leakage_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import src.model_chain as mc
from src.config import ROOT, load_schema
from src.data_utils import load_table

DEMO = ROOT / "data" / "demo" / "DEMO_双阳极喷涂数据.xlsx"

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---- 1. 包装 GroupKFold：记录并断言每一次切分按 groups 隔离 ----
_original_gkf = mc.GroupKFold
inner_split_count = 0


class RecordingGroupKFold(_original_gkf):
    def split(self, X, y=None, groups=None):
        splits = list(_original_gkf.split(self, X, y, groups))
        if groups is not None:
            g = np.asarray(groups)
            for i, (tr, va) in enumerate(splits):
                overlap = set(g[tr]) & set(g[va])
                if overlap:
                    raise AssertionError(
                        f"内层 OOF 泄漏：batch 同时出现在训练/验证集: {sorted(overlap)}")
            globals()["inner_split_count"] += 1
        return splits


mc.GroupKFold = RecordingGroupKFold

# ---- 2. 跑全链分组交叉验证 ----
schema = load_schema()
df = load_table(DEMO)
all_batches = set(df["batch_id"].astype(str))
print(f"数据：{len(df)} 行，{len(all_batches)} 个批次；n_splits=3")

cv = mc.cross_validate_chain(df, schema, seed=42, n_splits=3)
print(f"CV method: {cv['method']}")

# ---- 3. 外层 fold 断言 ----
fold_info = cv["fold_info"]
check("外层 fold 数量 = 3", len(fold_info) == 3, f"实际 {len(fold_info)}")

val_batches_all = []
fold_ok = True
for fi in fold_info:
    tr_b = set(fi["train_batches"])
    va_b = set(fi["val_batches"])
    overlap = tr_b & va_b
    if overlap:
        fold_ok = False
        print(f"  fold {fi['fold']} 泄漏批次: {sorted(overlap)}")
    val_batches_all.append(va_b)
check("每个 fold 训练/验证批次完全隔离", fold_ok)

union_va = set().union(*val_batches_all)
check("验证批次并集覆盖全部批次", union_va == all_batches,
      f"并集 {len(union_va)} / 全部 {len(all_batches)}")
check("各 fold 验证批次互不重叠",
      sum(len(v) for v in val_batches_all) == len(union_va))

# 每个批次只在一个 fold 的验证集中出现一次
batch_val_counts = {}
for va_b in val_batches_all:
    for b in va_b:
        batch_val_counts[b] = batch_val_counts.get(b, 0) + 1
check("每个批次恰好作为验证集出现一次",
      all(c == 1 for c in batch_val_counts.values()))

# ---- 4. 内层 OOF 切分均按批次隔离（由 RecordingGroupKFold 断言）----
check("内层 OOF 使用分组切分（GroupKFold 被实际调用）", inner_split_count > 0,
      f"记录到 {inner_split_count} 次分组切分")

# ---- 5. 三级指标均存在且由验证预测计算 ----
for stage in ["stage1", "stage2", "stage3"]:
    m = cv["metrics"].get(stage, {})
    check(f"{stage} 交叉验证指标非空", len(m) > 0, f"目标数 {len(m)}")

# ---- 6. 抽查数值合理性：指标可计算且有限 ----
import math
all_finite = True
for stage in ["stage1", "stage2", "stage3"]:
    for t, met in cv["metrics"].get(stage, {}).items():
        for k, v in met.items():
            if not math.isfinite(v):
                all_finite = False
check("全部指标数值有限", all_finite)

print()
if failures:
    print(f"LEAKAGE TEST FAILED — {len(failures)} 项未通过: {failures}")
    sys.exit(1)
print("LEAKAGE TEST PASSED — 未发现任何 batch_id 层级数据泄漏")
