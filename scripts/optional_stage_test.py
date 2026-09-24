# -*- coding: utf-8 -*-
"""V1.4 可选层测试（spec 四十四）

分别验证两种链路均可运行：
  A. Stage1 → Stage1.5 → Stage2 → Stage3（有熔融标签）
  B. Stage1 → Stage2 → Stage3（无熔融标签，V1.3 原链路）

并验证：B 链路的预测结果与"原始 Schema（未合并可选层）"训练的模型逐值一致
（即 V1.4 代码对无熔融数据是严格的 no-op）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.modes import load_schema_for_mode
from src.config import load_schema
from src.data_utils import load_table
from src.model_chain import train_chain, predict_chain

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


df = load_table("data/demo/DEMO_双阳极喷涂数据.xlsx")
schema_m = load_schema_for_mode("dual_anode")

# ---- A 链路：含熔融层 ----
rng = np.random.default_rng(21)
mi = (1.05 - 0.12 * df["porosity_pct"].astype(float) + rng.normal(0, 0.08, len(df))).clip(0.05, 1.25)
df_a = df.copy()
df_a["melting_index"] = mi
df_a["melting_state_class"] = pd.qcut(mi.rank(method="first"), 4,
                                      labels=["insufficiently_molten", "partially_molten",
                                              "fully_molten", "overheated"]).astype(str)
b_a = train_chain(df_a, schema_m, model_path=None, seed=42)
X_in = df_a.head(4)
sA, dA, pA, uA, mA = predict_chain(X_in, b_a, return_melting=True)
check("A. Stage1→1.5→2→3 链路可训练可预测",
      b_a["stage15_enabled"] and sA.shape == (4, 5) and dA.shape == (4, 5) and pA.shape == (4, 7)
      and "melting_index" in mA["continuous"].columns)

# ---- B 链路：无熔融层 ----
b_b = train_chain(df, schema_m, model_path=None, seed=42)
b_ref = train_chain(df, load_schema(), model_path=None, seed=42)  # 原始 Schema（V1.3 路径）
sB, dB, pB, uB = predict_chain(df.head(4), b_b)
sR, dR, pR, uR = predict_chain(df.head(4), b_ref)
same = (np.allclose(sB.values, sR.values, atol=0, rtol=0)
        and np.allclose(dB.values, dR.values, atol=0, rtol=0)
        and np.allclose(pB.values, pR.values, atol=0, rtol=0))
check("B. Stage1→2→3 链路可运行", b_b["stage15_enabled"] is False
      and sB.shape == (4, 5) and pB.shape == (4, 7))
check("B2. 无熔融数据时 V1.4 代码为严格 no-op（与原始 Schema 模型预测逐位一致）",
      same and np.allclose(uB.values, uR.values, atol=0, rtol=0))

print()
if failures:
    print(f"OPTIONAL STAGE TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("OPTIONAL STAGE TEST PASSED")
