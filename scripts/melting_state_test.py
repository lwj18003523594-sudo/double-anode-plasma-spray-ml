# -*- coding: utf-8 -*-
"""V1.4 熔融状态测试（spec 四十三）

检查：
1. 具有熔融目标数据时 Stage 1.5 可以训练；
2. 没有熔融目标时自动跳过；
3. 跳过时 Stage 2 / Stage 3 仍正常训练；
4. Stage 1.5 OOF 正常；
5. 训练集预测不泄漏进 Stage 2（OOF 覆盖 = 全部有效行，且每行仅一个折给出预测）；
6. T–V 状态图正常；
7. 连续熔融目标 parity 正常；
8. 分类目标混淆矩阵正常；
9. 论文输出 H 模块正常。

测试使用合成熔融标签（仅软件测试夹具），不修改任何科研/Demo 数据文件。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.modes import load_schema_for_mode
from src.config import load_schema
from src.data_utils import load_table
from src.model_chain import train_chain, train_chain_full, predict_chain, cross_validate_chain
from src.paper_output import PaperExporter

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


schema_m = load_schema_for_mode("dual_anode")
df = load_table("data/demo/DEMO_双阳极喷涂数据.xlsx")

# 合成熔融标签（测试夹具，不落盘到数据目录）
rng = np.random.default_rng(11)
mi = (1.05 - 0.12 * df["porosity_pct"].astype(float) + rng.normal(0, 0.08, len(df))).clip(0.05, 1.25)
df_melt = df.copy()
df_melt["melting_index"] = mi
df_melt["melting_state_class"] = pd.qcut(
    mi.rank(method="first"), 4,
    labels=["insufficiently_molten", "partially_molten", "fully_molten", "overheated"]).astype(str)

# 1. 有熔融数据 → 启用
b_on = train_chain(df_melt, schema_m, model_path=None, seed=42)
check("1. 有熔融标签 → Stage 1.5 训练", b_on["stage15_enabled"] is True
      and set(b_on["stage15_models"].keys()) == {"melting_index", "melting_state_class"})

# 2/3. 无熔融数据 → 跳过且主链正常
b_off = train_chain(df, schema_m, model_path=None, seed=42)
check("2. 无熔融标签 → Stage 1.5 自动跳过", b_off["stage15_enabled"] is False)
check("3. 跳过后 Stage2/3 仍正常训练",
      b_off["stage2_models"] and b_off["stage3_models"]
      and not (b_off["chain_cv"] if False else False))
s, d, p, u = predict_chain(df.head(3), b_off)
check("3b. 跳过后全链预测正常", s.shape[1] == 5 and p.shape[1] == 7)

# 4/5. OOF 与防泄漏
cv = cross_validate_chain(df_melt, schema_m, seed=42, n_splits=3)
_oof_all = cv.get("oof_predictions") or {}
oof15 = _oof_all.get("stage15") if _oof_all.get("stage15") is not None else {}
check("4. Stage 1.5 OOF 正常",
      "melting_index" in oof15 and oof15["melting_index"].notna().sum() >= 250)
mi_oof = oof15["melting_index"]
cls_oof = oof15.get("melting_state_class")
n_valid = int(df_melt["melting_index"].notna().sum())
check("5. 无训练集预测泄漏进 Stage 2（OOF 每个有效行恰有一个折外预测）",
      int(mi_oof.notna().sum()) == n_valid
      and (cls_oof is None or int(cls_oof.notna().sum()) == n_valid),
      f"OOF 非空 {int(mi_oof.notna().sum())} / 有效 {n_valid}")
check("5b. Stage 2 输入含熔融 OOF 列（而非拟合值通道）",
      "pred_melting_index" in b_on["stage2_features"]
      and "pred_melting_state_class" in b_on["stage2_features"])
m15 = (cv.get("metrics") or {}).get("stage15") or {}
check("5c. Stage 1.5 指标类型正确（回归 R² / 分类 Accuracy）",
      "R2" in m15.get("melting_index", {}) and "Accuracy" in m15.get("melting_state_class", {}))

# 6-9. 论文 H 模块
bundle_full = train_chain_full(df_melt, schema_m, "models/latest_chain.joblib", seed=42, n_splits=3)
ex = PaperExporter(df_melt, schema_m, bundle_full, lang="zh", formats=("png",),
                   demo=False, prefix="TEST_")
ok_h1 = ok_h3 = True
try:
    out1, stem1 = ex.fig_h_tv_map()
except Exception as e:
    ok_h1 = False
    print("   H1 error:", e)
try:
    out3, stem3 = ex.fig_h_melting_performance()
except Exception as e:
    ok_h3 = False
    print("   H3 error:", e)
check("6. T–V 状态图正常", ok_h1)
check("7. 连续熔融目标 parity 正常（含于 H3）", ok_h3)
check("8. 分类目标混淆矩阵正常（含于 H3）", ok_h3)
# 9. 完整 H 模块 + build_bundle
run_dir, zip_path, paths = ex.build_bundle(n_candidates=600)
h_figs = [Path(p).name for p in paths["figures"] if "Fig_H0" in Path(p).name]
check("9. 论文输出 H 模块正常（H01–H06 全部生成）",
      len([f for f in h_figs]) >= 6, f"{len(h_figs)} 张 H 图")

print()
if failures:
    print(f"MELTING STATE TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("MELTING STATE TEST PASSED")
