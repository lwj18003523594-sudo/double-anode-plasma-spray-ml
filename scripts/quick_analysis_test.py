# -*- coding: utf-8 -*-
"""V1.5 快速科研分析测试（spec 七十四）

夹具：标准平台 Excel / 多 Sheet 文献数据库 / 无 metadata Excel。
检查：自动识别、歧义确认、独立运行、不错误合并、不覆盖正式模型、结果输出正常。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_utils import load_table
from src import quick_analysis as qa

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


TMP = Path(__file__).resolve().parents[1] / "outputs" / "v15_test_tmp"
TMP.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(77)


def synth(n=70, batch=8, prefix="E", melt=True):
    df = pd.DataFrame({
        "experiment_id": [f"{prefix}{i:03d}" for i in range(n)],
        "batch_id": [f"B{i % batch:02d}" for i in range(n)],
        "anode1_current_A": rng.uniform(300, 600, n),
        "anode2_current_A": rng.uniform(300, 600, n),
        "Ar_flow_slpm": rng.uniform(30, 70, n),
        "H2_flow_slpm": rng.uniform(2, 12, n),
        "powder_feed_g_min": rng.uniform(15, 45, n),
        "spray_distance_mm": rng.uniform(90, 140, n),
        "traverse_speed_mm_s": rng.uniform(150, 450, n),
        "anode_spacing_mm": rng.uniform(4, 8, n),
        "particle_temperature_C": rng.uniform(2000, 2600, n),
        "particle_velocity_m_s": rng.uniform(350, 520, n),
        "porosity_pct": rng.uniform(0.8, 5, n),
        "bond_strength_MPa": rng.uniform(30, 75, n),
    })
    if melt:
        mi = rng.uniform(0.3, 1.1, n)
        df["melting_index"] = mi
        df["melting_state_class"] = pd.qcut(pd.Series(mi).rank(method="first"), 3,
                                            labels=["insufficiently_molten",
                                                    "partially_molten",
                                                    "fully_molten"]).astype(str)
    return df


# ---- 1. 标准双阳极 Excel（含熔融列）→ 自动识别 + 独立运行 ----
std_xlsx = TMP / "Standard_DualAnode.xlsx"
synth().to_excel(std_xlsx, sheet_name="Data", index=False)
_mode, _scores = qa.detect_mode(pd.read_excel(std_xlsx))
check("1. 标准平台 Excel 自动识别为双阳极模式", _mode == "dual_anode",
      f"scores={ {k: round(v, 2) for k, v in _scores.items()} }")

# 正式模型快照（验证不覆盖）
formal = Path("models") / "dual_anode" / "latest_chain.joblib"
formal_root = Path("models") / "latest_chain.joblib"
_before = formal.read_bytes() if formal.exists() else None
_before_root = formal_root.read_bytes() if formal_root.exists() else None

res = qa.quick_analyze_file(std_xlsx, run_prefix="QT_STD", demo=True)
r1 = res[0]
check("1b. 一键运行完成（分步状态含 done）",
      not r1["record"].get("failed") and any(s == "done" for _, s, _r in r1["steps"]),
      "；".join(f"{n}:{s}" + (f"（{_r}）" if _r and s in ("skip", "fail") else "")
                for n, s, _r in r1["steps"]))
check("1c. 熔融标签存在 → Stage 1.5 启用", r1["record"].get("stage15") is True)
check("1d. Quick Run 模型保存在 runs/quick_analysis（不覆盖正式模型）",
      (qa.QUICK_ROOT / r1["run_id"] / "model.joblib").exists()
      and (formal.read_bytes() if formal.exists() else None) == _before
      and (formal_root.read_bytes() if formal_root.exists() else None) == _before_root)
pkg = r1.get("package") or {}
check("1e. 科研结果输出正常（图表 + 洞察 + 解读）",
      pkg.get("figures", 0) > 0
      and (qa.QUICK_ROOT / r1["run_id"] / "results" / "insights" / "Insight_Report.md").exists()
      and Path(pkg.get("analysis_dir", "x")).exists()
      and any(p.name.endswith("_Analysis.md")
              for p in Path(pkg.get("analysis_dir", "x")).glob("*_Analysis.md")))

# ---- 2. 多 Sheet 文献数据库：Dataset_01 / Dataset_02 独立运行、不合并 ----
lit_xlsx = TMP / "Literature_Test_Database_PlasmaSpray_ML.xlsx"
d1, d2 = synth(60, 6, "L1"), synth(55, 5, "L2", melt=False)
meta = pd.DataFrame({"字段": ["arc_current_A"], "方向": ["prediction_only"],
                     "说明": ["测试元数据"]})
with pd.ExcelWriter(lit_xlsx, engine="openpyxl") as w:
    meta.to_excel(w, sheet_name="README_使用说明", index=False)
    d1.to_excel(w, sheet_name="Dataset_01", index=False)
    d2.to_excel(w, sheet_name="Dataset_02", index=False)
data_sheets, meta_sheets = qa.classify_sheets(pd.ExcelFile(lit_xlsx).sheet_names)
check("2. 多 Sheet 分类：Dataset_01/02 为数据表，README 为元数据",
      data_sheets == ["Dataset_01", "Dataset_02"] and "README_使用说明" in meta_sheets)
res2 = qa.quick_analyze_file(lit_xlsx, run_prefix="QT_LIT", demo=False)
check("2b. 两个 Dataset 各自独立 Run（不合并）",
      len(res2) == 2 and res2[0]["record"]["sheet"] == "Dataset_01"
      and res2[1]["record"]["sheet"] == "Dataset_02"
      and res2[0]["run_id"] != res2[1]["run_id"])
check("2c. Dataset_01（含熔融）Stage 1.5 启用 / Dataset_02（无熔融）未启用",
      res2[0]["record"].get("stage15") is True and res2[1]["record"].get("stage15") is False)
skip_txt = [n for n, s, _r in res2[1]["steps"] if s == "skip"]
check("2d. 无熔融数据 → Stage 1.5 显示自动跳过", any("熔融" in n for n in skip_txt))

# ---- 3. 无 metadata Excel（未定义优化方向）→ 跳过 Pareto ----
plain_xlsx = TMP / "Plain_NoMeta.xlsx"
synth(50, 5, "P", melt=False).to_excel(plain_xlsx, sheet_name="Data", index=False)
dirs = qa.read_objective_directions({"Data": pd.read_excel(plain_xlsx)})
check("3. 无 metadata → 无优化方向（prediction_only）", not dirs)
res3 = qa.quick_analyze_file(plain_xlsx, run_prefix="QT_PLAIN", demo=False)
r3 = res3[0]
skip3 = [(n, rsn) for n, s, rsn in r3["steps"] if s == "skip"]
check("3b. 未定义方向 → Pareto 自动跳过并提示（不擅自认定优化方向）",
      any("Pareto" in n for n, _ in skip3)
      and any("未定义优化目标方向" in (rsn or "") for _, rsn in skip3))

# ---- 4. Run History + 设为正式模型 ----
hist = qa.latest_runs(10)
check("4. Run History 记录最近运行（含 run_id/模式/验证方式）",
      len(hist) >= 3 and all({"run_id", "research_mode", "validation_method"} <= set(h)
                             for h in hist[:3]))
dst = qa.promote_to_formal(r1["run_id"], "dual_anode")
check("4b. 设为正式模型：需显式调用才复制（目标路径正确且存在）",
      dst.exists() and dst == Path(__file__).resolve().parents[1] / "models" / "dual_anode" / "latest_chain.joblib")

# ---- 测试清理：恢复测试前状态，不污染正式模型与运行历史 ----
import json as _json
restored = True
if _before is not None:
    formal.write_bytes(_before)  # 恢复原正式模型
    restored = formal.read_bytes() == _before
else:
    if formal.exists():
        formal.unlink()
hist_data = [h for h in qa.load_history() if not h["run_id"].startswith("QT_")]
qa.HISTORY_FILE.write_text(_json.dumps(hist_data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
check("4c. 测试清理：正式模型与运行历史恢复原状", restored and not formal.exists() if _before is None else restored)

print()
if failures:
    print(f"QUICK ANALYSIS TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("QUICK ANALYSIS TEST PASSED")
