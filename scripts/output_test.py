# -*- coding: utf-8 -*-
"""V1.3 结果输出模块测试

覆盖：论文指标表 / Parity / Residual / 特征重要性 / SHAP / PDP /
Pareto 前沿与表格 / 不确定性 / 元数据 JSON / 摘要 / ZIP 打包。
用法：python scripts/output_test.py
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, load_schema, load_objectives
from src.data_utils import load_table
from src.model_chain import train_chain_full
from src.optimize import random_pareto_search
from src.paper_output import PaperExporter, PaperOutputError

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def is_nonempty(p):
    return p is not None and Path(p).exists() and Path(p).stat().st_size > 100


DEMO = ROOT / "data" / "demo" / "DEMO_双阳极喷涂数据.xlsx"
MODEL = ROOT / "models" / "latest_chain.joblib"

schema = load_schema()
obj_cfg = load_objectives()
df = load_table(DEMO)

print("== 训练（n_splits=3 全链分组 CV）==")
bundle = train_chain_full(df, schema, MODEL, n_splits=3)
check("模型含逐行 OOF 预测记录",
      bundle["chain_cv"].get("oof_predictions", {}).get("stage1") is not None)

# 中文 + PNG+SVG
print("== 中文 / PNG+SVG 导出 ==")
ex = PaperExporter(df, schema, bundle, lang="zh", formats=("png_svg",), demo=True)
run_dir = ex.run_dir

p, df_a1 = ex.table_a1()
check("Table_A1 变量定义表", is_nonempty(p) and len(df_a1) > 30)
out, stem = ex.fig_a1_completeness()
check("Fig_A1 完整率图 PNG+SVG+数据+元数据",
      all(is_nonempty(x) for x in out), f"{len(out)} 个文件")
check("PNG 真实存在", any(str(x).endswith(".png") for x in out))
check("SVG 真实存在", any(str(x).endswith(".svg") for x in out))

out, _ = ex.fig_a2_distributions()
check("Fig_A2 分布图", all(is_nonempty(x) for x in out))
out, _ = ex.fig_a3_correlation()
check("Fig_A3 相关矩阵", all(is_nonempty(x) for x in out))

p, df_b1 = ex.table_b1()
check("Table_B1 模型指标表", is_nonempty(p) and len(df_b1) >= 15)

stage1_t0 = list(bundle["stage1_models"])[0]
out, _ = ex.fig_parity("stage1", stage1_t0)
check("Fig Parity (stage1)", all(is_nonempty(x) for x in out))
csv_parity = [x for x in out if str(x).endswith(".csv")][0]
head = open(csv_parity, encoding="utf-8-sig").readline()
check("Parity 数据含 DEMO 警示行", "DEMO SYNTHETIC DATA" in head)
import pandas as pd
body = pd.read_csv(csv_parity, comment="#")
check("Parity 数据含 experiment_id/batch_id/actual/predicted/residual",
      set(["experiment_id", "batch_id", "actual", "predicted", "residual"]).issubset(body.columns))

out, _ = ex.fig_residual("stage1", stage1_t0)
check("Fig Residual (stage1)", all(is_nonempty(x) for x in out))
out, _ = ex.fig_parity("stage3", "bond_strength_MPa")
check("Fig Parity (stage3, GP)", all(is_nonempty(x) for x in out))
out, _ = ex.fig_b_overview()
check("Fig B 总览 (R² 含负值全部展示)", all(is_nonempty(x) for x in out))

out, _ = ex.fig_c_importance("stage1", stage1_t0, topn=15)
check("Fig C 特征重要性", all(is_nonempty(x) for x in out))
try:
    out, _ = ex.fig_c_shap_summary("stage1", stage1_t0)
    check("Fig C SHAP Summary", all(is_nonempty(x) for x in out))
except PaperOutputError as e:
    print(f"[SKIP] SHAP Summary: {e}")
try:
    out, _ = ex.fig_c_pdp("stage1", stage1_t0, "total_current_A")
    check("Fig C PDP", all(is_nonempty(x) for x in out))
except PaperOutputError as e:
    print(f"[SKIP] PDP: {e}")

out, _ = ex.fig_d_chain()
check("Fig D 全链示意 + Excel", any(str(x).endswith(".xlsx") for x in out))

out = ex.fig_e_pareto(n_candidates=800)
check("Fig E Pareto 前沿", all(is_nonempty(x) for x in out[0]), f"解数 {out[2]['n_pareto']}")
p, df_e1, stats = ex.table_e1(800)
check("Table E1 Pareto 方案表", is_nonempty(p) and len(df_e1) == stats["n_pareto"])
p, df_e2 = ex.table_e2_ranges(800)
check("Table E2 参数范围表", is_nonempty(p) and len(df_e2) == 7)
out, _ = ex.fig_e2_ranges(800)
check("Fig E2 范围图", all(is_nonempty(x) for x in out))

out, _ = ex.fig_f_uncertainty("bond_strength_MPa")
check("Fig F 不确定性 + Table_F1", any(str(x).endswith(".xlsx") for x in out))

p_g = ex.table_g1_template()
check("Table G1 验证模板（仅接口）", is_nonempty(p_g))

p_man = ex.write_reproducibility_manifest()
import json
man = json.load(open(p_man, encoding="utf-8"))
need = ["dataset_version", "dataset_hash", "model_version", "schema_version", "date",
        "python_version", "package_versions", "random_seed", "cv_method",
        "sample_count", "batch_count", "targets", "features"]
check("可复现 manifest 字段齐全", all(k in man for k in need))

meta_files = list((run_dir / "metadata").glob("*Fig_*.json"))
check("论文图同名 metadata JSON", len(meta_files) >= 10, f"{len(meta_files)} 份")
m0 = json.load(open(meta_files[0], encoding="utf-8"))
check("metadata 含 figure_name/model_version/language/data_file",
      all(k in m0 for k in ["figure_name", "model_version", "language", "data_file"]))

p_sum = ex.write_results_summary(pareto_stats=stats, pareto_ranges=df_e2.to_dict("records"))
check("Results_Summary.md", is_nonempty(p_sum))
sum_text = Path(p_sum).read_text(encoding="utf-8")
check("Summary 无因果性表述", all(k not in sum_text for k in ["证明", "显著改善", "必然导致"]))

# PNG DEMO 水印检查：图内右下角文字（通过 matplotlib 重新打开无法查文字，检查文件名前缀）
pngs = [p.name for p in (run_dir / "figures").glob("*.png")]
check("Demo 文件名 DEMO_ 前缀", all(n.startswith("DEMO_") for n in pngs) and len(pngs) > 0,
      f"{len(pngs)} 张")

print("== ZIP 打包 ==")
run_dir2, zip_path, paths = PaperExporter(
    df, schema, bundle, lang="en", formats=("png",), demo=False, run_prefix="Paper_Output"
).build_bundle(n_candidates=600)
check("ZIP 文件生成", zip_path.exists() and zip_path.stat().st_size > 100000,
      f"{zip_path.name} {zip_path.stat().st_size // 1024} KB")
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    check("ZIP 含 figures/", any("figures/" in n and n.endswith(".png") for n in names))
    check("ZIP 含 tables/", any("tables/" in n and n.endswith(".xlsx") for n in names))
    check("ZIP 含 data/", any("data/" in n and n.endswith(".csv") for n in names))
    check("ZIP 含 metadata/", any("reproducibility_manifest.json" in n for n in names))
en_figs = list((run_dir2 / "figures").glob("*.png"))
check("English 论文图生成（无 DEMO 前缀）", len(en_figs) > 0 and
      all(not n.name.startswith("DEMO_") for n in en_figs), f"{len(en_figs)} 张")

print()
if failures:
    print(f"OUTPUT TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("OUTPUT TEST PASSED")
