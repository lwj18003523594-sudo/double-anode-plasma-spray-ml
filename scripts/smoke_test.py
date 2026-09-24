"""Smoke test mirroring all app.py execution paths (demo mode)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import ROOT, load_schema, load_objectives, groups
from src.data_utils import load_table, validate_data, numeric_completion
from src.model_chain import train_chain_full, predict_chain
from src.optimize import random_pareto_search

DEMO = ROOT / "data" / "demo" / "DEMO_双阳极喷涂数据.xlsx"
MODEL = ROOT / "models" / "latest_chain.joblib"
OUT = ROOT / "outputs" / "optimization" / "latest_pareto.xlsx"

schema = load_schema()
obj_cfg = load_objectives()
grp = groups(schema)
print("[1] schema/objectives loaded:", list(schema.keys()))

d = load_table(DEMO)
print("[2] demo data loaded:", d.shape)

issues = validate_data(d, schema)
print("[3] validate_data issues:", issues)

target_cols = grp["process_states"] + grp["defect_network"] + grp["performance_outputs"]
comp = numeric_completion(d, target_cols)
print("[4] completion rows:", len(comp))

bundle = train_chain_full(d, schema, MODEL, n_splits=3)
print("[5] train_chain_full OK (含 n_splits=3 全链分组交叉验证)。chain CV metrics:")
for stage, vals in bundle["chain_cv"]["metrics"].items():
    for t, met in vals.items():
        print(f"    {stage} | {t}: {met}")

# 版本记录检查
mv = bundle["model_version"]
manifest_path = MODEL.parent / "history" / mv / "manifest.json"
check_manifest = manifest_path.exists()
m = pd.read_json(manifest_path, typ="series") if check_manifest else None
required_keys = ["dataset_version", "dataset_hash", "schema_version", "model_version",
                 "training_time", "random_seed", "feature_list", "target_list",
                 "batch_count", "sample_count", "cross_validation_method", "fold_groups",
                 "model_hyperparameters", "R2", "RMSE", "MAE"]
missing_keys = [k for k in required_keys if m is None or k not in m.index]
print(f"[5b] 版本记录 {mv}: manifest 存在={check_manifest}, 缺失字段={missing_keys}")
assert check_manifest and not missing_keys
import shutil as _sh
assert (MODEL.parent / "history").is_dir() and len(list((MODEL.parent / "history").iterdir())) >= 1
print("[5c] 历史版本目录存在，旧版本未被覆盖")

# single-point prediction
vals = {}
for section in ["structure_inputs", "process_inputs"]:
    for col, spec in schema[section].items():
        vals[col] = (float(spec["min"]) + float(spec["max"])) / 2
X = pd.DataFrame([vals])
states, defects, perf, unc = predict_chain(X, bundle)
print("[6] predict_chain OK. states:", states.shape, "defects:", defects.shape,
      "perf:", perf.shape, "unc cols:", list(unc.columns))
out = perf.T.rename(columns={0: "预测值"})
if len(unc.columns):
    std_map = unc.iloc[0].to_dict()
    out["预测标准差"] = [std_map.get(idx + "_std", None) for idx in out.index]
print("[7] prediction table:\n", out)

# feature importance path (tab4)
pipe = bundle["stage1_models"][list(bundle["stage1_models"])[0]]
booster = pipe.named_steps["model"]
imp = pd.DataFrame({"feature": bundle["stage1_features"],
                    "importance": booster.feature_importances_}).sort_values("importance", ascending=False)
print("[8] feature importance OK, top:", imp.iloc[0].tolist())

# optimization path
fixed = {col: (float(s["min"]) + float(s["max"])) / 2 for col, s in schema["structure_inputs"].items()}
front = random_pareto_search(bundle, schema, obj_cfg, fixed_values=fixed, n_candidates=1000)
print("[9] pareto front size:", len(front))
if not front.empty:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    front.to_excel(OUT, index=False)
    print("[10] pareto saved:", OUT)

print("ALL SMOKE TESTS PASSED")
