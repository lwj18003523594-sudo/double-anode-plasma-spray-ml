from pathlib import Path
import pandas as pd
import numpy as np

def load_table(file_or_path):
    name = str(getattr(file_or_path, "name", file_or_path)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_or_path)
    if name.endswith((".xlsx",".xls")):
        try:
            return pd.read_excel(file_or_path, sheet_name="Experiments")
        except ValueError:
            # 用户上传的表格没有 "Experiments" 工作表时，退回第一个工作表
            return pd.read_excel(file_or_path, sheet_name=0)
    raise ValueError("仅支持 CSV / XLSX / XLS")

def validate_data(df, schema):
    issues = []
    all_required = []
    for section in ["meta","structure_inputs","process_inputs"]:
        for col, spec in schema.get(section, {}).items():
            if spec.get("required", True):
                all_required.append(col)
    missing_cols = [c for c in all_required if c not in df.columns]
    if missing_cols:
        issues.append(f"缺少必需列: {missing_cols}")

    if "experiment_id" in df.columns and df["experiment_id"].duplicated().any():
        issues.append("experiment_id 存在重复。")

    for section in ["structure_inputs","process_inputs"]:
        for col,spec in schema.get(section,{}).items():
            if col not in df.columns: 
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            if "min" in spec and (vals.dropna() < spec["min"]).any():
                issues.append(f"{col} 存在低于配置下限 {spec['min']} 的值。")
            if "max" in spec and (vals.dropna() > spec["max"]).any():
                issues.append(f"{col} 存在高于配置上限 {spec['max']} 的值。")
    return issues

def numeric_completion(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        out.append({"column":c, "n":len(s), "valid":int(s.notna().sum()),
                    "missing":int(s.isna().sum()), "missing_pct":float(s.isna().mean()*100)})
    return pd.DataFrame(out)


def compute_default_inputs(schema, df=None):
    """V1.3.2 「恢复默认参数」统一取值逻辑（纯函数，供界面与测试共用）。

    优先级（逐列独立判断）：
      1) 当前训练数据中位数（截断到设备允许范围）——存在有效数据时优先；
      2) config 中显式 default 值（若配置）——仅当该列无有效数据时使用；
      3) 设备允许范围中点——最后兜底。

    目的：默认值必须落在当前训练数据支持域内，避免"默认即外推"
    （如双阳极间距默认 10.5 mm 而训练数据仅 4–8 mm 的情形）。
    只计算界面默认值，不修改数据与任何模型逻辑。
    """
    inputs = {**schema.get("structure_inputs", {}), **schema.get("process_inputs", {})}
    medians = {}
    if df is not None:
        for col in inputs:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    medians[col] = float(s.median())
    defaults = {}
    for col, spec in inputs.items():
        if spec.get("categorical") or "min" not in spec or "max" not in spec:
            continue  # 分类/无范围列不参与数值默认值（如 torch_type、material_name）
        lo, hi = float(spec["min"]), float(spec["max"])
        if col in medians:
            defaults[col] = float(np.clip(medians[col], lo, hi))
        elif spec.get("default") is not None:
            defaults[col] = float(np.clip(float(spec["default"]), lo, hi))
        else:
            defaults[col] = (lo + hi) / 2.0
    return defaults
