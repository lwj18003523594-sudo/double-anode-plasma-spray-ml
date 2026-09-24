import numpy as np
import pandas as pd
from .model_chain import predict_chain
from .research_utils import pareto_row_support, support_label

def _dominates(a, b, directions):
    better_or_equal = True
    strictly_better = False
    for i,d in enumerate(directions):
        if d == "minimize":
            if a[i] > b[i]: better_or_equal = False
            if a[i] < b[i]: strictly_better = True
        else:
            if a[i] < b[i]: better_or_equal = False
            if a[i] > b[i]: strictly_better = True
    return better_or_equal and strictly_better

def pareto_mask(values, directions):
    n = len(values)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if _dominates(values[j], values[i], directions):
                keep[i] = False
                break
    return keep

def random_pareto_search(bundle, schema, objective_cfg, fixed_values=None, n_candidates=2500, seed=42,
                         return_stats=False, return_all=False, search_domain="training"):
    """多目标随机 Pareto 搜索（V1.3.2 增加优化搜索域选择，不改变模型与预测逻辑）。

    search_domain:
    - "training"（默认）：可优化变量只在当前训练数据支持域（bundle["training_domain"] 的
      min/max）内采样，不产生训练数据范围之外的候选点；训练域缺失的特征回退设备允许范围。
    - "device"：在设备允许可行域内采样（探索性，存在外推风险）；同时为每个候选点计算
      extrapolation_risk 标记列（“存在外推风险”/“无外推”），越界方案必须明确标记。

    fixed_values（如喷枪结构参数）为用户显式指定值，不参与采样、不被修改；
    其相对训练域的支持状态由界面“数据支持状态”列单独标注。
    """
    rng = np.random.default_rng(seed)
    fixed_values = fixed_values or {}
    domain = bundle.get("training_domain") or {}
    rows = {}
    for section in ["structure_inputs","process_inputs"]:
        for col,spec in schema[section].items():
            if col in fixed_values:
                rows[col] = np.repeat(float(fixed_values[col]), n_candidates)
            elif spec.get("optimizable", False):
                if search_domain == "training" and col in domain:
                    lo, hi = float(domain[col]["min"]), float(domain[col]["max"])
                    if hi < lo:
                        lo, hi = float(spec["min"]), float(spec["max"])
                else:
                    lo, hi = float(spec["min"]), float(spec["max"])
                rows[col] = rng.uniform(lo, hi, n_candidates)
            else:
                # midpoint for fixed structural factors not specified
                rows[col] = np.repeat((float(spec["min"])+float(spec["max"]))/2, n_candidates)

    X = pd.DataFrame(rows)
    if bundle.get("stage15_enabled"):
        states, defects, perf, unc, melt = predict_chain(X, bundle, return_uncertainty=True,
                                                         return_melting=True)
    else:
        states, defects, perf, unc = predict_chain(X, bundle, return_uncertainty=True)
        melt = None
    parts = [X, states.add_prefix("pred_"), defects.add_prefix("pred_"),
             perf.add_prefix("pred_"), unc]
    if melt is not None:
        # V1.4：熔融层预测进入候选表，使可选熔融状态约束（spec 27）可被过滤
        mp = pd.DataFrame(index=X.index)
        for t in melt["continuous"].columns:
            mp[f"pred_{t}"] = melt["continuous"][t].to_numpy()
        for t in melt["class"].columns:
            mp[f"pred_{t}"] = melt["class"][t].to_numpy()
        parts.append(mp)
    result = pd.concat(parts, axis=1)

    # V1.3.2：设备允许域（探索性）模式下，逐候选点标记外推风险
    if search_domain == "device" and domain:
        risk_labels = []
        for i in range(len(X)):
            row_params = {c: X.iloc[i][c] for c in domain if c in X.columns}
            status = pareto_row_support(row_params, domain)
            risk_labels.append("存在外推风险" if status == "extrap" else "无外推")
        result["extrapolation_risk"] = risk_labels

    # constraints
    feasible = pd.Series(True, index=result.index)
    for col,spec in objective_cfg.get("constraints",{}).items():
        pred_col = "pred_"+col if "pred_"+col in result.columns else col
        if pred_col not in result.columns:
            continue
        if "min" in spec:
            feasible &= result[pred_col] >= float(spec["min"])
        if "max" in spec:
            feasible &= result[pred_col] <= float(spec["max"])
    n_feasible = int(feasible.sum())
    result = result.loc[feasible].reset_index(drop=True)
    if result.empty:
        if return_all:
            return result, {"n_candidates": int(n_candidates), "n_feasible": n_feasible, "n_pareto": 0}, result
        if return_stats:
            return result, {"n_candidates": int(n_candidates), "n_feasible": n_feasible, "n_pareto": 0}
        return result

    obj_names_all = list(objective_cfg["objectives"].keys())
    # V1.4 防御：文献/裁剪链路下某些目标可能不存在（无对应模型），跳过缺失目标
    obj_names = [k for k in obj_names_all
                 if (f"pred_{k}" in result.columns) or (k in result.columns)]
    if not obj_names:
        empty_stats = {"n_candidates": int(n_candidates), "n_feasible": n_feasible,
                       "n_pareto": 0, "search_domain": search_domain}
        if return_all:
            return result.iloc[0:0], empty_stats, result
        if return_stats:
            return result.iloc[0:0], empty_stats
        return result.iloc[0:0]
    dirs = [objective_cfg["objectives"][k] for k in obj_names]
    cols = [("pred_"+k if "pred_"+k in result.columns else k) for k in obj_names]
    values = result[cols].to_numpy(float)
    mask = pareto_mask(values, dirs)
    front = result.loc[mask].copy()

    # stable representative sort: first objective ascending/descending
    if len(front):
        first_dir = dirs[0]
        front = front.sort_values(cols[0], ascending=(first_dir=="minimize"))
    front = front.reset_index(drop=True)
    extra_stats = {"search_domain": search_domain}
    if search_domain == "device" and "extrapolation_risk" in front.columns:
        extra_stats["n_extrap"] = int((front["extrapolation_risk"] == "存在外推风险").sum())
    if return_all:
        return front, {"n_candidates": int(n_candidates), "n_feasible": n_feasible,
                       "n_pareto": int(len(front)), **extra_stats}, result
    if return_stats:
        stats = {"n_candidates": int(n_candidates), "n_feasible": n_feasible,
                 "n_pareto": int(len(front)), **extra_stats}
        return front, stats
    return front
