from pathlib import Path
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, balanced_accuracy_score, f1_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor, XGBClassifier

from .features import add_physics_features
from .research_utils import training_domain, save_domain_files

def _metrics(y, pred):
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAE": float(mean_absolute_error(y, pred)),
    }

def _classif_metrics(y_true_codes, y_pred_codes):
    """分类目标（Stage 1.5）OOF 指标：不只报告 Accuracy，类别不平衡时看平衡准确率。"""
    return {
        "Accuracy": float(accuracy_score(y_true_codes, y_pred_codes)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true_codes, y_pred_codes)),
        "F1_macro": float(f1_score(y_true_codes, y_pred_codes, average="macro")),
    }

def _base_xgb(seed=42):
    return XGBRegressor(
        n_estimators=350, max_depth=4, learning_rate=0.035,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, objective="reg:squarederror",
        random_state=seed, n_jobs=-1
    )

def _fit_one_oof(X, y, groups=None, seed=42):
    mask = pd.notna(y)
    Xv = X.loc[mask].copy()
    yv = pd.Series(y).loc[mask].astype(float).copy()
    gv = pd.Series(groups).loc[mask] if groups is not None else None

    valid_group_cv = gv is not None and gv.nunique() >= 4
    if valid_group_cv:
        n_splits = min(5, int(gv.nunique()))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(Xv, yv, gv)
    else:
        n_splits = min(5, max(3, len(Xv)//25))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(Xv, yv)

    oof = pd.Series(index=Xv.index, dtype=float)
    for tr, va in splits:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _base_xgb(seed)),
        ])
        pipe.fit(Xv.iloc[tr], yv.iloc[tr])
        oof.iloc[va] = pipe.predict(Xv.iloc[va])

    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", _base_xgb(seed)),
    ])
    final_model.fit(Xv, yv)
    return final_model, oof, _metrics(yv.loc[oof.dropna().index], oof.dropna())


def _base_xgb_clf(seed=42):
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, random_state=seed, n_jobs=-1,
        eval_metric="mlogloss",
    )


def _fit_one_classif_oof(X, y, groups=None, seed=42):
    """分类目标（Stage 1.5：melting_state_class / splat_state_class 等）分组 OOF。

    与回归路径同一套折划分规则（有批次用 GroupKFold，否则 KFold），保证防泄漏一致。
    返回 (final_model, oof_codes, metrics, classes)；类别数 < 2 时返回 None。
    """
    mask = pd.notna(y)
    Xv = X.loc[mask].copy()
    yv_raw = pd.Series(y).loc[mask].astype(str).copy()
    gv = pd.Series(groups).loc[mask] if groups is not None else None

    classes = sorted(yv_raw.unique())
    if len(classes) < 2:
        return None
    cls_to_code = {c: i for i, c in enumerate(classes)}
    yv = yv_raw.map(cls_to_code).astype(float)

    valid_group_cv = gv is not None and gv.nunique() >= 4
    if valid_group_cv:
        n_splits = min(5, int(gv.nunique()))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(Xv, yv, gv)
    else:
        n_splits = min(5, max(3, len(Xv)//25))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(Xv, yv)

    oof = pd.Series(index=Xv.index, dtype=float)
    for tr, va in splits:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _base_xgb_clf(seed)),
        ])
        pipe.fit(Xv.iloc[tr], yv.iloc[tr])
        oof.iloc[va] = pipe.predict(Xv.iloc[va])

    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", _base_xgb_clf(seed)),
    ])
    final_model.fit(Xv, yv)
    ov = oof.dropna()
    av = yv.loc[ov.index]
    return final_model, oof, _classif_metrics(av, ov), classes

def _fit_performance_model(X, y, seed=42):
    mask = pd.notna(y)
    Xv = X.loc[mask].copy()
    yv = pd.Series(y).loc[mask].astype(float).copy()

    # GP for small/medium experimental data; RF fallback for larger sets.
    if len(Xv) <= 450 and Xv.shape[1] <= 45:
        kernel = ConstantKernel(1.0, (1e-2,1e2)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3)
        core = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=0)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", core)
        ])
        kind = "GaussianProcess"
    else:
        core = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", core)
        ])
        kind = "RandomForest"
    model.fit(Xv, yv)
    # 科研可靠性约定：严禁用训练集拟合值(pred = model.predict(X))计算 R2/RMSE/MAE
    # 作为模型评价。三级模型评价指标统一由 cross_validate_chain() 的
    # 全链分组交叉验证给出（同一 batch 不同时出现在训练/验证集）。
    return model, kind

def _usable_material_cols(df, schema):
    """材料/粉末属性层中"可数值化且在数据中存在"的列（分类列不进入模型特征）。"""
    out = []
    for col, spec in (schema.get("material_inputs") or {}).items():
        if col not in df.columns or spec.get("categorical"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            out.append(col)
    return out


def _melting_split(df, schema):
    """解析熔融层可用目标：{"cont": [...], "cat": [...]}（有效值 >= 20 才可用）。

    melting_data_source 等元数据字段不作为预测目标。
    铁律：温度超过熔点 ≠ 完全熔融 —— 平台不使用任何温度阈值规则自动生成熔融标签，
    只使用数据/文献/CFD 本身提供的熔融状态标签。
    """
    cont, cat = [], []
    for col, spec in (schema.get("melting_states") or {}).items():
        if col not in df.columns or spec.get("role") == "metadata":
            continue
        if col == "melting_data_source":
            continue
        s = df[col]
        if s.notna().sum() < 20:
            continue
        if spec.get("categorical") or not pd.api.types.is_numeric_dtype(s):
            cat.append(col)
        else:
            cont.append(col)
    return {"cont": cont, "cat": cat}


def train_chain(df, schema, model_path, seed=42):
    groups_cfg = {k:list(v.keys()) for k,v in schema.items()}
    state_cols = groups_cfg["process_states"]
    defect_cols = groups_cfg["defect_network"]
    perf_cols = groups_cfg["performance_outputs"]

    work = df.copy()
    # V1.4：所有模式字段允许缺省 —— 只使用数据中实际存在的输入列
    #（dual demo 数据齐全，此过滤为 no-op，训练路径不变）
    x_cols = [c for c in groups_cfg["structure_inputs"] + groups_cfg["process_inputs"]
              if c in work.columns]
    baseX = add_physics_features(work[x_cols])
    # V1.4 材料层：数值型材料/粉末属性并入 Stage 1 输入（无材料列时与旧版完全一致；
    # 已存在于 x_cols 的列不重复并入，避免重复特征名）
    material_cols = [c for c in _usable_material_cols(work, schema) if c not in x_cols]
    if material_cols:
        baseX = pd.concat([baseX, work[material_cols].apply(pd.to_numeric, errors="coerce")], axis=1)
    group_ids = work["batch_id"] if "batch_id" in work.columns else None

    bundle = {
        "x_cols": x_cols,
        "material_cols": material_cols,
        "seed": seed,
        "stage1_features": list(baseX.columns),
        "stage1_models": {}, "stage2_models": {}, "stage3_models": {},
        "stage15_models": {},
        "state_cols": state_cols, "defect_cols": defect_cols, "perf_cols": perf_cols,
        "metrics": {"stage1":{}, "stage2":{}, "stage3":{}, "stage15":{}}
    }

    # Stage 1: structure/process(/material) -> process states, using OOF for next stage
    stage1_oof = pd.DataFrame(index=work.index)
    for target in state_cols:
        if target not in work.columns or work[target].notna().sum() < 20:
            continue
        model, oof, met = _fit_one_oof(baseX, work[target], group_ids, seed=seed)
        bundle["stage1_models"][target] = model
        bundle["metrics"]["stage1"][target] = met
        stage1_oof[f"pred_{target}"] = oof

    # ---- V1.4 Stage 1.5：颗粒熔融与沉积状态（可选层）----
    # 启用条件：数据本身存在足够的熔融状态标签（>= 20 条有效值）。
    # 无标签 → 整层自动跳过，Stage 1 → 2 → 3 照常运行（与旧版逐位一致）。
    melt_split = _melting_split(work, schema)
    stage15_enabled = bool(melt_split["cont"] or melt_split["cat"])
    bundle["stage15_enabled"] = stage15_enabled
    bundle["stage15_targets"] = {"cont": melt_split["cont"], "cat": melt_split["cat"]}
    bundle["stage15_kind"] = {}
    bundle["stage15_classes"] = {}
    bundle["stage15_features"] = []
    stage15_oof = pd.DataFrame(index=work.index)
    if stage15_enabled:
        X15 = pd.concat([baseX, stage1_oof], axis=1)
        bundle["stage15_features"] = list(X15.columns)
        for target in melt_split["cont"]:
            model, oof, met = _fit_one_oof(X15, work[target], group_ids, seed=seed)
            bundle["stage15_models"][target] = model
            bundle["stage15_kind"][target] = "regression"
            bundle["metrics"]["stage15"][target] = met
            stage15_oof[f"pred_{target}"] = oof
        for target in melt_split["cat"]:
            res = _fit_one_classif_oof(X15, work[target], group_ids, seed=seed)
            if res is None:
                continue
            model, oof, met, classes = res
            bundle["stage15_models"][target] = model
            bundle["stage15_kind"][target] = "classification"
            bundle["stage15_classes"][target] = classes
            bundle["metrics"]["stage15"][target] = met
            stage15_oof[f"pred_{target}"] = oof
        if not bundle["stage15_models"]:
            stage15_enabled = False
            bundle["stage15_enabled"] = False
            bundle["metrics"].pop("stage15", None)
            stage15_oof = pd.DataFrame(index=work.index)

    # Stage 2: predicted process states (+ V1.4 熔融层 OOF) + original physics-aware inputs -> defect network
    # 防泄漏铁律：Stage 2 只吃 Stage 1.5 的 OOF 预测，绝不吃训练集拟合值。
    if stage15_enabled:
        X2 = pd.concat([baseX, stage1_oof, stage15_oof], axis=1)
    else:
        X2 = pd.concat([baseX, stage1_oof], axis=1)
    bundle["stage2_features"] = list(X2.columns)
    stage2_oof = pd.DataFrame(index=work.index)
    for target in defect_cols:
        if target not in work.columns or work[target].notna().sum() < 20:
            continue
        model, oof, met = _fit_one_oof(X2, work[target], group_ids, seed=seed)
        bundle["stage2_models"][target] = model
        bundle["metrics"]["stage2"][target] = met
        stage2_oof[f"pred_{target}"] = oof

    # Stage 3: predicted states (+ 熔融层) + predicted defects + original inputs -> performance
    # 注意：metrics["stage3"] 不在此处填写 —— 三级指标禁止用训练集拟合值计算，
    # 统一由 cross_validate_chain() 的全链分组交叉验证提供。
    if stage15_enabled:
        X3 = pd.concat([baseX, stage1_oof, stage15_oof, stage2_oof], axis=1)
    else:
        X3 = pd.concat([baseX, stage1_oof, stage2_oof], axis=1)
    bundle["stage3_features"] = list(X3.columns)
    for target in perf_cols:
        if target not in work.columns or work[target].notna().sum() < 20:
            continue
        model, kind = _fit_performance_model(X3, work[target], seed=seed)
        bundle["stage3_models"][target] = {"model": model, "kind": kind}

    if model_path is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
    return bundle

def _align(X, feature_names):
    X = X.copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_names]

def predict_chain(df_inputs, bundle, return_uncertainty=True, return_melting=False):
    # reindex：预测表单未提供的输入列自动按缺失值处理（中位数插补），不报错
    base = add_physics_features(df_inputs.reindex(columns=bundle["x_cols"]))
    mat_cols = bundle.get("material_cols") or []
    if mat_cols:
        # 材料/粉末属性列（预测表单未提供时自动按缺失值处理，由中位数插补）
        base = pd.concat([base, df_inputs.reindex(columns=mat_cols).apply(pd.to_numeric, errors="coerce")], axis=1)
    X1 = _align(base, bundle["stage1_features"])
    states = pd.DataFrame(index=df_inputs.index)
    for t,m in bundle["stage1_models"].items():
        states[t] = m.predict(X1)

    # ---- V1.4 Stage 1.5（可选层）：只在启用时计算；未启用时路径与旧版一致 ----
    melt_cont = pd.DataFrame(index=df_inputs.index)
    melt_class = pd.DataFrame(index=df_inputs.index)
    melt_pred = pd.DataFrame(index=df_inputs.index)
    if bundle.get("stage15_enabled"):
        X15 = _align(pd.concat([base, states.add_prefix("pred_")], axis=1),
                     bundle["stage15_features"])
        for t, m in bundle["stage15_models"].items():
            if bundle["stage15_kind"].get(t) == "classification":
                codes = np.asarray(m.predict(X15), dtype=float)
                melt_pred[f"pred_{t}"] = codes
                classes = bundle.get("stage15_classes", {}).get(t) or []
                melt_class[t] = [classes[int(c)] if 0 <= int(c) < len(classes) else str(c)
                                 for c in codes]
            else:
                vals = np.asarray(m.predict(X15), dtype=float)
                melt_pred[f"pred_{t}"] = vals
                melt_cont[t] = vals

    X2_parts = [base, states.add_prefix("pred_")] + ([melt_pred] if bundle.get("stage15_enabled") else [])
    X2 = _align(pd.concat(X2_parts, axis=1), bundle["stage2_features"])
    defects = pd.DataFrame(index=df_inputs.index)
    for t,m in bundle["stage2_models"].items():
        defects[t] = m.predict(X2)

    X3_parts = [base, states.add_prefix("pred_"), defects.add_prefix("pred_")] \
        + ([melt_pred] if bundle.get("stage15_enabled") else [])
    X3 = _align(pd.concat(X3_parts, axis=1), bundle["stage3_features"])
    perf = pd.DataFrame(index=df_inputs.index)
    unc = pd.DataFrame(index=df_inputs.index)
    for t,info in bundle["stage3_models"].items():
        m = info["model"]
        perf[t] = m.predict(X3)
        if return_uncertainty and info["kind"] == "GaussianProcess":
            # Manually pass through preprocessors to obtain GP std.
            Xi = m.named_steps["imputer"].transform(X3)
            Xs = m.named_steps["scale"].transform(Xi)
            mean, std = m.named_steps["model"].predict(Xs, return_std=True)
            perf[t] = mean
            unc[t+"_std"] = std
        elif return_uncertainty and info["kind"] == "RandomForest":
            Xi = m.named_steps["imputer"].transform(X3)
            rf = m.named_steps["model"]
            tree_preds = np.column_stack([tree.predict(Xi) for tree in rf.estimators_])
            unc[t+"_std"] = tree_preds.std(axis=1)

    if return_melting:
        return states, defects, perf, unc, {"continuous": melt_cont, "class": melt_class}
    return states, defects, perf, unc


# ============================================================
# 全链分组交叉验证（评价用）与最终部署模型训练的明确区分：
#   - cross_validate_chain(): 评价模型。按 batch_id 分组外层 CV，
#     每个 fold 内重新训练 Stage1→2→3 并只预测该 fold 的验证批次，
#     汇总全部外层验证预测后计算 R2/RMSE/MAE。
#   - train_chain_full():     部署模型。用全部真实数据训练一次最终
#     模型用于实际预测；同时调用 cross_validate_chain 得到诚实指标，
#     并写入版本记录（不覆盖旧版本）。
# ============================================================

def cross_validate_chain(df, schema, seed=42, n_splits=None):
    """整条三级代理模型的 batch_id 分组外层交叉验证。

    同一 batch_id 绝不会同时出现在同一 fold 的训练集和验证集。
    返回 {"method", "metrics", "fold_info"}；
    metrics = {"stage1": {target: {R2, RMSE, MAE}}, "stage2": ..., "stage3": ...}
    """
    groups_cfg = {k: list(v.keys()) for k, v in schema.items()}
    state_cols = groups_cfg["process_states"]
    defect_cols = groups_cfg["defect_network"]
    perf_cols = groups_cfg["performance_outputs"]

    empty = {"method": "none", "metrics": {"stage1": {}, "stage2": {}, "stage3": {}}, "fold_info": []}
    if len(df) == 0:
        return empty
    has_batch = "batch_id" in df.columns
    if has_batch:
        g = df["batch_id"].astype(str)
        n_groups = int(g.nunique())
    else:
        g, n_groups = None, 0
    # V1.4 spec 三十八：存在真实独立 batch → GroupKFold；否则小样本 LOOCV / 5 折 KFold。
    # 严禁人为制造 batch_id。
    use_grouped = has_batch and n_groups >= 2
    if use_grouped:
        if n_splits is None:
            n_splits = min(5, n_groups)
        if n_splits < 2:
            return empty
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, groups=g))
        method = f"GroupKFold(n_splits={n_splits}) by batch_id, full-chain (Stage1→2→3 retrained per fold)"
    else:
        n_rows = len(df)
        if n_rows < 6:
            return empty
        if n_rows <= 25:
            from sklearn.model_selection import LeaveOneOut
            splits = list(LeaveOneOut().split(df))
            method = f"LOOCV (n={n_rows}), full-chain (Stage1→2→3 retrained per fold)"
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
            splits = list(splitter.split(df))
            method = "KFold(n_splits=5, shuffle), full-chain (Stage1→2→3 retrained per fold)"

    preds = {
        "stage1": {t: pd.Series(np.nan, index=df.index) for t in state_cols},
        "stage2": {t: pd.Series(np.nan, index=df.index) for t in defect_cols},
        "stage3": {t: pd.Series(np.nan, index=df.index) for t in perf_cols},
    }
    fold_info = []

    for k, (tr_idx, va_idx) in enumerate(splits):
        tr_df = df.iloc[tr_idx]
        va_df = df.iloc[va_idx]

        # 防泄漏硬校验：同一 batch 不得同时出现在训练/验证集（仅分组模式）
        if use_grouped:
            tr_batches = set(tr_df["batch_id"].astype(str))
            va_batches = set(va_df["batch_id"].astype(str))
            overlap = tr_batches & va_batches
            if overlap:
                raise RuntimeError(
                    f"检测到数据泄漏：fold {k} 中以下 batch_id 同时出现在训练/验证集: {sorted(overlap)}")
        else:
            tr_batches, va_batches = set(), set()

        fold_bundle = train_chain(tr_df, schema, model_path=None, seed=seed)
        _states, _defects, _perf, _munc, _melt = predict_chain(va_df, fold_bundle,
                                                               return_melting=True)
        states, defects, perf, _unc = _states, _defects, _perf, _munc

        for t in fold_bundle["stage1_models"]:
            preds["stage1"][t].iloc[va_idx] = states[t].to_numpy()
        for t in fold_bundle["stage2_models"]:
            preds["stage2"][t].iloc[va_idx] = defects[t].to_numpy()
        for t in fold_bundle["stage3_models"]:
            preds["stage3"][t].iloc[va_idx] = perf[t].to_numpy()
        # V1.4：Stage 1.5 OOF（仅启用时产生；fold 内重新训练，只预测验证批次）
        # 连续目标与分类目标分开记录（避免 dtype 混用）
        if _melt is not None and fold_bundle.get("stage15_enabled"):
            for t in _melt["continuous"].columns:
                preds.setdefault("stage15_cont", {}).setdefault(
                    t, pd.Series(np.nan, index=df.index))
                preds["stage15_cont"][t].iloc[va_idx] = _melt["continuous"][t].to_numpy()
            for t in _melt["class"].columns:
                preds.setdefault("stage15_class", {}).setdefault(
                    t, pd.Series(index=df.index, dtype=object))
                preds["stage15_class"][t].iloc[va_idx] = _melt["class"][t].to_numpy()

        # V1.5：记录 stage1 首目标 Top5 特征（供跨折稳定性分析；不影响训练与预测）
        fi_top = None
        try:
            _t0 = next(iter(fold_bundle["stage1_models"]))
            _imp = pd.Series(fold_bundle["stage1_models"][_t0].named_steps["model"].feature_importances_,
                             index=fold_bundle["stage1_features"]).sort_values(ascending=False)
            fi_top = list(_imp.head(5).index)
        except Exception:
            fi_top = None

        fold_info.append({
            "fold": int(k),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "train_batches": sorted(tr_batches),
            "val_batches": sorted(va_batches),
            "stage1_fi_top": fi_top,
        })

    metrics = {"stage1": {}, "stage2": {}, "stage3": {}}
    for stage, cols in [("stage1", state_cols), ("stage2", defect_cols), ("stage3", perf_cols)]:
        for t in cols:
            pred = preds[stage][t].dropna()
            if len(pred) < 2:
                continue
            y = df.loc[pred.index, t]
            mask = y.notna()
            if int(mask.sum()) < 2:
                continue
            metrics[stage][t] = _metrics(y[mask], pred[mask])

    # V1.4：Stage 1.5 指标（连续→R2/RMSE/MAE；分类→Accuracy/Balanced_Accuracy/F1_macro）
    if "stage15_cont" in preds or "stage15_class" in preds:
        metrics["stage15"] = {}
        for t, pred in preds.get("stage15_cont", {}).items():
            pred_v = pred.dropna()
            if len(pred_v) < 2:
                continue
            y_raw = pd.to_numeric(df.loc[pred_v.index, t], errors="coerce")
            mask = y_raw.notna()
            if int(mask.sum()) < 2:
                continue
            metrics["stage15"][t] = _metrics(y_raw[mask], pred_v[mask])
        for t, pred in preds.get("stage15_class", {}).items():
            pred_v = pred.dropna()
            if len(pred_v) < 2:
                continue
            y_labels = df.loc[pred_v.index, t].astype(str)
            mask = y_labels.notna()
            if int(mask.sum()) < 2:
                continue
            p_labels = pred_v.astype(str)
            classes = sorted(set(y_labels[mask].unique()) | set(p_labels.unique()))
            code_map = {c: i for i, c in enumerate(classes)}
            metrics["stage15"][t] = _classif_metrics(y_labels[mask].map(code_map),
                                                     p_labels.map(code_map))

    oof_predictions = {stage: pd.DataFrame(preds[stage]) for stage in preds
                       if stage not in ("stage15_cont", "stage15_class")}
    if "stage15_cont" in preds or "stage15_class" in preds:
        oof_predictions["stage15"] = pd.DataFrame(
            {**preds.get("stage15_cont", {}), **preds.get("stage15_class", {})})

    return {
        "method": method,
        "metrics": metrics,
        "fold_info": fold_info,
        # 每行样本的外层验证预测（仅验证批次有值），供论文评价图使用；
        # 同一 batch 绝不跨训练/验证，故这些预测均来自未见过该批次的模型。
        "oof_predictions": oof_predictions,
    }


_XGB_HYPERPARAMS = {
    "n_estimators": 350, "max_depth": 4, "learning_rate": 0.035,
    "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0,
    "objective": "reg:squarederror",
}


def train_chain_full(df, schema, model_path, seed=42, n_splits=None):
    """训练最终部署模型 + 全链分组交叉验证评价 + 版本记录（不覆盖旧版本）。

    - 部署模型（model_path / latest_chain.joblib）使用全部数据训练，用于实际预测；
    - 评价指标来自 cross_validate_chain 的全链分组外层 CV；
    - 每次训练在 models/history/<model_version>/ 下保存独立副本与 manifest.json。
    """
    t0 = datetime.now()
    bundle = train_chain(df, schema, model_path, seed=seed)
    cv = cross_validate_chain(df, schema, seed=seed, n_splits=n_splits)
    bundle["chain_cv"] = cv

    model_version = "v" + t0.strftime("%Y%m%d_%H%M%S")
    bundle["model_version"] = model_version
    dataset_hash = hashlib.sha256(df.to_csv(index=True).encode("utf-8")).hexdigest()
    bundle["dataset_hash"] = dataset_hash
    # 训练数据覆盖范围（输入特征 min/max/median/Q1/Q3），用于外推检测；
    # 该范围来源于当前训练数据，并非设备极限。V1.4：覆盖范围包含材料数值列。
    domain = training_domain(df, bundle["x_cols"] + (bundle.get("material_cols") or []))
    bundle["training_domain"] = domain
    save_domain_files(domain, Path(model_path))
    schema_str = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    schema_version = "v1_" + hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:8]

    manifest = {
        "model_version": model_version,
        "training_time": t0.isoformat(timespec="seconds"),
        "random_seed": seed,
        "dataset_version": "auto_" + dataset_hash[:8],
        "dataset_hash": dataset_hash,
        "schema_version": schema_version,
        "batch_count": int(df["batch_id"].nunique()) if "batch_id" in df.columns else None,
        "sample_count": int(len(df)),
        "feature_list": {
            "stage1": bundle["stage1_features"],
            "stage2": bundle["stage2_features"],
            "stage3": bundle["stage3_features"],
        },
        "target_list": {
            "stage1": list(bundle["stage1_models"].keys()),
            "stage2": list(bundle["stage2_models"].keys()),
            "stage3": list(bundle["stage3_models"].keys()),
        },
        "cross_validation_method": cv["method"],
        "fold_groups": [
            {"fold": fi["fold"], "val_batches": fi["val_batches"]}
            for fi in cv["fold_info"]
        ],
        "model_hyperparameters": {
            "xgboost": _XGB_HYPERPARAMS,
            "stage1_stage2_cv": "GroupKFold by batch_id (OOF); fallback KFold if <4 groups",
            "stage3_small_data": "Pipeline(median imputer -> standard scaler -> GaussianProcess(Matern nu=1.5 + WhiteKernel, normalize_y))",
            "stage3_large_data": "Pipeline(median imputer -> RandomForest(n_estimators=500, min_samples_leaf=2))",
        },
        "R2": {s: {t: round(m["R2"], 4) for t, m in cv["metrics"].get(s, {}).items()}
               for s in ["stage1", "stage2", "stage3"]},
        "RMSE": {s: {t: round(m["RMSE"], 4) for t, m in cv["metrics"].get(s, {}).items()}
                 for s in ["stage1", "stage2", "stage3"]},
        "MAE": {s: {t: round(m["MAE"], 4) for t, m in cv["metrics"].get(s, {}).items()}
                for s in ["stage1", "stage2", "stage3"]},
        "role": {
            "evaluation": "全链分组交叉验证（fold 内重新训练 Stage1→2→3，仅预测验证批次）",
            "deployment": "使用全部数据训练的最终模型（latest_chain.joblib）",
        },
    }

    model_path = Path(model_path)
    # 部署模型文件必须包含 chain_cv（论文评价图依赖逐行 OOF 预测）——重新落盘
    joblib.dump(bundle, model_path)
    hist_dir = model_path.parent / "history" / model_version
    hist_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, hist_dir / "model.joblib")
    with open(hist_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(model_path.parent / "latest_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return bundle
