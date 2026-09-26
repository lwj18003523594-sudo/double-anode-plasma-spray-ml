# -*- coding: utf-8 -*-
"""V1.8 P1 · 三种代理模型架构的消融对照（方案 §5：双层主线 + 可验证缺陷支路）。

课题研究目标为「双层代理模型」（工艺 → 过程状态 → 性能）。平台 V1.7 的三级链
（工艺 → 过程 → 缺陷 → 性能）按方案 §5.2 定位为「缺陷增强链（研究对照）」。
本模块在**完全相同的外层分组交叉验证**下公平比较三种架构：

    direct     直连基线：工艺参数（含物理派生特征）直接 → 性能模型
    two_layer  双层主链：工艺 → Stage 1 过程状态（XGBoost OOF）→ 性能模型
    three_chain 缺陷增强链：平台既有三级链（model_chain.cross_validate_chain）

判定规则（方案 §5.2，写入报告）：
    - 端到端误差：同一外层折划分下比较 Stage 3 各目标 R² / RMSE / MAE；
    - 晋级条件：缺陷增强链须在「同一外层分组验证下稳定降低端到端误差、
      改善校准」才可晋级为正式预测路径；否则三级链仅作研究支路；
    - 禁止「先选最佳模型，再把同一轮选型分数称为无偏最终结果」（选型与
      报告分离：本对照的分数是选型证据，非无偏最终性能）。

设计铁律（V1.6/V1.7/V1.8 继承）：
    - 不修改 model_chain / optimize 等既有模块（不动清单保持零 diff）；
    - 外层折划分与 cross_validate_chain 完全一致（batch 分组 GroupKFold；
      小样本 LOOCV / KFold），三种架构用同一折集合；
    - 两层链的 Stage 3 输入 = 训练折内的 Stage 1 OOF + 验证折的 Stage 1 重训预测
      （防泄漏：与 model_chain 同口径）；
    - 固定随机种子 42；结论句由真实数值拼接。
"""
from datetime import datetime
import json

import numpy as np
import pandas as pd

from .config import ROOT
from .features import add_physics_features
from .model_chain import (_fit_one_oof, _fit_performance_model, _usable_material_cols,
                          cross_validate_chain)

ARCH_DIR = ROOT / "outputs" / "architecture"
RESULT_FILE = ARCH_DIR / "架构对照结果.json"
SEED = 42


def _outer_folds(df, seed=SEED):
    """外层折划分——与 model_chain.cross_validate_chain 完全一致的规则。"""
    n = len(df)
    if n < 6:
        return []
    if "batch_id" in df.columns:
        g = df["batch_id"].astype(str)
        n_groups = int(g.nunique())
        if n_groups >= 2:
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=max(2, min(5, n_groups)))
            return list(splitter.split(df, groups=g))
    if n <= 25:
        from sklearn.model_selection import LeaveOneOut
        return list(LeaveOneOut().split(df))
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(df))


def _base_X(df, schema):
    """设计时可获得输入的设计矩阵（结构 + 工艺 + 数值材料 + 物理派生特征）。"""
    groups_cfg = {k: list(v.keys()) for k, v in schema.items()}
    x_cols = [c for c in groups_cfg["structure_inputs"] + groups_cfg["process_inputs"]
              if c in df.columns]
    baseX = add_physics_features(df[x_cols])
    material_cols = [c for c in _usable_material_cols(df, schema) if c not in baseX.columns]
    if material_cols:
        baseX = pd.concat([baseX, df[material_cols].apply(pd.to_numeric, errors="coerce")], axis=1)
    return baseX


def _perf_targets(df, schema):
    groups_cfg = {k: list(v.keys()) for k, v in schema.items()}
    return [t for t in groups_cfg["performance_outputs"]
            if t in df.columns and df[t].notna().sum() >= 20]


def _metrics(y, pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    m = pd.notna(y) & pd.notna(pred)
    if int(m.sum()) < 2:
        return None
    return {"R2": float(r2_score(y[m], pred[m])),
            "RMSE": float(mean_squared_error(y[m], pred[m]) ** 0.5),
            "MAE": float(mean_absolute_error(y[m], pred[m])),
            "n": int(m.sum())}


def _cv_generic(df, schema, mode, seed=SEED):
    """通用外层 CV：mode = "direct" | "two_layer"。

    返回 {target: {"R2", "RMSE", "MAE", "n"}}（端到端，折外预测）。
    """
    folds = _outer_folds(df, seed)
    if not folds:
        raise ValueError("样本不足，无法划分外层交叉验证折。")
    baseX = _base_X(df, schema)
    groups = df["batch_id"] if "batch_id" in df.columns else None
    targets = _perf_targets(df, schema)
    oof = {t: pd.Series(np.nan, index=df.index) for t in targets}

    for tr_idx, va_idx in folds:
        tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]
        X_tr_base, X_va_base = baseX.iloc[tr_idx], baseX.iloc[va_idx]
        if mode == "direct":
            X_tr, X_va = X_tr_base, X_va_base
        else:  # two_layer：Stage 1 OOF（训练折内）+ 重训 Stage 1 预测验证折
            state_cols = [c for c in schema.get("process_states", {})
                          if c in tr_df.columns and tr_df[c].notna().sum() >= 20]
            feats_tr, feats_va = [], []
            g_tr = pd.Series(groups).iloc[tr_idx] if groups is not None else None
            for t in state_cols:
                model1, oof1, _ = _fit_one_oof(X_tr_base, tr_df[t], groups=g_tr, seed=seed)
                col = f"pred_{t}"
                feats_tr.append(pd.DataFrame({col: oof1.to_numpy()}, index=tr_df.index))
                feats_va.append(pd.DataFrame({col: model1.predict(X_va_base).ravel()},
                                             index=va_df.index))
            if feats_tr:
                X_tr = pd.concat([X_tr_base] + feats_tr, axis=1)
                X_va = pd.concat([X_va_base] + feats_va, axis=1)
            else:
                X_tr, X_va = X_tr_base, X_va_base
        for t in targets:
            y_tr = pd.to_numeric(tr_df[t], errors="coerce")
            m = y_tr.notna()
            if int(m.sum()) < 10:
                continue
            model, _kind = _fit_performance_model(
                X_tr.loc[m] if hasattr(X_tr, "loc") else X_tr[m.values], y_tr[m], seed=seed)
            oof[t].iloc[va_idx] = model.predict(X_va)
    out = {}
    for t in targets:
        met = _metrics(pd.to_numeric(df[t], errors="coerce"), oof[t])
        if met:
            out[t] = met
    return out


def compare_architectures(df, schema, *, seed=SEED, progress_cb=None):
    """三种架构同场消融对照（同一外层折集合）。

    返回 dict 并落盘 ARCH_DIR/架构对照结果.json + 架构对照报告.md。
    """
    def _cb(msg, frac=None):
        if progress_cb:
            try:
                progress_cb(msg, frac)
            except Exception:
                pass

    _cb("直连基线（direct）外层 CV", 0.0)
    direct = _cv_generic(df, schema, "direct", seed)
    _cb("双层主链（two_layer）外层 CV", 0.35)
    two_layer = _cv_generic(df, schema, "two_layer", seed)
    _cb("缺陷增强三级链（three_chain）外层 CV", 0.7)
    cv3 = cross_validate_chain(df, schema, seed=seed)
    three_chain = {}
    _oof3 = (cv3.get("oof_predictions") or {}).get("stage3")
    if _oof3 is not None and len(getattr(_oof3, "columns", [])):
        for t, pred in _oof3.items():
            met = _metrics(pd.to_numeric(df[t], errors="coerce"), pred)
            if met:
                three_chain[t] = met
    _cb("汇总与判定", 0.95)

    # ---- 逐目标对照与晋级判定（方案 §5.2：缺陷增强链须稳定降误差才晋级）----
    comparison, wins = {}, {"direct": 0, "two_layer": 0, "three_chain": 0}
    for t in sorted(set(direct) | set(two_layer) | set(three_chain)):
        row = {}
        for name, res in (("direct", direct), ("two_layer", two_layer),
                          ("three_chain", three_chain)):
            m = res.get(t)
            row[name] = (round(m["R2"], 4) if m else None)
            row[name + "_rmse"] = (round(m["RMSE"], 4) if m else None)
        if row["direct"] is not None and row["two_layer"] is not None:
            # 双层 vs 直连：过程状态是否带来增益（双层主线价值）
            row["two_layer_gain"] = round(row["two_layer"] - row["direct"], 4)
        if row["two_layer"] is not None and row["three_chain"] is not None:
            # 缺陷层是否带来进一步增益（晋级证据）
            row["defect_gain"] = round(row["three_chain"] - row["two_layer"], 4)
        # 胜者（R² 最高；None 不参赛）
        cand = {k: row[k] for k in ("direct", "two_layer", "three_chain")
                if row.get(k) is not None}
        if cand:
            winner = max(cand, key=cand.get)
            row["winner"] = winner
            wins[winner] += 1
        comparison[t] = row

    n_cmp = sum(1 for r in comparison.values() if r.get("winner"))
    two_better = sum(1 for r in comparison.values()
                     if (r.get("two_layer_gain") or 0) > 0.02)
    defect_better = sum(1 for r in comparison.values()
                        if (r.get("defect_gain") or 0) > 0.02)
    # 晋级判定（方案 §5.2 原则：稳定降低误差才晋级，否则保留为研究支路）
    if n_cmp and defect_better >= max(1, int(n_cmp * 0.5)) and \
            wins["three_chain"] >= wins["two_layer"]:
        promote = "defect_enhanced"
        promote_txt = ("缺陷增强链在多数性能目标上稳定优于双层（"
                       f"{defect_better}/{n_cmp} 目标 R² 增益 >0.02，胜 {wins['three_chain']} 目标）——"
                       "满足晋级条件，可作为正式预测路径。")
    else:
        promote = "two_layer"
        promote_txt = (f"缺陷增强链未达晋级门槛（{defect_better}/{n_cmp} 目标增益，"
                       f"三级链胜 {wins['three_chain']} vs 双层胜 {wins['two_layer']}）——"
                       "按方案 §5.2：双层为正式主链，三级链保留为研究支路（缺陷机制分析）。")

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "cv_method": cv3.get("method"),
        "architectures": {
            "direct": {"desc": "直连基线（工艺→性能，无过程/缺陷中介）", "metrics": direct},
            "two_layer": {"desc": "双层主链（工艺→过程状态 OOF→性能，课题主线）",
                          "metrics": two_layer},
            "three_chain": {"desc": "缺陷增强链（工艺→过程→缺陷→性能，V1.7 三级链）",
                            "metrics": three_chain},
        },
        "comparison": comparison, "wins": wins,
        "n_compared": n_cmp,
        "promote": promote, "promote_text": promote_txt,
        "selection_caveat": "本对照分数为模型选型证据（同轮比较），"
                           "非无偏最终性能；最终性能以晋级架构的独立验证为准。",
    }
    from .status_binding import bind_identity
    bind_identity(result, note="架构对照（选型证据）")

    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _write_report(result)
    _cb("架构对照完成", 1.0)
    return result


def _write_report(result):
    from .ui_labels import zh
    lines = ["# 架构消融对照报告（直连 vs 双层 vs 缺陷增强链）", "",
             f"- 生成时间：{result['generated_at']} ｜ CV：{result.get('cv_method')}",
             f"- 绑定：{result.get('binding', {}).get('model_version') or '无部署模型（纯选型对照）'}",
             "", "## 端到端（Stage 3 性能）折外指标对照", "",
             "| 性能目标 | 直连 R² | 双层 R² | 缺陷增强 R² | 双层增益 | 缺陷层增益 | 胜者 |",
             "|---|---|---|---|---|---|---|"]
    for t, r in result["comparison"].items():
        def _f(v):
            return f"{v:+.3f}" if v is not None else "—"
        lines.append(f"| {zh(t)} | {_f(r.get('direct'))} | {_f(r.get('two_layer'))} "
                     f"| {_f(r.get('three_chain'))} | {_f(r.get('two_layer_gain'))} "
                     f"| {_f(r.get('defect_gain'))} | {r.get('winner', '—')} |")
    lines += ["", f"## 晋级判定（方案 §5.2）", "",
              f"- 各架构胜出目标数：直连 {result['wins']['direct']} ｜ 双层 {result['wins']['two_layer']}"
              f" ｜ 缺陷增强 {result['wins']['three_chain']}（共 {result['n_compared']} 目标）",
              f"- **{result['promote_text']}**", "",
              "> " + result["selection_caveat"], "",
              "> 选型与报告分离：本对照用于选择正式预测路径；被选架构的最终性能"
              "以其独立外层验证为准，不得把本对照分数当无偏最终结果。"]
    (ARCH_DIR / "架构对照报告.md").write_text("\n".join(lines), encoding="utf-8")


def architecture_status(current_model_version=None, current_dataset_hash=None):
    try:
        with open(RESULT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        from .status_binding import freshness
        fr = freshness(data, current_model_version, current_dataset_hash)
        return {"done": True, "promote": data.get("promote"),
                "generated_at": data.get("generated_at"),
                "freshness": fr["state"], "freshness_note": fr["note"]}
    except Exception:
        return {"done": False}
