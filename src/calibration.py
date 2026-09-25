# -*- coding: utf-8 -*-
"""V1.7 L1 · 不确定度校准模块（核心价值验证第一层）。

回答的问题：模型给出的不确定度（σ）是否可信？

方法（对应指令集阶段一，按平台三级链式架构适配）：
  1. 带 σ 的全链交叉验证——折划分规则与 model_chain.cross_validate_chain
     完全一致（batch_id 分组 GroupKFold；无分组时小样本 LOOCV / 5 折 KFold），
     每折内重新训练 Stage1→(1.5)→2→3，对验证批次同时输出预测均值与标准差
     （Stage 3 为 GPR 后验 σ 或 RF 树间 σ，由 model_chain.predict_chain 提供）；
  2. 对每个性能目标计算校准指标：
     - 1σ/2σ 覆盖率（|y_true - oof_mean| ≤ k·σ 的比例，期望 ≈68% / ≈95%）；
     - z 分数方差 s = mean(((y_true-oof_mean)/oof_std)^2)，校准因子 k = sqrt(s)；
  3. 校准曲线（横轴名义置信度 1%~99%，纵轴实际覆盖率，对角线为理想线）；
  4. 各目标校准因子写入 models/calibration.json；下游（工艺预测 / 逆向设计候选表 /
     L3 效率基准的悲观惩罚）的 σ 展示统一乘以校准因子（文件不存在时因子=1）。

设计铁律（V1.6 继承）：
  - 本模块不修改 model_chain / optimize 等任何既有模块（不动清单 diff 保持为零），
    校准因子在调用方（展示层 / benchmark）应用；
  - 结论句一律由真实计算数值拼接（statement 风格），缺数据时如实降级，不编造；
  - 固定随机种子 42；图表走 plot_style（网页）与 paper_style（落盘）统一色板。
"""
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from .config import ROOT
from .model_chain import train_chain, predict_chain
from .research_utils import fmt_metric

CALIB_FILE = ROOT / "models" / "calibration.json"
CALIB_DIR = ROOT / "outputs" / "calibration"
SEED = 42

# 校准判定区间：1σ 覆盖率落在该区间视为「校准良好」（68% 的合理波动范围）
GOOD_COV1 = (0.55, 0.80)


def _chain_folds(df, seed=SEED):
    """折划分索引列表——规则与 model_chain.cross_validate_chain 完全一致。

    存在 >= 2 个真实 batch → GroupKFold by batch_id（防泄漏）；
    无分组且 <= 25 行 → LOOCV；否则 5 折 KFold(shuffle, seed)。
    返回 [(train_idx, val_idx), ...]；不可划分时返回 []。
    """
    n = len(df)
    if n < 6:
        return []
    if "batch_id" in df.columns:
        g = df["batch_id"].astype(str)
        n_groups = int(g.nunique())
        if n_groups >= 2:
            from sklearn.model_selection import GroupKFold
            n_splits = max(2, min(5, n_groups))
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(df, groups=g))
    if n <= 25:
        from sklearn.model_selection import LeaveOneOut
        return list(LeaveOneOut().split(df))
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(df))


def run_calibration(df, schema, *, seed=SEED, progress_cb=None):
    """运行带 σ 的全链 CV 并产出校准指标。

    返回 dict：{"method", "generated_at", "targets": {t: {n, cov_1sigma, cov_2sigma,
    z_var, factor, verdict}}, "oof": DataFrame}；同时落盘 calibration.json、
    校准曲线 PNG（matplotlib 论文版）与校准报告 md。
    """
    if df is None or len(df) < 10:
        raise ValueError("校准需要至少 10 行数据。")
    folds = _chain_folds(df, seed=seed)
    if not folds:
        raise ValueError("当前数据无法划分交叉验证折（样本过少或无有效 batch）。")

    def _cb(name, frac=None):
        if progress_cb:
            try:
                progress_cb(name, frac)
            except Exception:
                pass

    _cb("带 σ 全链交叉验证启动", 0.0)

    # ---- 逐折训练 + 对验证批次输出 mean/std（复制 cross_validate_chain 的折循环，额外保留 σ）----
    oof_rows = []          # 每行：{目标: (mean, std)} + y_true
    method_parts = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        tr_df = df.iloc[tr_idx]
        va_df = df.iloc[va_idx]
        fold_bundle = train_chain(tr_df, schema, model_path=None, seed=seed)
        _states, _defects, _perf, _unc = predict_chain(va_df, fold_bundle,
                                                        return_uncertainty=True)
        row_block = pd.DataFrame(index=va_df.index)
        for t in fold_bundle["stage3_models"]:
            if t not in va_df.columns:
                continue
            row_block[t + "__mean"] = _perf[t].to_numpy() if t in _perf else np.nan
            sc = t + "_std"
            row_block[t + "__std"] = _unc[sc].to_numpy() if sc in _unc else np.nan
            row_block[t + "__true"] = pd.to_numeric(va_df[t], errors="coerce").to_numpy()
        oof_rows.append(row_block)
        method_parts.append(f"fold{k}(n_val={len(va_idx)})")
        _cb(f"折 {k+1}/{len(folds)} 完成", (k + 1) / len(folds))

    oof_all = pd.concat(oof_rows).sort_index()

    # ---- 逐目标校准指标 ----
    targets_out = {}
    for t in [c[:-6] for c in oof_all.columns if c.endswith("__mean")]:
        sub = oof_all[[t + "__mean", t + "__std", t + "__true"]].dropna()
        # σ 非正（含 0 / NaN）的样本无法构造 z 分数，如实剔除并记录
        sub = sub[sub[t + "__std"] > 1e-12]
        n = len(sub)
        if n < 8:
            targets_out[t] = {"n": int(n), "verdict": "样本不足",
                             "note": "有效 (mean, σ, y_true) 三元组不足 8 个，无法评估校准性。"}
            continue
        z = ((sub[t + "__true"] - sub[t + "__mean"]) / sub[t + "__std"]).to_numpy()
        cov1 = float(np.mean(np.abs(z) <= 1.0))
        cov2 = float(np.mean(np.abs(z) <= 2.0))
        z_var = float(np.mean(z ** 2))
        factor = float(np.sqrt(z_var))
        if GOOD_COV1[0] <= cov1 <= GOOD_COV1[1]:
            verdict = "校准良好"
        elif cov1 < GOOD_COV1[0]:
            verdict = "过自信（区间偏窄，实际覆盖率低于名义值）"
        else:
            verdict = "过保守（区间偏宽，实际覆盖率高于名义值）"
        targets_out[t] = {
            "n": int(n), "cov_1sigma": round(cov1, 4), "cov_2sigma": round(cov2, 4),
            "z_var": round(z_var, 4), "factor": round(factor, 4), "verdict": verdict,
            "mean_abs_z": round(float(np.mean(np.abs(z))), 4),
        }
    if not targets_out:
        raise ValueError("没有可评估的性能目标（Stage 3 无可用模型或有效样本不足）。")

    method = ("带 σ 全链交叉验证（" + "；".join(method_parts[:3])
              + ("；…" if len(method_parts) > 3 else "")
              + "）；折划分规则与平台全链 CV 一致（batch 分组 GroupKFold / LOOCV / KFold）")

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": method, "seed": int(seed),
        "targets": targets_out,
        "oof_with_std": oof_all,
    }

    # ---- 落盘：calibration.json（因子供下游使用）----
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    persist = {k: v for k, v in result.items() if k != "oof_with_std"}
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(persist, f, ensure_ascii=False, indent=2)

    _cb("绘制校准曲线", 0.92)
    _plot_calibration_curve(targets_out, oof_all)
    _write_report(result)
    _cb("校准完成", 1.0)
    return result


def calibration_curve_points(oof_all, t):
    """校准曲线数据点：名义置信度(%) vs 实际覆盖率(%)（z 分数法）。"""
    sub = oof_all[[t + "__mean", t + "__std", t + "__true"]].dropna()
    sub = sub[sub[t + "__std"] > 1e-12]
    if len(sub) < 8:
        return None, None
    z = ((sub[t + "__true"] - sub[t + "__mean"]) / sub[t + "__std"]).to_numpy()
    nom = np.linspace(1, 99, 25)
    act = np.array([np.mean(np.abs(z) <= sps.norm.ppf(0.5 + p / 200.0)) * 100 for p in nom])
    return nom, act


def _plot_calibration_curve(targets_out, oof_all):
    """落盘 matplotlib 论文级校准曲线（89mm 单栏，多目标合一，统一色板）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plot_style as ps
    from . import paper_style as psty

    psty.apply_paper_style()
    fig, ax = plt.subplots(figsize=psty.fig_single(2.4))
    ok = [t for t, info in targets_out.items() if "cov_1sigma" in info]
    for i, t in enumerate(ok):
        nom, act = calibration_curve_points(oof_all, t)
        if nom is None:
            continue
        ax.plot(nom, act, "-o", ms=2.6, lw=1.2, color=ps.QUAL_CYCLE[i % len(ps.QUAL_CYCLE)],
                label=t)
    ax.plot([0, 100], [0, 100], "--", lw=1.0, color=ps.GRAY_REF, label="理想校准线")
    ax.set_xlabel("名义置信度 / %")
    ax.set_ylabel("实际覆盖率 / %")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    psty.add_stat_box(ax, f"n 目标={len(ok)} ｜ 方法：带 σ 全链 CV ｜ z 分数法", loc="lower right")
    out = CALIB_DIR / "校准曲线.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_report(result):
    """校准报告 md——所有结论句由数值拼接（反模板红线）。"""
    from .ui_labels import zh
    lines = ["# L1 · 不确定度校准报告", "",
             f"- 生成时间：{result['generated_at']}",
             f"- 方法：{result['method']}",
             f"- 随机种子：{result['seed']}", "",
             "## 逐目标校准指标", "",
             "| 性能目标 | 有效 n | 1σ 覆盖率（期望≈68%） | 2σ 覆盖率（期望≈95%） | z 方差 | 校准因子 k | 判定 |",
             "|---|---|---|---|---|---|---|"]
    for t, info in result["targets"].items():
        if "cov_1sigma" in info:
            lines.append(f"| {zh(t)} | {info['n']} | {info['cov_1sigma']*100:.1f}% "
                         f"| {info['cov_2sigma']*100:.1f}% | {info['z_var']:.2f} "
                         f"| {info['factor']:.2f} | {info['verdict']} |")
        else:
            lines.append(f"| {zh(t)} | {info.get('n', 0)} | — | — | — | — | {info['verdict']} |")
    lines += ["", "## 逐目标结论", ""]
    for t, info in result["targets"].items():
        if "cov_1sigma" in info:
            lines.append(f"- **{zh(t)}**（n={info['n']}）：1σ 实际覆盖率 {info['cov_1sigma']*100:.1f}%、"
                         f"2σ 覆盖率 {info['cov_2sigma']*100:.1f}%，z 分数方差 {info['z_var']:.2f}，"
                         f"校准因子 k={info['factor']:.2f}——{info['verdict']}。"
                         f"下游不确定度展示已按该因子修正（σ_cal = k × σ_model）。")
        else:
            lines.append(f"- **{zh(t)}**：{info.get('note', '样本不足，无法评估。')}")
    lines += ["", "> 指标全部来自带 σ 的全链交叉验证（折外预测，非训练集拟合值）；",
              "> 校准因子已写入 models/calibration.json，工艺预测 / 逆向设计 / L3 效率基准中的 σ 自动乘以该因子。"]
    out = CALIB_DIR / "校准报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def load_factors():
    """读取校准因子 {目标: k}；文件不存在或目标缺失时因子=1（行为与 V1.6 完全一致）。"""
    try:
        with open(CALIB_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {t: float(info.get("factor", 1.0))
                for t, info in (data.get("targets") or {}).items() if isinstance(info, dict)}
    except Exception:
        return {}


def apply_calibration(unc_df, factors=None):
    """把 Stage 3 的 σ DataFrame 乘以校准因子（列名 <目标>_std）。"""
    if unc_df is None or unc_df.empty:
        return unc_df
    factors = factors if factors is not None else load_factors()
    if not factors:
        return unc_df
    out = unc_df.copy()
    for c in out.columns:
        t = c[:-4] if c.endswith("_std") else None
        if t and t in factors and factors[t] and np.isfinite(factors[t]) and factors[t] > 0:
            out[c] = out[c] * factors[t]
    return out


def calibration_status():
    """供状态面板：是否已校准 + 通过与否。"""
    try:
        with open(CALIB_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"done": False}
    tg = data.get("targets") or {}
    ok = [i for i in tg.values() if isinstance(i, dict) and "cov_1sigma" in i]
    good = [i for i in ok if GOOD_COV1[0] <= i["cov_1sigma"] <= GOOD_COV1[1]]
    return {"done": True, "n_targets": len(ok), "n_good": len(good),
            "generated_at": data.get("generated_at"),
            "factors": {t: i.get("factor") for t, i in tg.items() if isinstance(i, dict)}}
