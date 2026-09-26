# -*- coding: utf-8 -*-
"""V1.7 L4 · 物理一致性检验（核心价值验证第四层）。

回答的问题：模型学到的规律是否符合等离子喷涂物理？

方法（对应指令集阶段四，按平台三级链式架构适配）：
  对每条物理期望 {x → y, 期望方向}，用两类证据交叉验证：
  (a) 数据证据：训练数据中 x-y 的 Spearman 相关系数符号（直接观测，
      需 >= 8 对有效数据；仅 experimental / literature / CFD 行参与——沿用平台
      data_origin 事实统计铁律，模型预测行不参与）；
  (b) 模型证据：y 所在层级模型（Stage 1/2/3）对 x（或其上游预测列 pred_x）的
      PDP（部分依赖曲线）稳健线性拟合（Theil-Sen）斜率符号。
      —— 平台 Stage 3 为 GPR / RF、Stage 1/2 为 XGBoost；GPR 无法用 TreeSHAP，
      故统一采用 partial_dependence（对三类模型均严谨适用）。

  判定（V1.8 四级语义，两类证据同源、非独立机理验证）：
    SUPPORT       两类证据符号一致且符合期望方向（「支持」）
    INSUFFICIENT  仅一类证据支持（「证据不足」，不下结论）
    OPPOSITE      两类证据一致但均与期望相反，或互相矛盾（「局部相反」，附原因与补实验建议）
    OUT_OF_SCOPE  两类证据均缺失（「超出适用域/不可检验」）

  OPPOSITE 项必须给出可能原因（变化范围窄 / 共线性 / 过拟合或噪声）与针对性补实验建议。

设计铁律（V1.6 继承）：不修改任何既有模块；结论句由真实数值拼接；固定种子 42。
"""
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from .config import ROOT
from .features import add_physics_features
from .status_binding import bind_identity, freshness

EXPECT_FILE = ROOT / "config" / "physics_expectations.yaml"
PHYS_DIR = ROOT / "outputs" / "physics"
RESULT_FILE = PHYS_DIR / "物理一致性结果.json"
SEED = 42

# V1.8（方案 §7）：L4 判定改为四级「合理性诊断」语义，且明确两类证据同源
# （Spearman 与模型 PDP 基于同一份数据/同一模型，不构成两组独立机理验证）。
VERDICT_ZH = {
    "SUPPORT": "支持",
    "INSUFFICIENT": "证据不足",
    "OPPOSITE": "局部相反",
    "OUT_OF_SCOPE": "超出适用域",
}
# 与 V1.7 判定词的映射（历史结果读取用）
_LEGACY_MAP = {"PASS": "SUPPORT", "WEAK": "INSUFFICIENT", "FAIL": "OPPOSITE",
               "UNVERIFIED": "OUT_OF_SCOPE"}

# 期望方向 → 符号
_DIR_SIGN = {"正": +1, "负": -1, "+": +1, "-": -1, "positive": +1, "negative": -1}
# 参与数据证据的事实行（沿用 insights 的 data_origin 铁律）
_FACT_ORIGINS = {"experimental", "literature", "cfd", ""}


def load_expectations():
    with open(EXPECT_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("expectations") or []


def _fact_rows(df):
    """直接观测证据只取事实行（model_prediction / optimization_candidate 不参与）。"""
    if "data_origin" in df.columns:
        return df[df["data_origin"].astype(str).str.lower().isin(_FACT_ORIGINS)]
    return df


def _layer_design_matrix(df, schema, bundle, layer):
    """构造指定层级模型的设计矩阵（与 train_chain 的构造规则一致）。

    layer: "stage1" | "stage2" | "stage3"
    Stage 2/3 的上游状态特征使用全链 CV 的 OOF 预测（bundle["chain_cv"]），
    与训练时防泄漏口径一致。
    """
    groups_cfg = {k: list(v.keys()) for k, v in schema.items()}
    x_cols = [c for c in groups_cfg["structure_inputs"] + groups_cfg["process_inputs"]
              if c in df.columns]
    baseX = add_physics_features(df[x_cols])
    material_cols = []
    for col, spec in (schema.get("material_inputs") or {}).items():
        if col in df.columns and not spec.get("categorical"):
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() > 0 and col not in baseX.columns:
                material_cols.append(col)
    if material_cols:
        baseX = pd.concat([baseX, df[material_cols].apply(pd.to_numeric, errors="coerce")], axis=1)
    if layer == "stage1":
        return baseX
    oof = ((bundle.get("chain_cv") or {}).get("oof_predictions") or {})
    parts = [baseX]
    oof1 = oof.get("stage1")
    if oof1 is not None and len(oof1):
        parts.append(oof1.reindex(df.index).add_prefix("pred_"))
    if bundle.get("stage15_enabled") and oof.get("stage15") is not None:
        parts.append(oof["stage15"].reindex(df.index).add_prefix("pred_"))
    if layer == "stage2":
        return pd.concat(parts, axis=1)
    oof2 = oof.get("stage2")
    if oof2 is not None and len(oof2):
        parts.append(oof2.reindex(df.index).add_prefix("pred_"))
    return pd.concat(parts, axis=1)


def _locate_target_model(bundle, y):
    """返回 (layer_key, model, X_layer_constructor 或 None)；y 无模型 → (None, None, None)。"""
    for layer, models in (("stage1", bundle.get("stage1_models")),
                          ("stage2", bundle.get("stage2_models")),
                          ("stage3", bundle.get("stage3_models"))):
        if models and y in models:
            m = models[y]["model"] if isinstance(models[y], dict) else models[y]
            return layer, m, layer
    return None, None, None


def _pdp_slope(model, X, feature_name):
    """对指定特征列做 PDP 并返回稳健线性拟合（Theil-Sen）斜率。

    返回 (slope, n_grid)；不可计算 → (None, 0)。
    """
    if feature_name not in X.columns:
        return None, 0
    from sklearn.inspection import partial_dependence
    Xv = X.dropna(subset=[feature_name])
    if len(Xv) < 10:
        return None, 0
    try:
        res = partial_dependence(model, Xv, features=[feature_name], kind="average")
    except Exception:
        return None, 0
    grid = np.asarray(res["grid_values"][0], dtype=float)
    avg = np.asarray(res["average"][0], dtype=float)
    ok = np.isfinite(grid) & np.isfinite(avg)
    grid, avg = grid[ok], avg[ok]
    if len(grid) < 4 or np.ptp(grid) <= 0:
        return None, int(len(grid))
    # Theil-Sen 稳健线性拟合（对 PDP 的单调性判断不受个别波动影响）
    slope = float(sps.theilslopes(avg, grid).slope)
    return slope, int(len(grid))


def _diagnose_fail(df, schema, x, y, sign_exp):
    """FAIL 项原因诊断：变化范围窄 / 共线性 / 疑似过拟合或噪声，附补实验建议。"""
    notes, advice = [], []
    xs = pd.to_numeric(df[x], errors="coerce").dropna()
    specs = {**{k: v for k, v in (schema.get("process_inputs") or {}).items()},
             **{k: v for k, v in (schema.get("structure_inputs") or {}).items()}}
    if x in specs and "min" in (specs[x] or {}):
        lo, hi = float(specs[x]["min"]), float(specs[x]["max"])
        if hi > lo:
            frac = (xs.max() - xs.min()) / (hi - lo)
            if frac < 0.15:
                notes.append(f"{x} 在当前数据中仅覆盖允许范围的 {frac*100:.0f}%（变化范围窄）")
                advice.append(f"建议在 {x} 的低/高两端（接近 {lo:g} 与 {hi:g}）各补 2–3 组实验")
    # 共线性：x 与其他数值输入的最大 |Spearman|
    cand_cols = [c for c in (schema.get("process_inputs") or {})
                 if c in df.columns and c != x]
    worst_corr, worst_col = 0.0, None
    for c in cand_cols[:25]:
        pair = df[[x, c]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 8:
            rho = abs(sps.spearmanr(pair[x], pair[c]).statistic)
            if np.isfinite(rho) and rho > worst_corr:
                worst_corr, worst_col = float(rho), c
    if worst_col and worst_corr >= 0.85:
        notes.append(f"{x} 与 {worst_col} 高度相关（|Spearman|={worst_corr:.2f}，共线性干扰方向判定）")
        advice.append(f"建议设计 {x} 与 {worst_col} 解耦的实验点（固定其一、扫描另一个）")
    if not notes:
        notes.append("未检出范围窄或强共线性，疑似过拟合或数据噪声主导")
        advice.append(f"建议围绕 {x}→{y} 关系补充在 {x} 靠近两端的中等重复实验（每点 2–3 次重复）")
    return notes, advice


def run_physics_check(df, bundle, schema, *, expectations=None):
    """执行物理一致性检验。

    返回 {"generated_at", "n_total", "verdicts": [逐条判定 dict], "summary"}；
    同时落盘结果 json 与 报告 md。
    """
    if df is None or bundle is None:
        raise ValueError("物理一致性检验需要已加载的数据与已训练的模型。")
    exps = expectations if expectations is not None else load_expectations()
    if not exps:
        raise ValueError("config/physics_expectations.yaml 中没有可检验的期望条目。")
    facts = _fact_rows(df)

    verdicts = []
    for i, e in enumerate(exps):
        x, y = e.get("x"), e.get("y")
        exp_sign = _DIR_SIGN.get(str(e.get("dir", "")).strip().lower())
        item = {"idx": i, "x": x, "y": y, "dir": e.get("dir"), "basis": e.get("basis", ""),
                "spearman": None, "n_pairs": 0, "pdp_slope": None, "pdp_grid": 0,
                "pdp_feature": None, "verdict": "UNVERIFIED", "notes": [], "advice": []}
        if not x or not y or exp_sign is None:
            item["notes"] = ["条目配置不完整（缺 x/y/dir）。"]
            verdicts.append(item)
            continue

        # ---- (a) 数据证据：Spearman（仅事实行）----
        pair = facts[[x, y]].apply(pd.to_numeric, errors="coerce").dropna() \
            if (x in facts.columns and y in facts.columns) else pd.DataFrame()
        if len(pair) >= 8:
            rho = sps.spearmanr(pair[x], pair[y]).statistic
            if np.isfinite(rho) and abs(rho) > 0.05:
                item["spearman"] = round(float(rho), 4)
                item["n_pairs"] = int(len(pair))
        elif x not in df.columns or y not in df.columns:
            item["notes"].append(f"变量 {x if x not in df.columns else y} 不在当前数据中。")

        # ---- (b) 模型证据：y 所在层级模型的 PDP 稳健斜率 ----
        layer, model, _ = _locate_target_model(bundle, y)
        if layer:
            X_layer = _layer_design_matrix(df, schema, bundle, layer)
            feats = bundle.get(f"{layer}_features") or list(X_layer.columns)
            feat_name = x if x in feats else (f"pred_{x}" if f"pred_{x}" in feats else None)
            if feat_name:
                slope, ng = _pdp_slope(model, X_layer, feat_name)
                item["pdp_feature"] = feat_name
                item["pdp_slope"] = round(slope, 6) if slope is not None else None
                item["pdp_grid"] = ng
            else:
                item["notes"].append(f"{x} 不在 {y} 的模型输入特征中（无法做 PDP）。")
        else:
            item["notes"].append(f"{y} 没有对应层级模型（当前数据未建模该目标）。")

        # ---- 判定（V1.8 四级语义：两类证据同源，非独立机理验证）----
        s_data = int(np.sign(item["spearman"])) if item["spearman"] is not None else None
        s_model = int(np.sign(item["pdp_slope"])) if item["pdp_slope"] not in (None, 0) else \
            (None if item["pdp_slope"] is None else 0)
        if s_data is None and s_model is None:
            item["verdict"] = "OUT_OF_SCOPE"
        elif s_data == exp_sign and s_model == exp_sign:
            item["verdict"] = "SUPPORT"
        elif s_data == -exp_sign and s_model == -exp_sign:
            item["verdict"] = "OPPOSITE"
        elif (s_data is not None and s_model is not None and s_data == -s_model):
            item["verdict"] = "OPPOSITE"
            item["notes"].append("数据证据与模型证据方向互相矛盾（需排查数据质量与模型拟合）。")
        else:
            # 仅一类支持；另一类缺失或为 0（无明确方向）——证据不足，不下结论
            item["verdict"] = "INSUFFICIENT"
        # 适用范围：当前数据中 x 的实际覆盖区间（V1.8 方案 §7：每条规则记录适用范围）
        if x in df.columns:
            xv = pd.to_numeric(df[x], errors="coerce").dropna()
            if len(xv):
                item["scope"] = f"x 覆盖区间 [{xv.min():.3g}, {xv.max():.3g}]（n={len(xv)}）"
        if item["verdict"] == "OPPOSITE":
            notes, advice = _diagnose_fail(df, schema, x, y, exp_sign)
            item["notes"] += notes
            item["advice"] = advice
        verdicts.append(item)

    n_map = {"SUPPORT": 0, "INSUFFICIENT": 0, "OPPOSITE": 0, "OUT_OF_SCOPE": 0}
    for v in verdicts:
        n_map[v["verdict"]] += 1
    n_checkable = n_map["SUPPORT"] + n_map["INSUFFICIENT"] + n_map["OPPOSITE"]
    summary = {**n_map, "n_checkable": n_checkable,
               "pass_ratio": round((n_map["SUPPORT"] + n_map["INSUFFICIENT"]) / n_checkable, 4)
               if n_checkable else None}

    result = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "seed": SEED, "verdicts": verdicts, "summary": summary,
              "verdict_semantics": "V1.8 四级：支持/证据不足/局部相反/超出适用域；"
                                   "Spearman 与模型 PDP 同源，非独立机理验证",
              "evidence_caveat": "本检验为物理合理性诊断（V1.8 措辞），"
                                "不构成对真实机理的独立验证证据。"}
    # V1.8：身份绑定 + 规则版本（期望表内容 hash）
    import hashlib as _hl
    rule_version = _hl.md5(EXPECT_FILE.read_bytes()).hexdigest()[:8]
    bind_identity(result, bundle=bundle, rule_version=rule_version,
                  note="规则版本 rule_v8_" + rule_version)

    PHYS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _write_report(result)
    return result


def _write_report(result):
    from .ui_labels import zh
    s = result["summary"]
    lines = ["# L4 · 物理合理性诊断报告（V1.8 语义）", "",
             f"- 生成时间：{result['generated_at']}",
             f"- 绑定：{result.get('binding', {}).get('model_version', '未知模型')} / "
             f"数据 {str(result.get('binding', {}).get('dataset_hash') or '')[:8] or '未知'}"
             f" / {result.get('binding', {}).get('note', '')}",
             "- 证据 A（直接观测）：事实行 Spearman 相关系数符号（model_prediction / "
             "optimization_candidate 行不参与）",
             "- 证据 B（模型推断）：y 所在层级模型 PDP 部分依赖曲线的 Theil-Sen 稳健斜率符号",
             "- **证据声明：A 与 B 基于同一份数据与同一模型，不构成两组独立机理验证**（V1.8）",
             "- 判定（四级）：一致且符合期望 → 支持；仅一类支持 → 证据不足；"
             "相反/矛盾 → 局部相反；均缺 → 超出适用域",
             "", "## 逐条判定表", "",
             "| # | 物理期望 | 期望方向 | Spearman ρ (n) | PDP 斜率 (特征) | 判定 | 适用范围 |",
             "|---|---|---|---|---|---|---|"]
    for v in result["verdicts"]:
        sp_txt = (f"{v['spearman']:+.2f} (n={v['n_pairs']})" if v["spearman"] is not None
                  else "不可用")
        pd_txt = (f"{v['pdp_slope']:+.3g} ({v['pdp_feature']}, 网格 {v['pdp_grid']} 点)"
                  if v["pdp_slope"] is not None else "不可用")
        vd = VERDICT_ZH.get(v["verdict"], v["verdict"])
        lines.append(f"| {v['idx']+1} | {zh(v['x'])} → {zh(v['y'])} | {v['dir']} "
                     f"| {sp_txt} | {pd_txt} | **{vd}** | {v.get('scope', '—')} |")
    lines += ["", "## 判定依据（物理机理假设，待数据检验）", ""]
    for v in result["verdicts"]:
        lines.append(f"- {zh(v['x'])} → {zh(v['y'])}（期望{v['dir']}）：{v['basis']}")
    fails = [v for v in result["verdicts"] if v["verdict"] == "OPPOSITE"]
    if fails:
        lines += ["", "## 局部相反项原因分析与补实验建议", ""]
        for v in fails:
            lines.append(f"### {zh(v['x'])} → {zh(v['y'])}（适用范围：{v.get('scope', '—')}）")
            for n_ in v["notes"]:
                lines.append(f"- 可能原因：{n_}")
            for a in v["advice"]:
                lines.append(f"- 补实验建议：{a}")
    lines += ["", "## 总结", "",
              f"- 可检验条目 {s['n_checkable']} 条：支持 {s['SUPPORT']} ｜ "
              f"证据不足 {s['INSUFFICIENT']} ｜ 局部相反 {s['OPPOSITE']} ｜ "
              f"超出适用域 {s['OUT_OF_SCOPE']}"]
    if s["n_checkable"]:
        ratio = s["pass_ratio"]
        concl = ("物理合理性总体良好（支持+证据不足占可检验条目的 {:.0f}%）。".format(ratio * 100)
                 if ratio >= 0.7 else
                 "物理合理性待改善：建议优先排查「局部相反」条目（见上）后补充实验。")
        lines.append(f"- {concl}")
    lines += ["", "> 全部符号与数值来自真实计算（Spearman / Theil-Sen 斜率），无人工设定结论；",
              "> 本检验为合理性诊断，不宣称发现普适因果规律（期望方向本身是待检验假设）。"]
    (PHYS_DIR / "物理合理性诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


def physics_status(current_model_version=None, current_dataset_hash=None):
    """供状态面板/总报告（V1.8：含时效性判定）。"""
    try:
        with open(RESULT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        fr = freshness(data, current_model_version, current_dataset_hash)
        # 旧版结果文件的判定词自动映射到 V1.8 语义
        summary = data.get("summary") or {}
        if "PASS" in summary and "SUPPORT" not in summary:
            summary = {(_LEGACY_MAP.get(k, k)): v for k, v in summary.items()}
        return {"done": True, "summary": summary,
                "generated_at": data.get("generated_at"),
                "freshness": fr["state"], "freshness_note": fr["note"],
                "report": "物理合理性诊断报告.md"}
    except Exception:
        return {"done": False}
