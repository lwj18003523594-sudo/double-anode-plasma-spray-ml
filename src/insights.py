# -*- coding: utf-8 -*-
"""V1.5 数据洞察与实验反馈引擎。

三源分级（spec 三十四）：
  直接观测（experimental / literature）
  模型推断（OOF / SHAP / FI / 不确定性）
  优化建议（Pareto / 实验建议）

红线（spec 三十五/七十七）：
- model_prediction / optimization_candidate 绝不进入真实事实统计或正式训练；
- 措辞只允许"观察到 / 数据显示 / 存在相关趋势"，禁止无证据的"导致/决定/证明"；
- 覆盖描述只写"当前数据主要覆盖……"，禁止"最佳范围"。
"""
from pathlib import Path

import numpy as np
import pandas as pd

FACT_ORIGINS = {"experimental", "literature", "cfd"}  # 可参与事实统计的来源


def _fact_df(df):
    """只保留可当作事实的行（experimental/literature/CFD；data_origin 缺失视为事实）。

    model_prediction / optimization_candidate 行一律排除——绝不参与真实事实统计。
    存在 data_origin 列但没有事实行时返回空表（不回退全表）。
    """
    if "data_origin" not in df.columns:
        return df
    raw = df["data_origin"]
    low = raw.astype(str).str.lower().str.strip()
    m_fact = low.isin(FACT_ORIGINS)
    m_empty = raw.isna() | (low == "") | (low == "nan") | (low == "none")
    return df.loc[m_fact | m_empty]


def coverage_stats(df, schema):
    """数据覆盖统计：记录/批次/试样/重复/缺失率 + 各输入 min/Q1/median/Q3/max。"""
    fdf = _fact_df(df)
    input_cols = []
    for sec in ["structure_inputs", "process_inputs", "material_inputs"]:
        input_cols += [c for c in (schema.get(sec) or {}) if c in df.columns]
    rows = []
    for c in input_cols:
        s = pd.to_numeric(fdf[c], errors="coerce")
        if s.notna().sum() == 0:
            rows.append({"变量": c, "有效样本": 0, "缺失率(%)": 100.0})
            continue
        rows.append({
            "变量": c,
            "有效样本": int(s.notna().sum()),
            "缺失率(%)": round(float(s.isna().mean() * 100), 1),
            "min": round(float(s.min()), 2),
            "Q1": round(float(s.quantile(0.25)), 2),
            "median": round(float(s.median()), 2),
            "Q3": round(float(s.quantile(0.75)), 2),
            "max": round(float(s.max()), 2),
        })
    cov = pd.DataFrame(rows)
    rep = None
    if "sample_id" in fdf.columns:
        g = fdf.groupby("sample_id").size()
        rep = int((g >= 2).sum())
    return {"coverage": cov, "repeat_samples": rep, "n_fact": int(len(fdf))}


def coverage_text(df, schema):
    """客观覆盖描述："当前数据主要覆盖……"（禁止"最佳范围"）。"""
    stats = coverage_stats(df, schema)
    cov = stats["coverage"]
    parts = []
    for _, r in cov.iterrows():
        if r.get("median") is not None and pd.notna(r.get("median")):
            parts.append(f"{r['变量']} {r['min']:g}–{r['max']:g}（中位数 {r['median']:g}）")
    txt = f"当前数据主要覆盖：" + ("；".join(parts[:8]) if parts else "有效输入变量不足")
    sparse = cov[cov.get("缺失率(%)", pd.Series(dtype=float)) > 30] if "缺失率(%)" in cov else pd.DataFrame()
    if len(sparse):
        txt += "。缺失率较高的变量：" + "、".join(sparse["变量"].tolist()[:5]) + "（>30%，属稀疏区）"
    if stats["repeat_samples"]:
        txt += f"；存在重复测量试样 {stats['repeat_samples']} 个。"
    return txt


def correlation_report(df, schema, max_pairs=18):
    """正向实验规律：仅事实行（experimental/literature/CFD）的 Pearson + Spearman。

    无显著性检验 → 全文不使用统计意义上的"显著"。
    """
    fdf = _fact_df(df)
    ins, outs = [], []
    for sec in ["structure_inputs", "process_inputs", "material_inputs"]:
        ins += [c for c in (schema.get(sec) or {}) if c in fdf.columns]
    for sec in ["process_states", "melting_states", "defect_network", "performance_outputs"]:
        outs += [c for c in (schema.get(sec) or {}) if c in fdf.columns]
    rows = []
    for y in outs:
        sy = pd.to_numeric(fdf[y], errors="coerce")
        if sy.notna().sum() < 5:
            continue
        for x in ins:
            sx = pd.to_numeric(fdf[x], errors="coerce")
            m = sx.notna() & sy.notna()
            if m.sum() < 5 or sx[m].nunique() < 2:
                continue
            try:
                pr = float(np.corrcoef(sx[m], sy[m])[0, 1])
                sp = float(pd.Series(sx[m]).corr(pd.Series(sy[m]), method="spearman"))
            except Exception:
                continue
            if abs(pr) < 0.25 and abs(sp) < 0.25:
                continue
            rows.append({"输入": x, "输出": y, "Pearson": round(pr, 3), "Spearman": round(sp, 3),
                         "n": int(m.sum())})
        if len(rows) >= max_pairs:
            break
    rows.sort(key=lambda r: -abs(r["Pearson"]))
    return pd.DataFrame(rows[:max_pairs])


def model_key_factors(bundle, topn=8):
    """模型规律：各级模型特征重要性 + 跨折稳定性（若 fold_info 记录了 fi）。"""
    out = {}
    cv = bundle.get("chain_cv") or {}
    for stage in ["stage1", "stage2", "stage15", "stage3"]:
        models = bundle.get(f"{stage}_models") or {}
        if not models:
            continue
        for t, m in list(models.items()):
            pipe = m.get("model") if isinstance(m, dict) else m
            if pipe is None or not hasattr(pipe, "named_steps"):
                continue
            booster = pipe.named_steps.get("model")
            if booster is None or not hasattr(booster, "feature_importances_"):
                continue
            feats = bundle.get(f"{stage}_features") or []
            imp = pd.Series(booster.feature_importances_, index=feats).sort_values(ascending=False)
            out[f"{stage}:{t}"] = imp.head(topn)
    # 跨折稳定性（stage1 首目标）
    stability = None
    fi = [f.get("stage1_fi_top") for f in (cv.get("fold_info") or [])
          if isinstance(f, dict) and f.get("stage1_fi_top")]
    if len(fi) >= 3:
        counts = {}
        for top in fi:
            for name in top:
                counts[name] = counts.get(name, 0) + 1
        k = len(fi)
        stability = {name: f"{c}/{k} 折进入 Top5" for name, c in sorted(counts.items(),
                                                                        key=lambda x: -x[1])[:5]}
    return out, stability


def anomaly_review(df, bundle=None):
    """【值得复核的数据】：OOF 误差最大 / 重复测量离散大 / 相似条件性能差异大。不删除任何数据。"""
    fdf = _fact_df(df)
    items = []
    cv = (bundle or {}).get("chain_cv") or {}
    oof_all = cv.get("oof_predictions") or {}
    for stage in ["stage3", "stage2", "stage1"]:
        stage_oof = oof_all.get(stage)
        if stage_oof is None:
            continue
        for t, pred in (stage_oof.items() if hasattr(stage_oof, "items") else []):
            if t not in df.columns:
                continue
            p = pred.dropna()
            if len(p) < 5:
                continue
            yv = pd.to_numeric(df.loc[p.index, t], errors="coerce")
            err = (p - yv).abs()
            err = err.dropna()
            if err.empty:
                continue
            top = err.sort_values(ascending=False).head(3)
            for idx, e in top.items():
                eid = df.loc[idx, "experiment_id"] if "experiment_id" in df.columns else str(idx)
                items.append({"类型": "预测误差大", "对象": str(eid), "目标": t,
                              "详情": f"OOF 绝对误差 {e:.3g}（该目标全部折外误差的最大值之一）"})
            break
        if items:
            break
    # 重复测量离散
    try:
        from .research_utils import replicate_stats
        rs = replicate_stats(fdf, "sample_id") or replicate_stats(fdf, "experiment_id")
        if rs is not None and len(rs):
            rs = rs[rs["std"].notna()]
            big = rs[rs["std"] > 0].sort_values("CV%", ascending=False).head(3)
            for _, r in big.iterrows():
                if pd.notna(r["CV%"]) and r["CV%"] > 30:
                    items.append({"类型": "重复测量离散大", "对象": str(r.get("sample_id", r.get("experiment_id", ""))),
                                  "目标": r["字段"], "详情": f"重复测量 CV%={r['CV%']:.1f}（n={r['n']}）"})
    except Exception:
        pass
    return pd.DataFrame(items)


def data_gap_suggestions(df, schema, bundle=None):
    """【建议补充实验区域】：稀疏/边界/空白参数区 + 训练域边缘。只作建议，不替用户定方案。"""
    domain = (bundle or {}).get("training_domain") or {}
    rows = []
    for c, dom in domain.items():
        s = pd.to_numeric(df[c], errors="coerce").dropna() if c in df.columns else pd.Series(dtype=float)
        if len(s) < 5:
            continue
        lo, hi = dom["min"], dom["max"]
        # 密度：按 4 分位分箱
        try:
            bins = np.quantile(s, [0, .25, .5, .75, 1.0])
            bins = np.unique(bins)
            if len(bins) >= 3:
                hist, _ = np.histogram(s, bins=bins)
                if hist.min() <= max(1, 0.1 * len(s)):
                    rows.append({"区域": f"{c} 的低密度区间", "说明": "该参数区间样本数明显偏少（<10% 样本）"})
        except Exception:
            pass
        # 边界：Q1/Q3 外到 min/max 之间的样本比例
        edge = ((s < dom["q1"]) | (s > dom["q3"])).mean()
        if edge < 0.15:
            rows.append({"区域": f"{c} 的 {lo:.5g}–{hi:.5g} 边界区",
                         "说明": f"接近数据边界的样本仅约 {edge*100:.0f}%，边界区证据有限"})
    if len(rows) > 8:
        rows = rows[:8]
    return pd.DataFrame(rows)


def nearest_experiments(vals, df, input_cols=None, k=5):
    """正向预测历史参考：距离当前输入最近的 3–5 个真实实验/文献数据。"""
    fdf = _fact_df(df)
    if input_cols is None:
        input_cols = [c for c in fdf.columns
                      if c not in ("experiment_id", "batch_id", "sample_id", "data_origin")]
    cols = [c for c in input_cols if c in fdf.columns and c in vals]
    if not cols or fdf.empty:
        return pd.DataFrame()
    scale = {}
    for c in cols:
        s = pd.to_numeric(fdf[c], errors="coerce")
        rng_ = (s.max() - s.min()) if s.notna().sum() > 1 else 1.0
        scale[c] = rng_ if rng_ and rng_ > 0 else 1.0
    diff2 = np.zeros(len(fdf))
    for c in cols:
        s = pd.to_numeric(fdf[c], errors="coerce").fillna(0).to_numpy()
        v = float(vals.get(c, 0) or 0)
        diff2 += ((s - v) / scale[c]) ** 2
    fdf = fdf.assign(_dist=np.sqrt(diff2)).sort_values("_dist").head(k)
    out = []
    for _, r in fdf.iterrows():
        out.append({
            "experiment_id": str(r.get("experiment_id", "—")),
            "来源": str(r.get("data_origin", "experimental/literature")),
            "归一化距离": round(float(r["_dist"]), 3),
            "主要参数": "；".join(f"{c}={r[c]:.4g}" for c in cols[:4] if pd.notna(r.get(c))),
        })
    return pd.DataFrame(out)


def pareto_evidence(front, df, domain, bundle, input_cols, max_rows=10):
    """Pareto 候选证据支持：数据支持度 / 最近真实实验 / 局部密度 / 预测不确定性 / 参数差异。"""
    from .research_utils import pareto_row_support, support_label
    from .model_chain import predict_chain

    fdf = _fact_df(df)
    ev_rows = []
    front_n = front.head(max_rows)
    # 不确定性：对候选点做一次部署模型预测（GP std）
    unc_map = {}
    try:
        X = front_n[[c for c in bundle.get("x_cols", []) if c in front_n.columns]]
        _s, _d, _p, unc = predict_chain(X, bundle, return_uncertainty=True)
        for t in unc.columns:
            unc_map[t] = unc[t].to_numpy()
    except Exception:
        pass
    for i, (_, r) in enumerate(front_n.iterrows(), start=1):
        params = {c: float(r[c]) for c in input_cols if c in r and pd.notna(r[c])}
        sup = support_label(pareto_row_support(params, domain)) if domain else "未知（旧模型）"
        near = nearest_experiments(params, fdf, input_cols, k=1)
        near_txt, diffs = "—", []
        if not near.empty:
            nr = near.iloc[0]
            near_txt = str(nr["experiment_id"])
            row0 = fdf.iloc[int(near.index[0])] if near.index[0] in fdf.index else None
            if row0 is not None:
                diffs = []
                for c in list(params)[:3]:
                    try:
                        rv = float(row0[c])
                        dv = params[c] - rv
                        if abs(dv) > 1e-9:
                            diffs.append(f"{c} {'+' if dv>0 else ''}{dv:.4g}")
                    except Exception:
                        continue
        # 局部密度：训练点归一化距离 < 0.5 的数量
        density = "—"
        if cols_ok := [c for c in params if c in fdf.columns]:
            try:
                arr = np.zeros(len(fdf))
                for c in cols_ok:
                    s = pd.to_numeric(fdf[c], errors="coerce")
                    rng_ = (s.max() - s.min()) or 1.0
                    arr += ((s.fillna(0).to_numpy() - params[c]) / rng_) ** 2
                density = f"{int((np.sqrt(arr) < 0.5).sum())} 个邻近训练点（归一化半径 0.5 内）"
            except Exception:
                pass
        unc_txt = "—"
        if unc_map:
            uvals = [abs(float(v[i - 1])) for v in unc_map.values() if i - 1 < len(v)]
            if uvals:
                unc_txt = f"预测标准差均值 ≈ {np.mean(uvals):.3g}"
        ev_rows.append({
            "方案": f"P{i:02d}",
            "数据支持": sup,
            "最近真实实验": near_txt,
            "与最近实验的主要参数差异": "；".join(diffs[:3]) if diffs else "几乎一致",
            "局部数据密度": density,
            "预测不确定性": unc_txt,
        })
    return pd.DataFrame(ev_rows)


def recommendation_basis(ev_row, front_row=None, objectives=None):
    """【为什么推荐这个点】：只包含目标权衡/预测结果/数据覆盖/邻近实验/不确定性（不编机理）。"""
    lines = [
        "[优化建议] 该点为当前模型与约束下的 Pareto 非支配方案之一，代表目标之间的权衡，"
        "不是绝对最优工艺。",
        f"[直接数据] 数据支持程度：{ev_row.get('数据支持', '—')}；最近真实实验："
        f"{ev_row.get('最近真实实验', '—')}；{ev_row.get('局部数据密度', '')}。",
    ]
    diffs = ev_row.get("与最近实验的主要参数差异", "")
    if diffs and diffs != "几乎一致":
        lines.append(f"[直接数据] 与最近实验的主要参数差异：{diffs}（建议验证这些参数的实际影响）。")
    if ev_row.get("预测不确定性") and ev_row["预测不确定性"] != "—":
        lines.append(f"[模型推断] {ev_row['预测不确定性']}；预测值为模型输出，非实验实测。")
    lines.append("[需进一步验证] 推荐依据不包含材料机理判断；工艺可行性请结合设备条件与后续实验确认。")
    return "\n\n".join(lines)


def build_insight_report(ctx, df, schema, bundle, out_dir):
    """生成 Insight_Report.md + Insight_Evidence.xlsx（I 模块）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    cov = coverage_stats(df, schema)
    sections.append(("数据覆盖", coverage_text(df, schema)
                     + f"\n\n- 数据记录数 {cov['n_fact']}（事实统计仅含 experimental/literature/CFD 行）"
                     + (f"；重复测量试样 {cov['repeat_samples']} 个" if cov['repeat_samples'] else "")))
    corr = correlation_report(df, schema)
    if len(corr):
        top_txt = "；".join(f"{r['输入']}×{r['输出']}(Pearson {r['Pearson']:+.2f}, n={r['n']})"
                            for _, r in corr.head(6).iterrows())
        sections.append(("变量相关性（直接观测）",
                         "[直接数据] 数据显示以下输入—输出对存在相关趋势（未做显著性检验，"
                         "不构成因果关系）：" + top_txt))
    else:
        sections.append(("变量相关性（直接观测）", "当前数据量不足以计算稳定的相关趋势。"))
    factors, stability = model_key_factors(bundle)
    if factors:
        k0 = list(factors.keys())[0]
        tf = "、".join(factors[k0].head(5).index.tolist())
        s_txt = ""
        if stability:
            s_txt = " 跨折稳定性：" + "；".join(f"{k}（{v}）" for k, v in list(stability.items())[:3])
        sections.append(("模型规律（模型推断）",
                         f"[模型推断] 在当前模型中（{k0}），较稳定的重要变量包括：{tf}。{s_txt}"
                         " 模型重要性仅表征统计关联。"))
    else:
        sections.append(("模型规律（模型推断）", "当前无可用模型解释结果。"))
    anom = anomaly_review(df, bundle)
    if len(anom):
        a_txt = "；".join(f"{r['对象']}（{r['详情']}）" for _, r in anom.head(4).iterrows())
        sections.append(("值得复核的数据（直接观测 + 模型推断）",
                         "[需进一步验证] " + a_txt + "。以上仅为复核建议，不自动删除任何数据。"))
    gaps = data_gap_suggestions(df, schema, bundle)
    if len(gaps):
        g_txt = "；".join(f"{r['区域']}" for _, r in gaps.head(5).iterrows())
        sections.append(("建议补充实验区域（优化建议）",
                         "[优化建议] " + g_txt + "。仅为数据采集建议，最终实验方案由研究者确定。"))
    if ctx:
        sections.append(("熔融状态", f"Stage 1.5：{'已启用（数据具有熔融标签）' if ctx.get('stage15_enabled') else '未启用（缺少熔融标签，主链正常）'}"))
    demo = ctx.get("_demo") if ctx else False
    md = ["# Insight Report（数据洞察与实验反馈）", "",
          "> 三源分级：直接观测（experimental/literature/CFD）/ 模型推断 / 优化建议。",
          "> model_prediction 与 optimization_candidate 不参与真实事实统计。",
          "> 本报告为科研辅助分析，不得自动当作正式论文结论。", ""]
    for title, content in sections:
        md += [f"## {title}", "", content, ""]
    md_path = out_dir / "Insight_Report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    # 证据表
    with pd.ExcelWriter(out_dir / "Insight_Evidence.xlsx", engine="openpyxl") as w:
        cov["coverage"].to_excel(w, sheet_name="Coverage", index=False)
        corr.to_excel(w, sheet_name="Correlations", index=False)
        anom.to_excel(w, sheet_name="Anomaly_Review", index=False)
        gaps.to_excel(w, sheet_name="Data_Gaps", index=False)
    return md_path, out_dir / "Insight_Evidence.xlsx"
