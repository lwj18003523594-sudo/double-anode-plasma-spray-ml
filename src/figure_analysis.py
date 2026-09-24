# -*- coding: utf-8 -*-
"""V1.5 图表科研解读引擎（模板化证据约束生成，禁止伪造）。

为每张主要科研图生成 8 段式 *_Analysis.md：
① 现象描述（只描述图内可观测模式：上升/下降/先升后降/峰值/谷值/平台/拐点/离散/组间差异/权衡）
② 数据证据（n / batch / mean / std / CV% / min-max / R² / RMSE / MAE / Pearson / Spearman 等）
③ 模型解释（"在当前模型中……"，只用 FI/SHAP/PDP/CV/OOF 证据）
④ 可能物理机理（证据门槛：无文献/显微/熔融实测 → 只能"可能与……有关/仍需验证"）
⑤ 当前局限性（样本量/外推/熔融是否模型预测/是否 Demo 等自动检查）
⑥ 中文论文表述草稿（数值与趋势保真，不加未支持的强机理结论）
⑦ English manuscript draft
⑧ 简洁结论

证据标签：[直接数据] [模型推断] [文献支持] [可能机理] [优化建议] [需进一步验证]
红线：没有显著性检验禁止使用"显著"；无文献时禁止生成引用；不伪造 DOI；
      Demo 数据必须标记"不可用于正式科研结论"；机理假设不得写成事实。
"""
from datetime import datetime

# 禁用词（无检验支撑时）
_FORBIDDEN_STAT = ["显著", "significant"]
# 结论强度上限词
_ABSOLUTE_WORDS = ["绝对最优", "完全证明", "必然导致", "最佳工艺", "最优工艺方案"]

TAG_DATA = "[直接数据]"
TAG_MODEL = "[模型推断]"
TAG_LIT = "[文献支持]"
TAG_MECH = "[可能机理]"
TAG_OPT = "[优化建议]"
TAG_TODO = "[需进一步验证]"


def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return "—"


def describe_trend(x, y):
    """根据 (x, y) 序列识别可观测模式（只报告图中真实存在的形态）。

    返回中文描述；样本不足或无变化时如实说明。不做任何拟合美化。
    """
    try:
        pairs = sorted([(float(a), float(b)) for a, b in zip(x, y)
                        if a == a and b == b])  # 过滤 NaN
    except (TypeError, ValueError):
        return "数据不足以判断趋势。"
    n = len(pairs)
    if n < 3:
        return f"有效数据点仅 {n} 个，不足以判断趋势。"
    ys = [b for _, b in pairs]
    ymin, ymax = min(ys), max(ys)
    if ymax - ymin < 1e-12:
        return "图中各点数值基本不变（近似平台区）。"
    # 单调性（允许少量噪声点：比例阈值 0.75）
    up = sum(1 for i in range(1, n) if ys[i] > ys[i - 1])
    dn = sum(1 for i in range(1, n) if ys[i] < ys[i - 1])
    spread = _fmt(ymax - ymin)
    if up / (n - 1) >= 0.75:
        return f"整体呈上升趋势（总变化幅度约 {spread}）。"
    if dn / (n - 1) >= 0.75:
        return f"整体呈下降趋势（总变化幅度约 {spread}）。"
    # 单峰 / 单谷
    imax, imin = ys.index(max(ys)), ys.index(min(ys))
    if 0 < imax < n - 1 and ys[imax] - min(ys[0], ys[-1]) > 0.25 * (ymax - ymin):
        return "呈现先升后降形态，存在峰值区。"
    if 0 < imin < n - 1 and max(ys[0], ys[-1]) - ys[imin] > 0.25 * (ymax - ymin):
        return "呈现先降后升形态，存在谷值区。"
    return f"未表现为单调趋势，数据点较为离散（极差约 {spread}）。"


def analyze_figure(kind, *, target=None, stage=None, n=None, batch_count=None,
                   r2=None, rmse=None, mae=None, pearson=None, spearman=None,
                   top_features=None, stability=None, pareto_count=None,
                   trend_desc=None, class_counts=None, demo=False,
                   sample_warning=None, extrapolation=False, melting_from_model=False,
                   has_microstructure=False, literature_refs=None, model_type="XGBoost",
                   cv_method=None, extra_notes=None):
    """生成 8 段式科研解读（纯文本模板 + 调用方传入的真实统计量）。

    所有数值必须由调用方从真实数据/模型计算后传入；本函数绝不编造数字。
    literature_refs: [(title_or_source, year_or_doi)]，仅当 Source_Metadata 提供时非空。
    """
    zh_t = target or "该目标"
    sections = []
    lit_txt = ""
    if literature_refs:
        lit_txt = "；".join(f"{t}（{d}）" for t, d in literature_refs[:3])
    else:
        lit_txt = "当前数据未提供文献来源信息（Source_Metadata），因此不作文献引用。"

    # ① 现象描述
    phen = trend_desc if trend_desc else "见图中各数据点的实际分布。"
    s1 = f"{TAG_DATA} 图为{zh_t}的相关结果。{phen}"
    if class_counts:
        cc = "；".join(f"{k} {v} 个" for k, v in class_counts.items())
        s1 += f" 类别构成：{cc}。"
    sections.append(("一、现象描述", s1))

    # ② 数据证据
    ev = []
    if n is not None:
        ev.append(f"样本数 n={n}")
    if batch_count is not None:
        ev.append(f"独立批次 {batch_count} 个")
    if r2 is not None:
        ev.append(f"R²={_fmt(r2)}")
    if rmse is not None:
        ev.append(f"RMSE={_fmt(rmse)}")
    if mae is not None:
        ev.append(f"MAE={_fmt(mae)}")
    if pearson is not None:
        ev.append(f"Pearson r={_fmt(pearson)}")
    if spearman is not None:
        ev.append(f"Spearman ρ={_fmt(spearman)}")
    if pareto_count is not None:
        ev.append(f"Pareto 非支配解 {pareto_count} 个")
    s2 = (TAG_DATA + " " + "，".join(ev) + "。" if ev
          else TAG_DATA + " 当前未提供汇总统计量，请结合原始数据表核对。")
    if cv_method:
        s2 += f" 评价方式：{cv_method}（折外预测，非训练集拟合值）。"
    sections.append(("二、数据证据", s2))

    # ③ 模型解释
    s3 = f"{TAG_MODEL} 在当前模型（{model_type}）中，"
    if top_features:
        tf = "、".join(top_features[:5])
        s3 += f"重要性较高的输入变量为：{tf}。"
        if stability:
            s3 += f" 其中 {stability} 在交叉验证各折中表现稳定，属于模型中较稳定的重要变量。"
        s3 += "模型响应趋势仅表征输入—输出的统计关联，不构成因果关系。"
    else:
        s3 += "模型解释细节见特征重要性 / SHAP / PDP 结果。"
    sections.append(("三、模型解释", s3))

    # ④ 可能物理机理（证据门槛）
    mech_evidence = []
    if has_microstructure:
        mech_evidence.append("显微组织/缺陷实测数据")
    if melting_from_model is False and kind in ("tv_map", "melting_dist", "melting_perf"):
        mech_evidence.append("熔融状态实测标签")
    if literature_refs:
        mech_evidence.append("文献报道")
    if mech_evidence:
        s4 = (TAG_MECH + f" 结合{'、'.join(mech_evidence)}，图中的趋势"
              f"可能与粒子加热/冷却动力学、熔融铺展行为及涂层形成过程有关，"
              f"但具体机理{'已有部分文献支持' if literature_refs else '仍需进一步验证'}。")
    else:
        s4 = (TAG_MECH + " 当前缺乏显微组织、熔融实测与文献证据，"
              f"仅可推测该趋势可能与粒子热历史及熔融沉积过程有关（推测），仍需进一步验证。")
    sections.append(("四、可能物理机理", s4))

    # ⑤ 当前局限性
    lims = []
    if demo:
        lims.append("**当前为模拟演示数据，全部结果仅用于软件功能验证，不可用于正式科研结论**")
    if n is not None and n < 10:
        lims.append(f"样本量极少（n={n}），结果仅适合功能测试或探索性分析")
    elif n is not None and n <= 25:
        lims.append(f"样本量较小（n={n}，文献小样本），结论强度有限")
    if batch_count is not None and batch_count < 4:
        lims.append(f"独立批次仅 {batch_count} 个，分组交叉验证稳定性有限")
    if extrapolation:
        lims.append("结果涉及训练数据边界或外推区域，可靠性降低")
    if melting_from_model:
        lims.append("熔融状态来自模型预测，而非直接实验测量")
    if r2 is not None and r2 < 0.3:
        lims.append(f"交叉验证 R²={_fmt(r2)}，模型预测能力有限，请谨慎解读")
    if sample_warning:
        lims.append(sample_warning)
    s5 = (TAG_TODO + " " + ("；".join(lims) + "。") if lims
          else TAG_TODO + " 当前未检测到明显局限性因素。")
    sections.append(("五、当前局限性", s5))

    # ⑥ 中文论文表述草稿
    zh_stat = "；".join(ev[:6]) if ev else ""
    zh_draft = (f"图 X 展示了{zh_t}的结果。{phen}"
                + (f"（{zh_stat}）" if zh_stat else "")
                + " 结果基于" + ("模拟演示数据（仅软件功能验证）。" if demo else "当前实验/文献数据。")
                + " 在当前模型中，上述趋势反映输入变量与输出之间的统计关联；"
                  "受样本量与数据覆盖限制，机理层面的解释仍需进一步实验验证。")
    sections.append(("六、中文论文表述草稿", zh_draft))

    # ⑦ English manuscript draft
    en_draft = (
        f"Figure X presents the results for {zh_t}. {phen} "
        + (f"({zh_stat}). " if zh_stat else "")
        + ("Data are synthetic demo values for software validation only. " if demo else
           "Data are from the current experimental/literature dataset. ")
        + "Within the current model, the observed trend reflects a statistical association "
          "between input variables and the output; mechanistic interpretation requires "
          "further experimental verification.")
    sections.append(("七、English manuscript draft", en_draft))

    # ⑧ 简洁结论
    strength = "探索性" if (demo or (n is not None and n <= 25) or (r2 is not None and r2 < 0.3)) \
        else "有一定数据与模型支持"
    s8 = (f"{TAG_DATA} 综合而言，{zh_t}的图中规律为{strength}结果：{phen} "
          f"该描述不构成因果结论，也不代表最优工艺选择。")
    sections.append(("八、简洁结论", s8))

    # 文献支持附录（不伪造）
    lit_section = f"{TAG_LIT} 文献支持：{lit_txt}"

    # 禁用词自检（无检验支撑时移除"显著"；绝对化措辞直接拒绝）
    body = [f"# 科研解读（自动生成草稿）",
            f"> 生成时间：{datetime.now().isoformat(timespec='seconds')} ｜ "
            f"证据标签：直接数据 / 模型推断 / 文献支持 / 可能机理 / 优化建议 / 需进一步验证",
            f"> 本解读由平台基于图表真实数据自动生成，为科研辅助草稿，不得直接当作正式论文结论。", ""]
    for title, content in sections:
        body.append(f"## {title}")
        body.append(content)
        body.append("")
    body.append(f"## 附：来源与引用")
    body.append(lit_section)
    if extra_notes:
        body.append("")
        body.extend(extra_notes)
    md = "\n".join(body)

    for w in _ABSOLUTE_WORDS:
        if w in md:
            md = md.replace(w, "值得进一步评估的")
    # "显著"仅在无统计检验语境出现时降级为"较明显"
    if "p<" not in md and "p <" not in md and "显著性" not in md:
        md = md.replace("显著", "较明显")
    return md


def write_analysis_md(fig_path_base, md_text):
    """将解读写入同名 *_Analysis.md（fig_path_base 为不含扩展名的主文件路径）。"""
    from pathlib import Path
    p = Path(str(fig_path_base) + "_Analysis.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md_text, encoding="utf-8")
    return p


def summarize_analyses(analysis_dir):
    """汇总目录内全部 *_Analysis.md 为 Figure_Analysis_Summary.md。"""
    from pathlib import Path
    d = Path(analysis_dir)
    items = sorted(d.glob("*_Analysis.md")) if d.exists() else []
    lines = ["# Figure Analysis Summary（图表科研解读汇总）", "",
             f"> 共 {len(items)} 份图表解读；每份均为证据约束的自动草稿，"
             "不构成正式论文结论。", ""]
    for p in items:
        first = ""
        try:
            txt = p.read_text(encoding="utf-8")
            for ln in txt.splitlines():
                if ln.startswith("## 八、"):
                    first = ln.replace("## 八、", "").strip()
                    break
            concl = ""
            started = False
            for ln in txt.splitlines():
                if started and ln.strip():
                    concl = ln.strip()
                    break
                if ln.startswith("## 八、"):
                    started = True
            lines.append(f"## {p.stem}")
            lines.append(f"- 简洁结论：{concl}")
            lines.append("")
        except Exception:
            continue
    out = d / "Figure_Analysis_Summary.md"
    d.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
