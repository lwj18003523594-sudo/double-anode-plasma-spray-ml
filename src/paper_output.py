# -*- coding: utf-8 -*-
"""科研结果输出与论文图表引擎（V1.3）

原则：
- 模型评价图（parity / residual / R² 总览）只使用全链分组交叉验证的 OOF 预测，
  禁止使用训练集拟合值；
- 数据是什么就画什么：不删异常点、不隐藏负 R²、不改 Pareto 解；
- 所有图带同名 metadata JSON 与绘图数据文件，可回溯到模型与数据版本；
- 路径全部 pathlib，macOS / Windows 通用。
"""
import hashlib
import json
import platform as _platform
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ROOT
from .features import add_physics_features
from .model_chain import predict_chain, _XGB_HYPERPARAMS
from .optimize import random_pareto_search
from .paper_labels import (paper_label, axis_label, stage_paper_label,
                           misc_label, target_slug)
from .paper_style import (apply_paper_style, save_figure,
                          fig_single, fig_double, fig_mid, add_stat_box)
# V1.6（P0-5）：论文图与网页端同源色板——hex 只定义于 src/plot_style.py
from .plot_style import (GRAY_INK, GRAY_MUTED, GRAY_REF, GRAY_BORDER, GRAY_PAPER,
                         BLUE_SIGNAL, BLUE_MID, BLUE_LIGHT, PINK_ACCENT, PINK_MID,
                         QUAL_CYCLE)
from .research_utils import (pareto_row_support, support_label, log_event)
from .modes import MELTING_CLASS_ZH

SCHEMA_SECTIONS = {
    "meta": ("基础信息", "Identity"),
    "structure_inputs": ("双阳极结构参数", "Structure input"),
    "process_inputs": ("工艺参数", "Process input"),
    "process_states": ("射流与粒子状态", "Process state (Stage 1 output)"),
    "defect_network": ("涂层缺陷网络", "Defect network (Stage 2 output)"),
    "performance_outputs": ("涂层性能", "Performance (Stage 3 output)"),
}

STAGE_OF_SECTION = {"process_states": "stage1", "defect_network": "stage2",
                    "performance_outputs": "stage3"}

FIG_DPI_NOTE = "PNG 600 dpi"


class PaperOutputError(Exception):
    """用户可读的输出失败原因。"""


def ensure_output_tree():
    """建立 outputs/ 标准目录骨架（幂等）。"""
    for sub in ["paper_figures/zh", "paper_figures/en", "paper_tables", "figure_data",
                "model_evaluation", "explainability", "optimization", "validation", "reports"]:
        (ROOT / "outputs" / sub).mkdir(parents=True, exist_ok=True)


def _version_info():
    out = {"python": _platform.python_version()}
    for mod, key in [("sklearn", "scikit-learn"), ("xgboost", "xgboost"),
                     ("pandas", "pandas"), ("numpy", "numpy")]:
        try:
            out[key] = __import__(mod).__version__
        except Exception:
            out[key] = "unknown"
    return out


class PaperExporter:
    """一次输出会话：绑定 数据/模型/语言/格式，写入独立运行目录。"""

    def __init__(self, df, schema, bundle, lang="zh", formats=("png", "svg"),
                 demo=False, run_dir=None, run_prefix="Run", prefix=None,
                 registry=None, objective_cfg=None):
        self.df = df
        self.schema = schema
        self.bundle = bundle
        # None preserves the legacy configured export. A supplied configuration
        # belongs to this run and must be used consistently in every E figure/table.
        self.objective_cfg = objective_cfg
        self.lang = lang
        self.formats = ("png_svg" if f == "png_svg" else f for f in formats)
        self.formats = tuple(self.formats)
        self.demo = demo
        # V1.6（P0-2）：可选证据注册表——传入时八段式解读注册 fig.* 证据；
        # 默认 None，旧调用路径零修改向后兼容。
        self.registry = registry
        ensure_output_tree()
        if run_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = ROOT / "outputs" / "reports" / f"{run_prefix}_{ts}"
        self.run_dir = Path(run_dir)
        for sub in ["figures", "tables", "data", "metadata", "analysis", "insights"]:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        # V1.5：图表科研解读目录（*_Analysis.md；不改变 A–H 文件结构）
        self.analysis_dir = self.run_dir / "analysis"
        # 论文图样式按语言一次性应用（本实例创建的所有图生效）
        apply_paper_style(lang)

        self.cv = bundle.get("chain_cv") or {}
        self.metrics = self.cv.get("metrics") or {}
        self.oof = self.cv.get("oof_predictions") or {}
        # V1.4：文件名前缀可按研究模式指定（DEMO_/DUAL_/APS_/CASCADE_/LIT_）；
        # 不指定时保持 V1.3 行为（demo→DEMO_，否则无前缀），A–G 模块命名不受影响。
        self.demo_prefix = prefix if prefix is not None else ("DEMO_" if demo else "")
        self._manifest = None
        self._captions = []
        # 版本化文件名后缀（任务 10）：Dataset + Model 标签，图表/数据/元数据一致
        dv = bundle.get("dataset_hash")
        mv = bundle.get("model_version")
        self._ver_tag = ""
        if dv:
            self._ver_tag += f"_DS{str(dv)[:8]}"
        if mv:
            self._ver_tag += f"_MV{str(mv).split('_')[-1]}"

    # ---------- 基础工具 ----------
    def _stem(self, code, name):
        slug = target_slug(name, lang=self.lang) if name else ""
        prefix = self.demo_prefix
        base = f"{prefix}{code}_{slug}" if slug else f"{prefix}{code}"
        return base + self._ver_tag

    def _save_fig(self, fig, code, name=None):
        stem = self._stem(code, name)
        saved = save_figure(fig, stem, self.run_dir / "figures",
                            formats=self.formats, lang=self.lang, demo=self.demo)
        return saved, stem

    def _save_data(self, df_data, code, name=None):
        """绘图原始数据 CSV（保留完整精度）。"""
        stem = self._stem(code, name)
        path = self.run_dir / "data" / f"{stem}.csv"
        if self.demo:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write("# DEMO SYNTHETIC DATA - FOR SOFTWARE TESTING ONLY\n")
            df_data.to_csv(path, index=False, encoding="utf-8-sig", mode="a")
        else:
            df_data.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _save_table(self, df_table, code, name=None, note=None):
        """论文表格 Excel：Demo 模式在首个工作表顶部加警示行；数值按论文惯例取整。"""
        stem = self._stem(code, name)
        path = self.run_dir / "tables" / f"{stem}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            if note:
                pd.DataFrame({"说明": [note]}).to_excel(w, sheet_name="注意", index=False)
            df_table.to_excel(w, sheet_name="Table", index=False)
        return path

    def _save_analysis(self, stem_base, md_text):
        """V1.5：图表科研解读（*_Analysis.md）；失败静默（不影响主图输出）。"""
        try:
            from .figure_analysis import write_analysis_md
            return write_analysis_md(self.analysis_dir / stem_base, md_text)
        except Exception as e:
            log_event("analysis_save_error", f"{stem_base}: {e}")
            return None

    def _analysis_for(self, kind, stem_base, eid=None, **kw):
        """生成并保存 8 段式科研解读（证据约束模板，见 figure_analysis）。

        V1.6：可选 eid——self.registry 存在时注册 fig.* 证据并追加证据溯源段。
        """
        try:
            from .figure_analysis import analyze_figure
            kw.setdefault("demo", self.demo)
            if self.registry is not None and eid:
                kw["eid"] = eid
                kw["registry"] = self.registry
            md = analyze_figure(kind, **kw)
            return self._save_analysis(stem_base, md)
        except Exception as e:
            log_event("analysis_gen_error", f"{stem_base}: {e}")
            return None

    def _fig_meta(self, code, stem, target=None, model_type=None, data_file=None,
                  extra=None, r2=None, rmse=None, mae=None, caption=None):
        manifest = self.reproducibility_manifest()
        meta = {
            "figure_name": stem,
            "dataset_version": manifest["dataset_version"],
            "model_version": manifest["model_version"],
            "target": target,
            "model_type": model_type,
            "cv_method": self.cv.get("method", "none"),
            "sample_count": manifest["sample_count"],
            "batch_count": manifest["batch_count"],
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "generation_time": datetime.now().isoformat(timespec="seconds"),
            "language": self.lang,
            "demo_data": self.demo,
            "data_file": data_file,
        }
        if caption:
            meta["caption_suggestion"] = caption
            self._captions.append({"figure": stem, "caption": caption})
        if extra:
            meta.update(extra)
        path = self.run_dir / "metadata" / f"{stem}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path

    def _cap(self, kind, target=None, model_type=None, n=None):
        """客观图注草稿：只描述变量/模型/CV方式/数据类型，无机理与因果结论。"""
        lang = self.lang
        literature = self.demo_prefix.startswith("QUICK_LIT_")
        dtype = ("模拟演示数据（仅软件测试）" if self.demo else
                 "文献数据（工艺体系以原文为准）" if literature else "真实实验数据")
        dtype_en = ("synthetic demo data (software testing only)" if self.demo else
                    "published literature data" if literature else "real experimental data")
        t = paper_label(target, lang) if target else ""
        cv = self.cv.get("method") or ("未记录交叉验证方式" if lang == "zh" else
                                        "cross-validation method unavailable")
        if kind == "parity":
            return (f"图为{t}的实验值与预测值对比。预测值来自{cv}的折外（OOF）预测；"
                    f"模型类型：{model_type}；样本数 n={n}；数据类型：{dtype}。虚线为 y=x 参考线。"
                    if lang == "zh" else
                    f"Experimental vs. predicted {t}. Predicted values are out-of-fold predictions from "
                    f"{cv}; model: {model_type}; n={n}; data: {dtype_en}. Dashed line: y = x.")
        if kind == "residual":
            return (f"图为{t}的残差图（预测值 − 实验值），预测值来自{cv}。"
                    f"模型类型：{model_type}；数据类型：{dtype}。"
                    if lang == "zh" else
                    f"Residuals (predicted − experimental) of {t} from {cv}; "
                    f"model: {model_type}; data: {dtype_en}.")
        if kind == "overview":
            return (f"各级模型各预测目标在{cv}下的决定系数 R² 汇总（包含全部目标，未作筛选）。数据类型：{dtype}。"
                    if lang == "zh" else
                    f"Cross-validated R² of all targets across model stages ({cv}); all targets shown unfiltered; "
                    f"data: {dtype_en}.")
        if kind == "importance":
            return (f"图为{t}的 XGBoost 内置特征重要性（Top N）。模型训练采用{cv}。数据类型：{dtype}。"
                    if lang == "zh" else
                    f"XGBoost built-in feature importance for {t} (Top N); training used {cv}; data: {dtype_en}.")
        if kind == "shap":
            return (f"图为{t}的 SHAP 特征贡献分布（XGBoost）。数据类型：{dtype}。"
                    if lang == "zh" else
                    f"SHAP summary of feature contributions for {t} (XGBoost); data: {dtype_en}.")
        if kind == "pdp":
            return (f"图为{t}对所选输入特征的部分依赖图（PDP），显示模型平均预测响应趋势，"
                    f"仅表征模型关联规律，不代表因果关系。数据类型：{dtype}。"
                    if lang == "zh" else
                    f"Partial dependence of {t} on the selected feature (model response trend, "
                    f"association only, not causation); data: {dtype_en}.")
        if kind == "correlation":
            return (f"图为关键变量间的皮尔逊相关系数矩阵，仅表征统计相关性，不代表因果关系。数据类型：{dtype}。"
                    if lang == "zh" else
                    "Pearson correlation matrix of key variables (statistical association only, "
                    f"not causation); data: {dtype_en}.")
        if kind == "completeness":
            return (f"图为各关键变量的数据完整率。数据类型：{dtype}；样本数 n={n}。"
                    if lang == "zh" else
                    f"Data completeness of key variables; data: {dtype_en}; n={n}.")
        if kind == "distributions":
            return (f"图为主要工艺参数的数据分布（直方图）。数据类型：{dtype}。"
                    if lang == "zh" else
                    f"Distributions of main process parameters (histograms); data: {dtype_en}.")
        if kind == "pareto":
            if self.objective_cfg is not None:
                names = list(self.objective_cfg.get("objectives", {}))
                return (f"图为已确认优化目标 {', '.join(names)} 的 Pareto 非支配解；"
                        f"数据类型：{dtype}；候选点为模型预测，不是验证实验。")
            return (f"图为多目标 Pareto 非支配解在预测孔隙率—预测结合强度平面上的分布，"
                    f"颜色表示预测耦合损伤。结果来自当前三级代理模型与给定约束下的候选搜索，"
                    f"不构成单一最佳方案。数据类型：{dtype}。"
                    if lang == "zh" else
                    f"Pareto non-dominated solutions in the predicted porosity–bond strength plane, "
                    f"colored by predicted coupled damage; from candidate search under the current model "
                    f"and constraints; no single optimum implied; data: {dtype_en}.")
        if kind == "ranges":
            return (f"图为当前模型与约束条件下 Pareto 解的工艺参数范围（归一化至配置允许范围），"
                    f"不代表普适的工艺区间结论。数据类型：{dtype}。"
                    if lang == "zh" else
                    "Pareto solution parameter ranges under the current model and constraints "
                    "(normalized to configured bounds); not an absolute optimum; data: " + dtype_en + ".")
        if kind == "uncertainty":
            return (f"图为部署模型（使用全部数据训练）对{t}的预测值 ± 1 标准差与实验值对比。"
                    f"该图非交叉验证评价。数据类型：{dtype}。"
                    if lang == "zh" else
                    f"Deployment-model (trained on all data) prediction ± 1 std for {t} versus "
                    f"experimental values; not a cross-validated evaluation; data: {dtype_en}.")
        if kind == "chain":
            return (f"图为中值输入条件下结构/工艺参数 → 射流与粒子状态 → 缺陷网络 → 涂层性能的全链预测示意，"
                    f"三级性能含预测标准差。数据类型：{dtype}。"
                    if lang == "zh" else
                    "Full-chain prediction illustration at median inputs: structure/process → jet & particle "
                    f"state → defect network → performance (stage 3 with prediction std); data: {dtype_en}.")
        if kind == "tv_map":
            return (f"图为颗粒温度—速度状态图，数据点按数据本身的熔融状态类别标签着色"
                    f"（无标签时仅显示散点，不划分熔融区域）。数据类型：{dtype}。"
                    if lang == "zh" else
                    "Particle temperature-velocity map; points colored by melting state class labels "
                    "from the data itself (scatter only when labels absent); data: " + dtype_en + ".")
        if kind == "melting_distribution":
            return ("图为颗粒熔融状态类别分布（样本数与百分比），类别来自数据/文献本身提供的标签。"
                    f"数据类型：{dtype}。"
                    if lang == "zh" else
                    "Melting state class distribution (counts and percentages); class labels from the "
                    f"data/literature; data: {dtype_en}.")
        if kind == "melting_performance":
            return ("图为颗粒熔融状态模型（Stage 1.5）的交叉验证评价：连续目标为实验值-预测值对比（OOF），"
                    "分类目标为混淆矩阵（OOF）。数据类型：{dtype}。".replace("{dtype}", dtype)
                    if lang == "zh" else
                    "Stage 1.5 melting-state model evaluation via cross-validated OOF predictions: "
                    "parity for continuous targets, confusion matrix for classification targets; data: "
                    + dtype_en + ".")
        if kind == "melting_corr":
            return ("图为熔融指数与缺陷/性能指标之间的关联趋势，仅描述统计关联，不代表因果关系。"
                    f"数据类型：{dtype}。"
                    if lang == "zh" else
                    "Association trends between melting index and defect/performance indicators "
                    f"(statistical association only, not causation); data: {dtype_en}.")
        if kind == "melting_importance":
            return ("图为 Stage 1.5 熔融状态模型的 XGBoost 内置特征重要性（Top N），"
                    f"反映模型所用的统计关联结构。数据类型：{dtype}。"
                    if lang == "zh" else
                    "XGBoost built-in feature importance of the Stage 1.5 melting-state model (Top N); "
                    f"data: {dtype_en}.")
        return ""

    def write_captions(self):
        path = self.run_dir / "Fig_Captions.md"
        lines = ["# 图注建议（Caption Suggestions）", "",
                 "> 仅客观描述变量、模型、交叉验证方式与数据类型，供撰写论文时参考；", 
                 "> 不包含机理解释或因果结论。", ""]
        for c in self._captions:
            lines.append(f"## {c['figure']}")
            lines.append("")
            lines.append(c["caption"])
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ---------- 可复现信息 ----------
    def reproducibility_manifest(self):
        if self._manifest is not None:
            return self._manifest
        df = self.df
        dataset_hash = hashlib.sha256(df.to_csv(index=True).encode("utf-8")).hexdigest()
        schema_str = json.dumps(self.schema, sort_keys=True, ensure_ascii=False)
        self._manifest = {
            "dataset_version": "auto_" + dataset_hash[:8],
            "dataset_hash": dataset_hash,
            "model_version": self.bundle.get("model_version", "unknown"),
            "schema_version": "v1_" + hashlib.sha256(schema_str.encode()).hexdigest()[:8],
            "date": datetime.now().isoformat(timespec="seconds"),
            "python_version": _platform.python_version(),
            "package_versions": _version_info(),
            "random_seed": self.bundle.get("seed", 42),
            "cv_method": self.cv.get("method", "none"),
            "sample_count": int(len(df)),
            "batch_count": int(df["batch_id"].nunique()) if "batch_id" in df else None,
            "targets": {
                s: list(self.bundle.get(f"{s}_models", {}).keys())
                for s in ["stage1", "stage2", "stage3"]},
            "features": {
                s: self.bundle.get(f"{s}_features", [])
                for s in ["stage1", "stage2", "stage3"]},
            "demo_data": self.demo,
        }
        return self._manifest

    def write_reproducibility_manifest(self):
        path = self.run_dir / "metadata" / "reproducibility_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.reproducibility_manifest(), f, ensure_ascii=False, indent=2)
        return path

    def write_results_summary(self, pareto_stats=None, pareto_ranges=None, top_features=None):
        """客观结果摘要：只汇总数字，不生成机理结论。"""
        man = self.reproducibility_manifest()
        lines = ["# 结果摘要（Results Summary）", "",
                 f"- 生成时间：{man['date']}",
                 f"- 数据版本：{man['dataset_version']}　模型版本：{man['model_version']}",
                 f"- 样本数：{man['sample_count']}　喷涂批次数：{man['batch_count'] if man['batch_count'] is not None else '未提供'}",
                 f"- 交叉验证方法：{man['cv_method']}",
                 f"- 模式：{'DEMO 模拟数据（仅测试软件功能，不得用于科研结论）' if self.demo else '文献数据（不可直接验证双阳极性能）' if self.demo_prefix.startswith('QUICK_LIT_') else '真实实验数据'}",
                 "", "## 模型评价（折外交叉验证，非训练集拟合值）", ""]
        for stage in ["stage1", "stage2", "stage3"]:
            lines.append(f"### {stage_paper_label(stage, self.lang)}")
            lines.append("")
            lines.append("| 目标 | 模型类型 | R² | RMSE | MAE |")
            lines.append("|---|---|---|---|---|")
            for t, met in (self.metrics.get(stage) or {}).items():
                kind = self._model_kind(stage, t)
                lines.append(f"| {paper_label(t, self.lang)} | {kind} | "
                             f"{met['R2']:.4f} | {met['RMSE']:.4f} | {met['MAE']:.4f} |")
            lines.append("")
        if top_features:
            lines.append("## XGBoost 特征重要性（Top 10，一级模型首个目标）")
            lines.append("")
            for f, v in top_features[:10]:
                lines.append(f"- {paper_label(f, self.lang)}: {v:.4f}")
            lines.append("")
        if pareto_stats:
            lines.append("## 多目标 Pareto 优化")
            lines.append("")
            lines.append(f"- 候选点：{pareto_stats['n_candidates']}　"
                         f"满足约束：{pareto_stats['n_feasible']}　"
                         f"Pareto 非支配解：{pareto_stats['n_pareto']}")
            lines.append("")
            lines.append("当前模型与约束条件下的 Pareto 工艺参数范围：")
            lines.append("")
            lines.append("| 工艺参数 | min | 25% | median | 75% | max |")
            lines.append("|---|---|---|---|---|---|")
            for row in (pareto_ranges or []):
                g = lambda *keys: next((row[k] for k in keys if k in row), "-")
                lines.append(f"| {g('工艺参数', 'parameter')} | {g('最小值', 'min')} | "
                             f"{g('25%')} | {g('中位数', 'median')} | {g('75%')} | {g('最大值', 'max')} |")
            lines.append("")
        lines.append("> 本摘要仅汇总客观计算结果，不含机理结论；因果性表述需由实验验证支持。")
        path = self.run_dir / "Results_Summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _model_kind(self, stage, target):
        if stage in ("stage1", "stage2"):
            return "XGBoost"
        return self.bundle["stage3_models"].get(target, {}).get("kind", "-")

    def _zip_results(self, zip_dir):
        """将运行目录压缩为同名 zip。"""
        zip_dir = Path(zip_dir)
        zip_path = zip_dir.parent / f"{zip_dir.name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(zip_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(zip_dir.parent))
        return zip_path

    # ---------- 模块 A：数据集概况 ----------
    def table_a1(self):
        rows = []
        for sec, (zh_sec, en_sec) in SCHEMA_SECTIONS.items():
            for col, spec in self.schema.get(sec, {}).items():
                s = pd.to_numeric(self.df[col], errors="coerce") if col in self.df else None
                rows.append({
                    "变量层级": zh_sec if self.lang == "zh" else en_sec,
                    "内部变量名": col,
                    "中文名称": paper_label(col, "zh", with_unit=False),
                    "英文名称": paper_label(col, "en", with_unit=False),
                    "单位": spec.get("unit", "-"),
                    "角色": "输入" if sec in ("structure_inputs", "process_inputs")
                            else ("中间量" if sec == "process_states" else "输出"),
                    "是否模型输入": "是" if sec in ("structure_inputs", "process_inputs") else "否",
                    "是否模型输出": "是" if sec in ("process_states", "defect_network", "performance_outputs") else "否",
                    "数据数量": int(len(self.df)) if s is None else int(s.notna().sum()),
                    "缺失数量": 0 if s is None else int(s.isna().sum()),
                })
        df_t = pd.DataFrame(rows)
        note = "DEMO SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY" if self.demo else None
        path = self._save_table(df_t, "Table_A1_Variable_Definitions", note=note)
        return path, df_t

    def _key_fields(self):
        cols = []
        for sec in ["structure_inputs", "process_inputs", "process_states",
                    "defect_network", "performance_outputs"]:
            cols += [c for c in self.schema.get(sec, {}) if c in self.df.columns]
        return cols

    def fig_a1_completeness(self):
        cols = self._key_fields()
        comp = {c: float(pd.to_numeric(self.df[c], errors="coerce").notna().mean() * 100)
                for c in cols}
        items = sorted(comp.items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=fig_mid(max(3.0, 0.32 * len(items) + 1.0)))
        names = [paper_label(c, self.lang) for c, _ in items]
        vals = [v for _, v in items]
        ax.barh(names, vals, color=BLUE_SIGNAL, alpha=0.85)
        ax.set_xlabel(f"{axis_label('completeness', self.lang)} (%)")
        ax.set_xlim(0, 105)
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8.5)
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_A1_Data_Completeness")
        data = self._save_data(pd.DataFrame({"column": [c for c, _ in items],
                                             "completeness_pct": vals}),
                               "Data_A1_Data_Completeness")
        meta = self._fig_meta("A", stem, data_file=Path(data).name,
                              caption=self._cap("completeness", n=len(self.df)))
        return saved + [data, meta], stem

    def fig_a2_distributions(self):
        cols = [c for c in self.schema.get("process_inputs", {}) if c in self.df.columns]
        n = len(cols)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=fig_double(2.4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for i, c in enumerate(cols):
            ax = axes[i]
            vals = pd.to_numeric(self.df[c], errors="coerce").dropna()
            ax.hist(vals, bins=20, color=BLUE_SIGNAL, alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.set_title(paper_label(c, self.lang), fontsize=10.5)
            ax.tick_params(labelsize=8.5)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_A2_Input_Distribution")
        data = self._save_data(self.df[cols].apply(pd.to_numeric, errors="coerce"),
                               "Data_A2_Input_Distribution")
        meta = self._fig_meta("A", stem, data_file=Path(data).name,
                              caption=self._cap("distributions"))
        return saved + [data, meta], stem

    def fig_a3_correlation(self):
        cols = [c for c in self._key_fields()
                if pd.to_numeric(self.df[c], errors="coerce").notna().sum() >= 3]
        X = self.df[cols].apply(pd.to_numeric, errors="coerce")
        corr = X.corr()
        fig, ax = plt.subplots(figsize=fig_double(max(5.0, 0.42 * len(cols) + 1.5)))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        short = [paper_label(c, self.lang, with_unit=False) for c in cols]
        ax.set_xticks(range(len(cols)), short, rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(cols)), short, fontsize=7.5)
        fig.colorbar(im, ax=ax, shrink=0.8, label=axis_label("相关性", self.lang))
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_A3_Correlation_Matrix")
        data = self._save_data(corr, "Data_A3_Correlation_Matrix")
        meta = self._fig_meta("A", stem, data_file=Path(data).name,
                              extra={"note": "correlation is not causation"},
                              caption=self._cap("correlation"))
        return saved + [data, meta], stem

    # ---------- 模块 B：模型评价（严格 OOF / grouped-CV） ----------
    def _oof_series(self, stage, target):
        stage_oof = self.oof.get(stage)
        oof = stage_oof.get(target) if stage_oof is not None else None
        if oof is None:
            raise PaperOutputError(
                "当前模型缺少全链分组交叉验证的逐行 OOF 预测记录，请先重新训练模型（V1.2+ 版本）。")
        pred = oof.dropna()
        actual = self.df.loc[pred.index, target]
        mask = actual.notna()
        return actual[mask], pred[mask]

    def _stats_of(self, stage, target):
        met = (self.metrics.get(stage) or {}).get(target)
        if met:
            return met["R2"], met["RMSE"], met["MAE"]
        a, p = self._oof_series(stage, target)
        from .model_chain import _metrics
        r = _metrics(a, p)
        return r["R2"], r["RMSE"], r["MAE"]

    def table_b1(self):
        rows = []
        for stage in ["stage1", "stage2", "stage3"]:
            for t, met in (self.metrics.get(stage) or {}).items():
                a, p = self._oof_series(stage, t)
                rows.append({
                    "模型层级" if self.lang == "zh" else "Stage": stage_paper_label(stage, self.lang),
                    "目标指标" if self.lang == "zh" else "Target": paper_label(t, self.lang),
                    "模型类型" if self.lang == "zh" else "Model": self._model_kind(stage, t),
                    "样本数" if self.lang == "zh" else "n": int(len(a)),
                    "独立batch数" if self.lang == "zh" else "Batches":
                        int(self.df.loc[a.index, "batch_id"].nunique()) if "batch_id" in self.df else None,
                    "CV方法" if self.lang == "zh" else "CV method":
                        "GroupKFold(batch_id)" if self.cv else "-",
                    "R²": round(float(met["R2"]), 4),
                    "RMSE": round(float(met["RMSE"]), 4),
                    "MAE": round(float(met["MAE"]), 4),
                })
        df_t = pd.DataFrame(rows)
        note = "DEMO SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY" if self.demo else None
        path = self._save_table(df_t, "Table_B1_Model_Performance", note=note)
        return path, df_t

    def fig_parity(self, stage, target):
        actual, pred = self._oof_series(stage, target)
        r2, rmse, mae = self._stats_of(stage, target)
        # V1.6（P0-5）：Nature 单栏 89mm 版式；1:1 中性灰虚线 + 信号蓝散点 + 统一注脚框
        fig, ax = plt.subplots(figsize=fig_single(3.6))
        lo = float(min(actual.min(), pred.min()))
        hi = float(max(actual.max(), pred.max()))
        pad = (hi - lo) * 0.06 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=GRAY_REF,
                linewidth=1.0, label="y = x", zorder=1)
        ax.scatter(actual, pred, s=26, c=BLUE_SIGNAL, alpha=0.75, edgecolors="none", zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(axis_label("actual", self.lang))
        ax.set_ylabel(axis_label("predicted", self.lang))
        ax.set_title(f"{misc_label('title_parity', self.lang)} — {paper_label(target, self.lang)}",
                     fontsize=11.5)
        cv_txt = self.cv.get("method") or "—"
        stats_txt = f"R² = {r2:.3f}\nRMSE = {rmse:.3g}\nMAE = {mae:.3g}\nn = {len(actual)}\nCV: {cv_txt}"
        add_stat_box(ax, stats_txt, loc="upper left")
        fig.tight_layout()
        saved, stem = self._save_fig(fig, f"Fig_B{self._stage_bno(stage)}_{target_slug(target)}_Parity")
        code = f"Data_B{self._stage_bno(stage)}"
        stem_d = f"{self.demo_prefix}{code}_{target_slug(target)}_Parity"
        data = self.run_dir / "data" / f"{stem_d}.csv"
        df_data = pd.DataFrame({
            "experiment_id": self.df.loc[actual.index, "experiment_id"].values if "experiment_id" in self.df else np.array(actual.index),
            "batch_id": self.df.loc[actual.index, "batch_id"].values if "batch_id" in self.df else "",
            "target": target,
            "actual": actual.values,
            "predicted": pred.values,
            "residual": (pred - actual).values,
        })
        if self.demo:
            with open(data, "w", encoding="utf-8-sig") as f:
                f.write("# DEMO SYNTHETIC DATA - FOR SOFTWARE TESTING ONLY\n")
            df_data.to_csv(data, index=False, encoding="utf-8-sig", mode="a")
        else:
            df_data.to_csv(data, index=False, encoding="utf-8-sig")
        meta = self._fig_meta("B", Path(saved[0]).stem, target=target,
                              model_type=self._model_kind(stage, target),
                              data_file=data.name, r2=r2, rmse=rmse, mae=mae,
                              extra={"plot": "parity", "oof_only": True},
                              caption=self._cap("parity", target, self._model_kind(stage, target), n=len(actual)))
        self._analysis_for("parity", Path(saved[0]).stem,
                           eid=f"fig.parity_{stage}_{target_slug(target).lower()}",
                           target=target,
                           n=int(len(actual)), r2=r2, rmse=rmse, mae=mae,
                           model_type=self._model_kind(stage, target),
                           cv_method="分组交叉验证 OOF（折外预测）",
                           trend_desc="散点围绕 y = x 参考线分布，反映折外预测与实验值的总体一致性。")
        return saved + [data, meta], stem

    def fig_residual(self, stage, target):
        actual, pred = self._oof_series(stage, target)
        resid = pred - actual
        r2, rmse, mae = self._stats_of(stage, target)
        # V1.6：Nature 单栏版式；零线中性灰、残差点强调粉、统一注脚框直接标注
        fig, ax = plt.subplots(figsize=fig_single(3.2))
        ax.axhline(0, linestyle="--", color=GRAY_REF, linewidth=1.0, zorder=1)
        ax.scatter(pred, resid, s=26, c=PINK_ACCENT, alpha=0.75, edgecolors="none", zorder=2)
        ax.set_xlabel(axis_label("predicted", self.lang))
        ax.set_ylabel(axis_label("residual", self.lang))
        ax.set_title(f"{misc_label('title_residual', self.lang)} — {paper_label(target, self.lang)}",
                     fontsize=11.5)
        cv_txt = self.cv.get("method") or "—"
        add_stat_box(ax, f"RMSE = {rmse:.3g}\nMAE = {mae:.3g}\nn = {len(actual)}\nCV: {cv_txt}",
                     loc="upper right")
        fig.tight_layout()
        saved, stem = self._save_fig(fig, f"Fig_B{self._stage_bno(stage)}_{target_slug(target)}_Residual")
        stem_d = f"{self.demo_prefix}Data_B{self._stage_bno(stage)}_{target_slug(target)}_Residual"
        data = self.run_dir / "data" / f"{stem_d}.csv"
        df_data = pd.DataFrame({
            "experiment_id": self.df.loc[actual.index, "experiment_id"].values if "experiment_id" in self.df else np.array(actual.index),
            "batch_id": self.df.loc[actual.index, "batch_id"].values if "batch_id" in self.df else "",
            "target": target,
            "actual": actual.values,
            "predicted": pred.values,
            "residual": resid.values,
        })
        if self.demo:
            with open(data, "w", encoding="utf-8-sig") as f:
                f.write("# DEMO SYNTHETIC DATA - FOR SOFTWARE TESTING ONLY\n")
            df_data.to_csv(data, index=False, encoding="utf-8-sig", mode="a")
        else:
            df_data.to_csv(data, index=False, encoding="utf-8-sig")
        meta = self._fig_meta("B", Path(saved[0]).stem, target=target,
                              model_type=self._model_kind(stage, target),
                              data_file=data.name, r2=r2, rmse=rmse, mae=mae,
                              extra={"plot": "residual", "oof_only": True},
                              caption=self._cap("residual", target, self._model_kind(stage, target)))
        return saved + [data, meta], stem

    def _stage_bno(self, stage):
        return {"stage1": 1, "stage2": 2, "stage3": 3}[stage]

    def fig_b_overview(self):
        rows = []
        for stage in ["stage1", "stage2", "stage3"]:
            for t in (self.metrics.get(stage) or {}):
                rows.append((stage, t, float(self.metrics[stage][t]["R2"])))
        rows.sort(key=lambda r: r[2])
        fig, ax = plt.subplots(figsize=fig_mid(max(3.0, 0.36 * len(rows) + 1.2)))
        names = [f"{stage_paper_label(s, self.lang)} · {paper_label(t, self.lang)}" for s, t, _ in rows]
        vals = [v for _, _, v in rows]
        # V1.6：Stage 三色改用统一分类循环色（蓝/灰/粉家族，禁 seaborn 残留色）
        _stage_color = {"stage1": QUAL_CYCLE[0], "stage2": QUAL_CYCLE[1], "stage3": QUAL_CYCLE[2]}
        colors = [_stage_color[s] for s, _, _ in rows]
        ax.barh(names, vals, color=colors, alpha=0.88)
        ax.axvline(0, color=GRAY_REF, linewidth=0.8)
        ax.set_xlabel(axis_label("R2overview", self.lang))
        ax.set_xlim(min(0.0, min(vals) - 0.1), 1.05)
        for i, v in enumerate(vals):
            ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_B_Model_Performance_Overview")
        data = self._save_data(pd.DataFrame({"stage": [s for s, _, _ in rows],
                                             "target": [t for _, t, _ in rows],
                                             "R2": vals}),
                               "Data_B_Model_Performance_Overview")
        meta = self._fig_meta("B", Path(saved[0]).stem, data_file=Path(data).name,
                              extra={"plot": "overview", "oof_only": True,
                                     "note": "不同量纲的 RMSE 不在同一纵轴比较"},
                              caption=self._cap("overview"))
        return saved + [data, meta], stem

    # ---------- 模块 C：可解释性 ----------
    def _stage_X(self, stage):
        x_cols = self.bundle["x_cols"]
        base = add_physics_features(self.df[x_cols])
        if stage == "stage1":
            feats = self.bundle["stage1_features"]
            return base.reindex(columns=feats)
        if stage == "stage2":
            oof1 = self.oof.get("stage1")
            if oof1 is None:
                raise PaperOutputError("缺少一级模型 OOF 预测记录，请先重新训练模型。")
            X = pd.concat([base, oof1.add_prefix("pred_")], axis=1)
            feats = self.bundle["stage2_features"]
            return X.reindex(columns=feats)
        raise PaperOutputError("三级模型为 GP/RF，不适用该 XGBoost 解释方式。")

    def _xgb_pipe(self, stage, target):
        if stage not in ("stage1", "stage2"):
            raise PaperOutputError("当前模型暂未启用该 SHAP 解释方式（三级模型为 GP/RF）。")
        pipe = self.bundle.get(f"{stage}_models", {}).get(target)
        if pipe is None:
            raise PaperOutputError("该目标当前没有已训练模型。")
        return pipe

    def fig_c_importance(self, stage, target, topn=15):
        pipe = self._xgb_pipe(stage, target)
        booster = pipe.named_steps["model"]
        feats = self.bundle[f"{stage}_features"]
        imp = pd.DataFrame({"feature": feats, "importance": booster.feature_importances_})
        imp = imp.sort_values("importance", ascending=False).head(int(topn)).iloc[::-1]
        fig, ax = plt.subplots(figsize=fig_single(max(2.8, 0.34 * len(imp) + 1.0)))
        ax.barh([paper_label(f, self.lang) for f in imp["feature"]], imp["importance"],
                color=BLUE_SIGNAL, alpha=0.88)
        ax.set_xlabel(axis_label("importance", self.lang))
        fig.tight_layout()
        saved, stem = self._save_fig(fig, f"Fig_C_{target_slug(target)}_Importance_Top{int(topn)}")
        data = self._save_data(pd.DataFrame({
            "feature": imp["feature"].values,
            "feature_label_zh": [paper_label(f, "zh", with_unit=False) for f in imp["feature"]],
            "feature_label_en": [paper_label(f, "en", with_unit=False) for f in imp["feature"]],
            "importance": imp["importance"].values,
        }), f"Data_C_{target_slug(target)}_Importance_Top{int(topn)}")
        meta = self._fig_meta("C", Path(saved[0]).stem, target=target, model_type="XGBoost",
                              data_file=Path(data).name, extra={"plot": "feature_importance",
                                                                "topn": int(topn)},
                              caption=self._cap("importance", target))
        self._analysis_for("importance", Path(saved[0]).stem,
                           eid=f"fig.importance_{stage}_{target_slug(target).lower()}",
                           target=target,
                           n=int(len(self.df)),
                           top_features=[paper_label(f, self.lang) for f in imp["feature"].tolist()[::-1]],
                           model_type="XGBoost", cv_method="分组交叉验证训练的最终部署模型",
                           trend_desc="各输入变量的模型内置重要性存在明显层级差异。")
        return saved + [data, meta], stem

    def fig_c_shap_summary(self, stage, target):
        pipe = self._xgb_pipe(stage, target)
        try:
            import shap
        except Exception as e:
            raise PaperOutputError(f"SHAP 库不可用：{e}")
        X = self._stage_X(stage)
        y = self.df[target]
        X = X.loc[y.notna()]
        if len(X) > 400:
            X = X.sample(400, random_state=42)
        try:
            booster = pipe.named_steps["model"]
            Xi = pipe.named_steps["imputer"].transform(X)
            expl = shap.TreeExplainer(booster)
            sv = expl.shap_values(Xi)
            apply_paper_style(self.lang)
            fig = plt.figure(figsize=fig_double(5.4))
            shap.summary_plot(sv, Xi, feature_names=[paper_label(f, self.lang) for f in X.columns],
                              show=False)
            fig = plt.gcf()
            fig.set_size_inches(*fig_double(5.4))
            fig.tight_layout()
        except Exception as e:
            raise PaperOutputError(f"当前模型暂未启用该 SHAP 解释方式。（{type(e).__name__}）")
        saved, stem = self._save_fig(fig, f"Fig_C_{target_slug(target)}_SHAP_Summary")
        data = self._save_data(pd.DataFrame(sv, columns=X.columns, index=X.index),
                               f"Data_C_{target_slug(target)}_SHAP_Summary")
        meta = self._fig_meta("C", Path(saved[0]).stem, target=target, model_type="XGBoost",
                              data_file=Path(data).name, extra={"plot": "shap_summary"},
                              caption=self._cap("shap", target))
        return saved + [data, meta], stem

    def fig_c_shap_dependence(self, stage, target, feature):
        pipe = self._xgb_pipe(stage, target)
        try:
            import shap
        except Exception as e:
            raise PaperOutputError(f"SHAP 库不可用：{e}")
        if feature not in self.bundle[f"{stage}_features"]:
            raise PaperOutputError("所选特征不在该模型特征列表中。")
        X = self._stage_X(stage)
        y = self.df[target]
        X = X.loc[y.notna()]
        try:
            booster = pipe.named_steps["model"]
            Xi = pipe.named_steps["imputer"].transform(X)
            sv = shap.TreeExplainer(booster).shap_values(Xi)
            col = list(X.columns).index(feature)
        except Exception as e:
            raise PaperOutputError(f"当前模型暂未启用该 SHAP 解释方式。（{type(e).__name__}）")
        fig, ax = plt.subplots(figsize=fig_single(3.6))
        sc = ax.scatter(X[feature], sv[:, col], s=22, c=sv[:, col], cmap="coolwarm", alpha=0.8)
        ax.set_xlabel(paper_label(feature, self.lang))
        ax.set_ylabel(axis_label("SHAP值", self.lang))
        ax.set_title(f"SHAP Dependence — {paper_label(target, self.lang)}", fontsize=11.5)
        fig.colorbar(sc, ax=ax, label=axis_label("SHAP值", self.lang))
        fig.tight_layout()
        saved, stem = self._save_fig(fig, f"Fig_C_{target_slug(target)}_SHAP_Dep_{target_slug(feature)}")
        data = self._save_data(pd.DataFrame({
            "feature_value": X[feature].values,
            "shap_value": sv[:, col],
            "feature": feature,
        }), f"Data_C_{target_slug(target)}_SHAP_Dep_{target_slug(feature)}")
        meta = self._fig_meta("C", Path(saved[0]).stem, target=target, model_type="XGBoost",
                              data_file=Path(data).name, extra={"plot": "shap_dependence",
                                                                "feature": feature},
                              caption=self._cap("shap", target))
        return saved + [data, meta], stem

    def fig_c_pdp(self, stage, target, feature):
        if stage == "stage3":
            model_info = self.bundle["stage3_models"].get(target)
            if model_info is None:
                raise PaperOutputError("该目标当前没有已训练模型。")
            model = model_info["model"]
            feats = self.bundle["stage3_features"]
            oof1 = self.oof.get("stage1"); oof2 = self.oof.get("stage2")
            if oof1 is None or oof2 is None:
                raise PaperOutputError("缺少 OOF 预测记录，请先重新训练模型。")
            base = add_physics_features(self.df[self.bundle["x_cols"]])
            X = pd.concat([base, oof1.add_prefix("pred_"), oof2.add_prefix("pred_")], axis=1)
            X = X.reindex(columns=feats)
        else:
            model = self._xgb_pipe(stage, target)
            X = self._stage_X(stage)
        if feature not in X.columns:
            raise PaperOutputError("所选特征不在该模型特征列表中。")
        y = self.df[target]
        X = X.loc[y.notna()]
        grid = np.linspace(X[feature].min(), X[feature].max(), 30)
        means = []
        for v in grid:
            Xc = X.copy()
            Xc[feature] = v
            means.append(float(model.predict(Xc).mean()))
        fig, ax = plt.subplots(figsize=fig_single(3.2))
        ax.plot(grid, means, "-", color=BLUE_SIGNAL, linewidth=1.6)
        ax.set_xlabel(paper_label(feature, self.lang))
        ax.set_ylabel(axis_label("PDP响应", self.lang))
        ax.set_title(f"{'部分依赖图（PDP）' if self.lang == 'zh' else 'Partial Dependence Plot (PDP)'}"
                     f" — {paper_label(target, self.lang)}", fontsize=11)
        fig.tight_layout()
        saved, stem = self._save_fig(fig, f"Fig_C_{target_slug(target)}_PDP_{target_slug(feature)}")
        data = self._save_data(pd.DataFrame({"feature_value": grid, "mean_prediction": means,
                                             "feature": feature}),
                               f"Data_C_{target_slug(target)}_PDP_{target_slug(feature)}")
        meta = self._fig_meta("C", Path(saved[0]).stem, target=target,
                              model_type=self._model_kind(stage, target),
                              data_file=Path(data).name,
                              extra={"plot": "pdp",
                                     "note": "model response trend, not causality"},
                              caption=self._cap("pdp", target))
        return saved + [data, meta], stem

    # ---------- 模块 D：全链预测 ----------
    def fig_d_chain(self, inputs=None):
        schema = self.schema
        vals = {}
        for sec in ["structure_inputs", "process_inputs"]:
            for col, spec in schema[sec].items():
                if spec.get("categorical") or "min" not in spec or "max" not in spec:
                    continue  # V1.5：分类/无范围列不进入全链示意输入
                vals[col] = float((inputs or {}).get(col, (spec["min"] + spec["max"]) / 2))
        X = pd.DataFrame([vals])
        states, defects, perf, unc = predict_chain(X, self.bundle)
        std_map = unc.iloc[0].to_dict() if len(unc.columns) else {}

        # V1.5 修复：只显示实际建模的列（文献/快速分析数据可能缺部分状态列）
        levels = [
            ("I. 结构/工艺参数" if self.lang == "zh" else "I. Structure / process",
             [(c, vals[c], None) for c in list(schema["structure_inputs"]) + list(schema["process_inputs"])]),
            ("II. 射流与粒子状态" if self.lang == "zh" else "II. Jet & particle state",
             [(t, states.iloc[0][t], None) for t in self.bundle["state_cols"]
              if t in states.columns]),
            ("III. 涂层缺陷网络" if self.lang == "zh" else "III. Defect network",
             [(t, defects.iloc[0][t], None) for t in self.bundle["defect_cols"]
              if t in defects.columns]),
            ("IV. 涂层性能 (预测 ± 标准差)" if self.lang == "zh" else "IV. Performance (pred. ± std)",
             [(t, perf.iloc[0][t], std_map.get(t + "_std")) for t in self.bundle["perf_cols"]
              if t in perf.columns]),
        ]

        n_items = sum(len(items) for _, items in levels)
        total = len(levels) * 0.055 + n_items * 0.024 + (len(levels) - 1) * 0.05
        fig_h = max(5.0, 9.0 * total + 1.0)
        # V1.6：双栏 183mm 版式；箭头/边框改中性灰家族
        fig, ax = plt.subplots(figsize=fig_double(fig_h))
        ax.axis("off")
        y = 0.99
        dy_t = 0.05    # 标题行高
        dy_i = 0.022   # 条目行高
        dy_a = 0.045   # 箭头间隙
        for li, (title, items) in enumerate(levels):
            ax.text(0.5, y, title, ha="center", va="center", fontsize=11,
                    fontweight="bold", transform=ax.transAxes)
            y -= dy_t
            for name, value, std in items:
                label = paper_label(name, self.lang)
                if std is not None and not (isinstance(std, float) and np.isnan(std)):
                    txt = f"{label}: {value:.2f} ± {std:.2f}"
                else:
                    txt = f"{label}: {value:.2f}"
                ax.text(0.5, y, txt, ha="center", va="center", fontsize=8.5,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.25", fc=GRAY_PAPER,
                                  ec=GRAY_BORDER, lw=0.6))
                y -= dy_i
            if li < len(levels) - 1:
                ax.annotate("", xy=(0.5, y - dy_a * 0.6), xytext=(0.5, y),
                            xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=GRAY_REF, lw=1.0))
                y -= dy_a
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_D1_Process_to_Performance_Chain")

        rows = []
        for title, items in levels:
            for name, value, std in items:
                rows.append({"层级": title, "指标": paper_label(name, self.lang),
                             "预测值": round(float(value), 3),
                             "预测标准差": round(float(std), 3) if std is not None and not np.isnan(std) else None})
        df_out = pd.DataFrame(rows)
        excel = self.run_dir / "tables" / f"{self.demo_prefix}Table_D1_Full_Chain_Prediction.xlsx"
        with pd.ExcelWriter(excel, engine="openpyxl") as w:
            if self.demo:
                pd.DataFrame({"说明": ["DEMO SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY"]}).to_excel(
                    w, sheet_name="注意", index=False)
            df_out.to_excel(w, sheet_name="Chain", index=False)
        meta = self._fig_meta("D", Path(saved[0]).stem,
                              data_file=excel.name, extra={"plot": "full_chain",
                                                           "inputs": vals},
                              caption=self._cap("chain"))
        return saved + [excel, meta], stem

    # ---------- 模块 E：Pareto ----------
    def _pareto(self, n_candidates=2500, x_obj="porosity_pct", y_obj="bond_strength_MPa",
                color_obj="coupled_damage_rate"):
        obj_cfg = self.objective_cfg if self.objective_cfg is not None else {
                   "objectives": {"porosity_pct": "minimize", "bond_strength_MPa": "maximize",
                                  "coupled_damage_rate": "minimize"},
                   "constraints": {"deposition_efficiency_pct": {"min": 45},
                                   "powder_feed_g_min": {"min": 15}}}
        if self.objective_cfg is not None:
            targets = list(obj_cfg.get("objectives", {}))
            if len(targets) < 2:
                raise PaperOutputError("至少需要两个已训练涂层性能目标才能生成 Pareto 图表")
            x_obj, y_obj = targets[:2]
            color_obj = targets[2] if len(targets) > 2 else None
        fixed = {col: (float(s["min"]) + float(s["max"])) / 2
                 for col, s in self.schema["structure_inputs"].items()}
        front, stats, candidates = random_pareto_search(
            self.bundle, self.schema, obj_cfg, fixed_values=fixed,
            n_candidates=n_candidates, return_all=True)
        return front, stats, candidates, (x_obj, y_obj, color_obj)

    def fig_e_pareto(self, n_candidates=2500, x_obj="porosity_pct", y_obj="bond_strength_MPa",
                     color_obj="coupled_damage_rate"):
        front, stats, cand, (x_obj, y_obj, color_obj) = self._pareto(
            n_candidates, x_obj, y_obj, color_obj)
        if front.empty:
            raise PaperOutputError("没有找到满足当前约束的候选点，无法绘制 Pareto 前沿。")
        apply_paper_style(self.lang)
        fig, ax = plt.subplots(figsize=fig_single(4.4))
        cx = f"pred_{x_obj}"; cy = f"pred_{y_obj}"
        cc = f"pred_{color_obj}" if color_obj else None
        # V1.5 防御：文献/快速分析数据可能缺部分目标列 → 回退到实际可用的预测目标
        _avail = [c for c in front.columns if c.startswith("pred_") and not c.endswith("_std")]
        if cx not in front.columns and _avail:
            cx = _avail[0]
        if cy not in front.columns and len(_avail) > 1:
            cy = _avail[1] if _avail[1] != cx else _avail[-1]
        # V1.6：候选点中性灰半透明、前沿信号蓝描边（同源色板）
        ax.scatter(cand[cx], cand[cy], s=10, c=GRAY_REF, alpha=0.45, label=misc_label("candidates", self.lang))
        if cc in front.columns and cc not in (cx, cy):
            sc = ax.scatter(front[cx], front[cy], c=front[cc], s=42, cmap="viridis",
                            edgecolors=BLUE_SIGNAL, linewidths=0.9, label=misc_label("pareto_front", self.lang), zorder=3)
            fig.colorbar(sc, ax=ax, label=paper_label(cc.replace("pred_", ""), self.lang))
        else:
            ax.scatter(front[cx], front[cy], s=42, c=BLUE_LIGHT,
                       edgecolors=BLUE_SIGNAL, linewidths=0.9, label=misc_label("pareto_front", self.lang), zorder=3)
        ax.set_xlabel(paper_label(cx.replace("pred_", ""), self.lang))
        ax.set_ylabel(paper_label(cy.replace("pred_", ""), self.lang))
        ax.set_title("Pareto Front", fontsize=12)
        ax.legend(loc="best")
        fig.tight_layout()
        saved, stem = self._save_fig(fig, f"Fig_E1_Pareto_Front_{target_slug(x_obj)}_{target_slug(y_obj)}")
        data = self._save_data(front[[c for c in [cx, cy, cc] if c and c in front.columns] +
                                     list(self.schema["process_inputs"])],
                               f"Data_E1_Pareto_Front_{target_slug(x_obj)}_{target_slug(y_obj)}")
        meta = self._fig_meta("E", Path(saved[0]).stem, data_file=Path(data).name,
                              extra={"plot": "pareto_front", "stats": stats,
                                     "x": x_obj, "y": y_obj, "color": color_obj,
                                     "note": "no single 'best solution' is labeled"},
                              caption=self._cap("pareto"))
        self._analysis_for("pareto", Path(saved[0]).stem,
                           eid=f"fig.pareto_front_{target_slug(x_obj).lower()}_{target_slug(y_obj).lower()}",
                           n=int(stats.get("n_feasible", 0)),
                           pareto_count=int(stats.get("n_pareto", 0)),
                           model_type="三级代理模型",
                           cv_method="候选搜索（预测值非实验实测）",
                           trend_desc=f"非支配解在 {paper_label(x_obj, self.lang)}—"
                                      f"{paper_label(y_obj, self.lang)} 平面上构成权衡前沿，"
                                      "各解代表目标间的不同取舍，不存在单一最优。")
        return saved + [data, meta], stem, stats

    def table_e1(self, n_candidates=2500):
        front, stats, _, _ = self._pareto(n_candidates)
        if front.empty:
            raise PaperOutputError("没有找到满足当前约束的候选点。")
        disp = front.copy()
        domain = self.bundle.get("training_domain") or {}
        rows = []
        for i, (_, r) in enumerate(disp.iterrows(), start=1):
            row = {"方案编号": i}
            for c in self.schema["process_inputs"]:
                row[paper_label(c, self.lang)] = round(float(r[c]), 2)
            targets = (list(self.objective_cfg.get("objectives", {}))
                       if self.objective_cfg is not None else
                       ["porosity_pct", "bond_strength_MPa", "coupled_damage_rate",
                        "deposition_efficiency_pct"])
            for t in targets:
                if f"pred_{t}" not in r:
                    continue  # V1.5 防御：目标列缺失时跳过该列
                row[f"预测{paper_label(t, self.lang)}"] = round(float(r[f"pred_{t}"]), 3)
                if f"{t}_std" in r and not pd.isna(r[f"{t}_std"]):
                    row[f"预测{paper_label(t, self.lang)}标准差"] = round(float(r[f"{t}_std"]), 3)
            status = pareto_row_support({c: r[c] for c in self.schema["process_inputs"]}, domain)
            row["数据支持状态"] = support_label(status, self.lang)
            rows.append(row)
        df_t = pd.DataFrame(rows)
        note = "DEMO SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY" if self.demo else None
        path = self._save_table(df_t, "Table_E1_Pareto_Solutions", note=note)
        return path, df_t, stats

    def table_e2_ranges(self, n_candidates=2500):
        front, _, _, _ = self._pareto(n_candidates)
        if front.empty:
            raise PaperOutputError("没有找到满足当前约束的候选点。")
        rows = []
        for c in self.schema["process_inputs"]:
            s = front[c]
            rows.append({"工艺参数" if self.lang == "zh" else "Parameter": paper_label(c, self.lang),
                         "min": round(float(s.min()), 2),
                         "25%": round(float(s.quantile(0.25)), 2),
                         "median": round(float(s.median()), 2),
                         "75%": round(float(s.quantile(0.75)), 2),
                         "max": round(float(s.max()), 2)})
        df_t = pd.DataFrame(rows)
        note = "当前模型与约束条件下的 Pareto 工艺参数范围（不代表普适的工艺区间结论）" if self.lang == "zh" else \
               "Pareto parameter ranges under the current model and constraints"
        if self.demo:
            note = "DEMO SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY | " + note
        path = self._save_table(df_t, "Table_E2_Pareto_Process_Ranges", note=note)
        return path, df_t

    def fig_e2_ranges(self, n_candidates=2500):
        front, _, _, _ = self._pareto(n_candidates)
        if front.empty:
            raise PaperOutputError("没有找到满足当前约束的候选点。")
        params = list(self.schema["process_inputs"])
        fig, ax = plt.subplots(figsize=fig_mid(max(2.8, 0.4 * len(params) + 1.0)))
        for i, c in enumerate(params):
            lo_s, hi_s = self.schema["process_inputs"][c]["min"], self.schema["process_inputs"][c]["max"]
            lo = (front[c].min() - lo_s) / (hi_s - lo_s)
            hi = (front[c].max() - lo_s) / (hi_s - lo_s)
            med = (front[c].median() - lo_s) / (hi_s - lo_s)
            ax.plot([lo, hi], [i, i], "-", color=BLUE_SIGNAL, linewidth=5, alpha=0.6, solid_capstyle="round")
            ax.plot(med, i, "o", color=PINK_ACCENT, markersize=6, zorder=3)
        ax.set_yticks(range(len(params)), [paper_label(c, self.lang) for c in params])
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("0 – 1（在配置允许范围内的归一化位置）" if self.lang == "zh"
                      else "normalized position within configured bounds")
        ax.set_title("Pareto 工艺参数范围" if self.lang == "zh" else "Pareto parameter ranges", fontsize=11.5)
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_E2_Pareto_Process_Ranges")
        rows = []
        for c in params:
            rows.append({"parameter": c,
                         "min": float(front[c].min()), "25%": float(front[c].quantile(0.25)),
                         "median": float(front[c].median()), "75%": float(front[c].quantile(0.75)),
                         "max": float(front[c].max())})
        data = self._save_data(pd.DataFrame(rows), "Data_E2_Pareto_Process_Ranges")
        meta = self._fig_meta("E", Path(saved[0]).stem, data_file=Path(data).name,
                              extra={"plot": "pareto_ranges",
                                     "note": "not an 'absolute optimal process window'"},
                              caption=self._cap("ranges"))
        return saved + [data, meta], stem

    # ---------- 模块 F：不确定性 ----------
    def _deployment_predictions(self):
        """部署模型（全量训练）对全部样本的预测与标准差。
        明确标注：这是部署模型预测，不是交叉验证评价。"""
        x_cols = self.bundle["x_cols"]
        X = self.df[x_cols]
        states, defects, perf, unc = predict_chain(X, self.bundle)
        return states, defects, perf, unc

    def fig_f_uncertainty(self, target="bond_strength_MPa"):
        if target not in self.bundle["stage3_models"]:
            raise PaperOutputError("该目标不在三级性能模型中。")
        kind = self.bundle["stage3_models"][target]["kind"]
        if kind != "GaussianProcess":
            raise PaperOutputError("该模型不提供真实标准差，未人为编造不确定性。")
        states, defects, perf, unc = self._deployment_predictions()
        pred = perf[target]
        std = unc[f"{target}_std"]
        actual = pd.to_numeric(self.df[target], errors="coerce")
        idx = np.arange(len(pred))
        # V1.6：中幅 120mm；预测点信号蓝、实测对照点中性灰（强调家族留给残差语义）
        fig, ax = plt.subplots(figsize=fig_mid(3.6))
        ax.errorbar(idx, pred, yerr=std, fmt="o", ms=3.2, ecolor=BLUE_LIGHT,
                    elinewidth=0.9, capsize=1.6, color=BLUE_SIGNAL,
                    label=axis_label("预测不确定性", self.lang))
        m = actual.notna()
        ax.scatter(idx[m], actual[m], s=12, c=GRAY_MUTED, zorder=3, label=axis_label("actual", self.lang))
        ax.set_xlabel(axis_label("sample", self.lang))
        ax.set_ylabel(paper_label(target, self.lang))
        ax.legend(loc="best")
        fig.tight_layout()
        saved, stem = self._save_fig(fig, f"Fig_F1_Prediction_Uncertainty_{target_slug(target)}")
        lower = pred - 1.96 * std
        upper = pred + 1.96 * std
        data = self._save_data(pd.DataFrame({
            "sample_index": idx, "target": target,
            "actual": actual.values, "predicted": pred.values, "std": std.values,
            "lower_bound": lower.values, "upper_bound": upper.values,
        }), f"Data_F1_Uncertainty_{target_slug(target)}")
        meta = self._fig_meta("F", Path(saved[0]).stem, target=target, model_type=kind,
                              data_file=Path(data).name,
                              extra={"plot": "uncertainty",
                                     "note": "deployment model (trained on all data), not CV"},
                              caption=self._cap("uncertainty", target))
        table = pd.DataFrame({
            "actual": actual.values, "predicted": pred.values, "std": std.values,
            "lower_bound": lower.values, "upper_bound": upper.values,
        }).round(4)
        note = "DEMO SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY" if self.demo else None
        tpath = self._save_table(table, f"Table_F1_Prediction_Uncertainty_{target_slug(target)}", note=note)
        return saved + [data, tpath, meta], stem

    # ---------- 模块 G：实验验证（仅接口） ----------
    def table_g1_template(self):
        cols = ["方案编号", "方案类型(baseline/ML_recommended/validation)",
                "工艺参数(JSON)", "模型预测值", "模型不确定性", "实验实测值", "预测误差", "备注"]
        df_t = pd.DataFrame(columns=cols)
        path = self.run_dir / "tables" / f"{self.demo_prefix}Table_G1_Experimental_Validation.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            pd.DataFrame({"说明": ["实验验证数据模板：仅保存接口，平台不生成虚假实验数据。"
                                   if self.lang == "zh" else
                                   "Template only; the platform never fabricates validation data."]}
                         ).to_excel(w, sheet_name="注意", index=False)
            df_t.to_excel(w, sheet_name="Validation", index=False)
        return path

    # ---------- 打包 ----------
    # ================================================================
    # V1.4 H｜颗粒熔融与沉积状态（可选模块：Stage 1.5 未启用或数据不足时跳过，
    # 绝不生成空图或伪数据；不修改已有 A–G 模块）
    # ================================================================
    def _stage15_ready(self):
        return bool(self.bundle.get("stage15_enabled")) and \
            bool(self.bundle.get("stage15_models"))

    def _melt_oof(self, target):
        """Stage 1.5 目标的 OOF 序列（与 B 模块同一防泄漏原则）。"""
        stage_oof = self.oof.get("stage15")
        oof = stage_oof.get(target) if stage_oof is not None else None
        if oof is None or oof.dropna().empty:
            raise PaperOutputError("当前数据不足，无法生成该图。（缺少 Stage 1.5 逐行 OOF 记录）")
        pred = oof.dropna()
        actual = self.df.loc[pred.index, target]
        mask = actual.notna()
        return actual[mask], pred[mask]

    def _melt_state_col(self):
        """数据中的熔融状态类别列（有真实标签才有，不自动划分熔融区）。"""
        for c in ["melting_state_class", "splat_state_class"]:
            if c in self.df.columns and self.df[c].notna().sum() > 0:
                return c
        return None

    def _melt_particle_cols(self):
        t_col = v_col = None
        for c in self.df.columns:
            lc = str(c).lower()
            if t_col is None and "particle_temperature" in lc:
                t_col = c
            if v_col is None and "particle_velocity" in lc:
                v_col = c
        return t_col, v_col

    def fig_h_tv_map(self):
        """核心图 H1：颗粒温度—速度状态图（按真实熔融类别着色；无类别则纯散点）。"""
        t_col, v_col = self._melt_particle_cols()
        if t_col is None or v_col is None:
            raise PaperOutputError("当前数据不足，无法生成该图。（缺少粒子温度/粒子速度列）")
        t = pd.to_numeric(self.df[t_col], errors="coerce")
        v = pd.to_numeric(self.df[v_col], errors="coerce")
        m = t.notna() & v.notna()
        if int(m.sum()) < 3:
            raise PaperOutputError("当前数据不足，无法生成该图。（粒子温度/速度有效值不足）")
        cls_col = self._melt_state_col()
        # V1.6：单栏 89mm 版式；熔融类别着色改统一分类循环色（禁 seaborn 残留色）
        fig, ax = plt.subplots(figsize=fig_single(3.8))
        if cls_col:
            labels = self.df.loc[m, cls_col].astype(str)
            order = [c for c in ["insufficiently_molten", "partially_molten",
                                 "fully_molten", "overheated"] if c in set(labels)]
            order += [c for c in sorted(set(labels)) if c not in order]
            palette = list(QUAL_CYCLE)
            for i, lb in enumerate(order):
                sel = labels == lb
                name = MELTING_CLASS_ZH.get(lb, lb) if self.lang == "zh" else lb
                ax.scatter(v[m][sel], t[m][sel], s=22, alpha=0.75, edgecolors="none",
                           color=palette[i % len(palette)], label=f"{name} (n={int(sel.sum())})")
            ax.legend(fontsize=7.5, frameon=False, loc="best")
            title = "颗粒温度—速度状态图（按熔融状态类别着色）" if self.lang == "zh" \
                else "Particle temperature-velocity map (colored by melting state class)"
        else:
            ax.scatter(v[m], t[m], s=22, alpha=0.75, edgecolors="none", c=BLUE_SIGNAL)
            title = "颗粒温度—速度状态图（无熔融类别标签，仅散点）" if self.lang == "zh" \
                else "Particle temperature-velocity map (no class labels, scatter only)"
        ax.set_xlabel(axis_label(v_col, self.lang))
        ax.set_ylabel(axis_label(t_col, self.lang))
        ax.set_title(title, fontsize=11.5)
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_H01_Particle_TV_Map")
        df_data = pd.DataFrame({v_col: v[m], t_col: t[m]})
        if cls_col:
            df_data[cls_col] = self.df.loc[m, cls_col].astype(str).values
        data = self._save_data(df_data, "Data_H01_Particle_TV_Map")
        meta = self._fig_meta("H", Path(saved[0]).stem,
                              extra={"plot": "tv_map", "colored_by": cls_col or "none"},
                              caption=self._cap("tv_map") if hasattr(self, "_cap") else None)
        self._analysis_for("tv_map", Path(saved[0]).stem,
                           eid=f"fig.tv_map_{target_slug(t_col).lower()}",
                           n=int(m.sum()),
                           trend_desc="数据点在粒子温度—速度平面形成特定聚集区，"
                                      "反映当前工艺所覆盖的粒子热动力状态空间。",
                           melting_from_model=False,
                           class_counts=(df_data[cls_col].value_counts().to_dict()
                                         if cls_col else None))
        return saved + [data, meta], stem

    def fig_h_melting_distribution(self):
        """核心图 H2：熔融状态分布（样本数 + 百分比；仅有真实类别标签时输出）。"""
        cls_col = self._melt_state_col()
        if cls_col is None:
            raise PaperOutputError("当前数据不足，无法生成该图。（缺少熔融状态类别标签）")
        counts = self.df[cls_col].dropna().astype(str).value_counts()
        order = [c for c in ["insufficiently_molten", "partially_molten",
                             "fully_molten", "overheated"] if c in counts.index]
        order += [c for c in counts.index if c not in order]
        names = [MELTING_CLASS_ZH.get(c, c) if self.lang == "zh" else c for c in order]
        vals = [int(counts[c]) for c in order]
        total = sum(vals)
        fig, ax = plt.subplots(figsize=fig_single(3.2))
        bars = ax.bar(range(len(order)), vals, color=BLUE_SIGNAL, alpha=0.85)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(names, fontsize=8.5)
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{vals[i]}\n({vals[i] / total * 100:.1f}%)",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("样本数" if self.lang == "zh" else "Count")
        ax.set_title("颗粒熔融状态分布" if self.lang == "zh" else "Melting state distribution",
                     fontsize=11.5)
        ax.set_ylim(0, max(vals) * 1.25)
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_H02_Melting_State_Distribution")
        df_data = pd.DataFrame({"melting_state_class": order, "count": vals,
                                "pct": [v / total * 100 for v in vals]})
        data = self._save_data(df_data, "Data_H02_Melting_State_Distribution")
        meta = self._fig_meta("H", Path(saved[0]).stem,
                              extra={"plot": "melting_distribution"},
                              caption=self._cap("melting_distribution") if hasattr(self, "_cap") else None)
        self._analysis_for("melting_dist", Path(saved[0]).stem, n=int(total),
                           eid="fig.melting_distribution",
                           class_counts={MELTING_CLASS_ZH.get(c, c): int(v)
                                         for c, v in zip(order, vals)},
                           trend_desc="各熔融状态类别的样本数与占比存在差异，"
                                      "类别标签来自数据/文献本身。")
        return saved + [data, meta], stem

    def fig_h_melting_performance(self):
        """核心图 H3：熔融状态模型评价——连续目标 parity（OOF），分类目标混淆矩阵（OOF）。"""
        targets = list((self.bundle.get("stage15_models") or {}).keys())
        if not targets:
            raise PaperOutputError("当前数据不足，无法生成该图。（Stage 1.5 未启用）")
        kinds = self.bundle.get("stage15_kind") or {}
        cont = [t for t in targets if kinds.get(t) == "regression"]
        cat = [t for t in targets if kinds.get(t) == "classification"]
        n_panels = len(cont) + len(cat)
        if n_panels == 0:
            raise PaperOutputError("当前数据不足，无法生成该图。")
        # V1.6：多面板按列数取单栏×n（Nature 模数）；色值全部同源家族
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(fig_single(3.9)[0] * n_panels, 3.9), squeeze=False)
        stats_out = {}
        for j, t in enumerate(cont):
            ax = axes[0][j]
            actual, pred = self._melt_oof(t)
            met = (self.metrics.get("stage15") or {}).get(t) or {}
            r2, rmse, mae = met.get("R2"), met.get("RMSE"), met.get("MAE")
            lo = float(min(actual.min(), pred.min()))
            hi = float(max(actual.max(), pred.max()))
            pad = (hi - lo) * 0.05 + 1e-9
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=GRAY_REF,
                    linewidth=1.0, zorder=1)
            ax.scatter(actual, pred, s=24, c=BLUE_SIGNAL, alpha=0.75, edgecolors="none", zorder=2)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(axis_label("actual", self.lang))
            ax.set_ylabel(axis_label("predicted", self.lang))
            ax.set_title(f"{paper_label(t, self.lang)} (OOF)", fontsize=10.5)
            stats_out[t] = {"R2": r2, "RMSE": rmse, "MAE": mae, "n": int(len(actual))}
            add_stat_box(ax, f"R²={r2:.3f}\nRMSE={rmse:.3g}\nn={len(actual)}",
                         loc="upper left")
        for j, t in enumerate(cat, start=len(cont)):
            ax = axes[0][j]
            actual, pred = self._melt_oof(t)
            a_lab = actual.astype(str)
            p_lab = pred.astype(str)
            classes = sorted(set(a_lab.unique()) | set(p_lab.unique()))
            cm = pd.crosstab(a_lab, p_lab).reindex(index=classes, columns=classes, fill_value=0)
            ax.imshow(cm.values, cmap="Blues")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            short = [MELTING_CLASS_ZH.get(c, c) if self.lang == "zh" else c for c in classes]
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short, fontsize=7)
            for i in range(len(classes)):
                for k in range(len(classes)):
                    ax.text(k, i, str(cm.values[i, k]), ha="center", va="center",
                            fontsize=7.5, color=GRAY_INK)
            acc = float(np.trace(cm.values) / max(1, cm.values.sum()))
            met = (self.metrics.get("stage15") or {}).get(t) or {}
            ax.set_title(f"{paper_label(t, self.lang)} (OOF)\nAcc={acc:.3f} "
                         f"BalAcc={met.get('Balanced_Accuracy', float('nan')):.3f}",
                         fontsize=9.5)
            ax.set_xlabel(axis_label("predicted", self.lang))
            ax.set_ylabel(axis_label("actual", self.lang))
            stats_out[t] = {"Accuracy": met.get("Accuracy"), "Balanced_Accuracy": met.get("Balanced_Accuracy"),
                            "F1_macro": met.get("F1_macro"), "n": int(len(actual))}
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_H03_Melting_Model_Performance")
        rows = []
        for t, st in stats_out.items():
            rows.append({"target": t, **{k: v for k, v in st.items()}})
        data = self._save_data(pd.DataFrame(rows), "Data_H03_Melting_Model_Performance")
        meta = self._fig_meta("H", Path(saved[0]).stem,
                              extra={"plot": "melting_model_performance", "oof_only": True,
                                     "stats": stats_out},
                              caption=self._cap("melting_performance") if hasattr(self, "_cap") else None)
        _r2_first = next((v.get("R2") for v in stats_out.values() if v.get("R2") is not None), None)
        _eid_h3 = "fig.melting_performance" + (f"_{self._slug_first(stats_out)}" if stats_out else "")
        self._analysis_for("melting_perf", Path(saved[0]).stem, r2=_r2_first,
                           eid=_eid_h3,
                           n=int(len(self.df)), model_type="Stage 1.5 XGBoost",
                           cv_method="分组交叉验证 OOF（连续目标 Parity / 分类目标混淆矩阵）",
                           melting_from_model=True,
                           trend_desc="熔融状态模型的折外预测与实验/文献标签总体一致性见图；"
                                      "分类目标请同时参考平衡准确率。")
        return saved + [data, meta], stem

    @staticmethod
    def _slug_first(stats_out):
        """H3 证据 ID 的目标 slug（首个目标，小写字母数字）。"""
        return "".join(ch for ch in str(next(iter(stats_out), "")).lower() if ch.isalnum())

    def _fig_h_corr_grid(self, pair_cols, code, title_zh, title_en):
        """熔融指数与缺陷/性能指标的关联趋势网格（只描述关联，不生成因果结论）。"""
        melt_col = "melting_index"
        if melt_col not in self.df.columns:
            melt_col = "melting_fraction_pct"
        if melt_col not in self.df.columns:
            raise PaperOutputError("当前数据不足，无法生成该图。（缺少 melting_index / melting_fraction_pct）")
        pairs = [(melt_col, y) for y in pair_cols if y in self.df.columns]
        if not pairs:
            raise PaperOutputError("当前数据不足，无法生成该图。（缺少对应缺陷/性能列）")
        n = len(pairs)
        ncols = 2 if n > 1 else 1
        nrows = (n + 1) // 2
        # V1.6：多面板按列数取单栏×n；散点信号蓝、趋势虚线中性灰
        _fw = fig_double(3.5 * nrows)[0] if ncols == 2 else fig_single(3.5)[0]
        fig, axes = plt.subplots(nrows, ncols, figsize=(_fw, 3.5 * nrows), squeeze=False)
        for idx, (x, y) in enumerate(pairs):
            ax = axes[idx // ncols][idx % ncols]
            xv = pd.to_numeric(self.df[x], errors="coerce")
            yv = pd.to_numeric(self.df[y], errors="coerce")
            m = xv.notna() & yv.notna()
            ax.scatter(xv[m], yv[m], s=20, c=BLUE_SIGNAL, alpha=0.7, edgecolors="none")
            if int(m.sum()) >= 3 and xv[m].nunique() > 1:
                z = np.polyfit(xv[m], yv[m], 1)
                xs = np.linspace(float(xv[m].min()), float(xv[m].max()), 50)
                ax.plot(xs, np.polyval(z, xs), "--", color=GRAY_REF, linewidth=1.0)
                r = float(np.corrcoef(xv[m], yv[m])[0, 1])
                ax.set_title(f"{paper_label(y, self.lang)}  r={r:.2f}", fontsize=9.5)
            else:
                ax.set_title(paper_label(y, self.lang), fontsize=9.5)
            ax.set_xlabel(paper_label(x, self.lang))
            ax.set_ylabel(paper_label(y, self.lang))
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig.suptitle(title_zh if self.lang == "zh" else title_en, fontsize=11.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        saved, stem = self._save_fig(fig, code)
        rows = []
        for x, y in pairs:
            xv = pd.to_numeric(self.df[x], errors="coerce")
            yv = pd.to_numeric(self.df[y], errors="coerce")
            m = xv.notna() & yv.notna()
            rows.append(pd.DataFrame({x: xv[m], y: yv[m]}))
        df_data = pd.concat(rows, axis=1)
        data = self._save_data(df_data, code.replace("Fig_", "Data_"))
        meta = self._fig_meta("H", Path(saved[0]).stem,
                              extra={"plot": "correlation_trend_only",
                                     "note": "关联趋势，不构成因果结论"},
                              caption=self._cap("melting_corr") if hasattr(self, "_cap") else None)
        return saved + [data, meta], stem

    def fig_h_melting_vs_defects(self):
        """核心图 H4：熔融状态 → 缺陷（关联趋势，非因果）。"""
        return self._fig_h_corr_grid(
            ["porosity_pct", "lamellar_gap_pct", "unmelted_particle_pct", "defect_connectivity_index"],
            "Fig_H04_Melting_vs_Defects",
            "熔融指数与缺陷指标的关联（趋势描述，非因果）",
            "Melting index vs defect indicators (trend only, not causal)")

    def fig_h_melting_vs_performance(self):
        """核心图 H5：熔融状态 → 性能（关联趋势，非因果）。"""
        return self._fig_h_corr_grid(
            ["bond_strength_MPa", "hardness_HV", "wear_rate_mg_m", "coupled_damage_rate"],
            "Fig_H05_Melting_vs_Performance",
            "熔融指数与涂层性能的关联（趋势描述，非因果）",
            "Melting index vs coating performance (trend only, not causal)")

    def fig_h_melting_importance(self):
        """核心图 H6：Stage 1.5 特征重要性（影响颗粒熔融状态的结构/工艺/材料/过程因素）。"""
        models = self.bundle.get("stage15_models") or {}
        if not models:
            raise PaperOutputError("当前数据不足，无法生成该图。（Stage 1.5 未启用）")
        t0 = list(models.keys())[0]
        pipe = models[t0]
        if not hasattr(pipe, "named_steps") or "model" not in pipe.named_steps:
            raise PaperOutputError("当前数据不足，无法生成该图。")
        booster = pipe.named_steps["model"]
        if not hasattr(booster, "feature_importances_"):
            raise PaperOutputError("当前数据不足，无法生成该图。（该模型类型不支持特征重要性）")
        names = self.bundle.get("stage15_features") or []
        imp = pd.Series(booster.feature_importances_, index=names).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=fig_single(4.2))
        y = np.arange(len(imp))[::-1]
        ax.barh(y, imp.values, color=BLUE_SIGNAL, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([paper_label(f, self.lang) for f in imp.index], fontsize=8)
        ax.set_xlabel("Feature importance", fontsize=9)
        ax.set_title(f"{paper_label(t0, self.lang)} — "
                     + ("影响因素 Top %d" % len(imp) if self.lang == "zh" else "Top %d features" % len(imp)),
                     fontsize=11)
        fig.tight_layout()
        saved, stem = self._save_fig(fig, "Fig_H06_Melting_Feature_Importance")
        data = self._save_data(pd.DataFrame({"feature": imp.index, "importance": imp.values}),
                               "Data_H06_Melting_Feature_Importance")
        meta = self._fig_meta("H", Path(saved[0]).stem, target=t0,
                              extra={"plot": "stage15_importance"},
                              caption=self._cap("melting_importance") if hasattr(self, "_cap") else None)
        return saved + [data, meta], stem

    def build_bundle(self, n_candidates=2500):
        """生成完整论文图表包，返回 (run_dir, zip_path, summary)。"""
        paths = {"figures": [], "tables": [], "data": [], "metadata": []}
        man = self.reproducibility_manifest()

        # A
        p, _ = self.table_a1(); paths["tables"].append(p)
        for fn in [self.fig_a1_completeness, self.fig_a2_distributions, self.fig_a3_correlation]:
            out = fn()
            self._collect(paths, out)

        # B
        p, _ = self.table_b1(); paths["tables"].append(p)
        for stage in ["stage1", "stage2", "stage3"]:
            for t in list((self.metrics.get(stage) or {}).keys()):
                self._collect(paths, self.fig_parity(stage, t))
                self._collect(paths, self.fig_residual(stage, t))
        self._collect(paths, self.fig_b_overview())

        # C（首个 stage1 目标 + 首个 stage2 目标）
        for stage in ["stage1", "stage2"]:
            targets = list(self.bundle.get(f"{stage}_models", {}).keys())
            if targets:
                t0 = targets[0]
                self._collect(paths, self.fig_c_importance(stage, t0, topn=15))
                try:
                    self._collect(paths, self.fig_c_shap_summary(stage, t0))
                except PaperOutputError:
                    pass

        # D
        if self.bundle.get("stage3_models"):
            self._collect(paths, self.fig_d_chain())

        # E（V1.5：分步容错——文献/快速分析数据缺目标列时跳过 E，不影响其他模块）
        stats, rng_df = None, pd.DataFrame()
        if self.bundle.get("stage3_models") and (self.objective_cfg is None or
                                                   len(self.objective_cfg.get("objectives", {})) >= 2):
            try:
                self._collect(paths, self.fig_e_pareto(n_candidates=n_candidates)[0:2])
                p, _, stats = self.table_e1(n_candidates); paths["tables"].append(p)
                p, rng_df = self.table_e2_ranges(n_candidates); paths["tables"].append(p)
                self._collect(paths, self.fig_e2_ranges(n_candidates))
            except PaperOutputError:
                pass
            except Exception as e:
                log_event("paper_export_e_error", f"{e}")

        # F（每个 GP 目标）
        for t in list(self.bundle.get("stage3_models", {}).keys()):
            try:
                self._collect(paths, self.fig_f_uncertainty(t))
            except PaperOutputError:
                pass

        # G
        paths["tables"].append(self.table_g1_template())

        # H（V1.4：仅当 Stage 1.5 启用且数据支持时生成；数据不足的子图自动跳过，不生成空图）
        if self._stage15_ready():
            for fn in [self.fig_h_tv_map, self.fig_h_melting_distribution,
                       self.fig_h_melting_performance, self.fig_h_melting_vs_defects,
                       self.fig_h_melting_vs_performance, self.fig_h_melting_importance]:
                try:
                    self._collect(paths, fn())
                except PaperOutputError:
                    pass  # 该子图数据不足，跳过
                except Exception as e:
                    log_event("paper_export_h_error", f"{getattr(fn, '__name__', fn)}: {e}")

        # I（V1.5：数据洞察与实验反馈；单步失败不影响 A–H 已生成结果）
        try:
            from .insights import build_insight_report
            from .context import build_context
            _ctx = build_context("unknown", "demo" if self.demo else "real", df=self.df,
                                 bundle=self.bundle)
            _ctx["_demo"] = self.demo
            md_i, xlsx_i = build_insight_report(_ctx, self.df, self.schema, self.bundle,
                                                self.run_dir / "insights",
                                                registry=self.registry)
            paths.setdefault("insights", []).extend([md_i, xlsx_i])
        except Exception as e:
            log_event("paper_export_i_error", f"{e}")

        # V1.5：图表科研解读汇总
        try:
            from .figure_analysis import summarize_analyses
            summary_an = summarize_analyses(self.analysis_dir)
            paths.setdefault("analysis", []).append(summary_an)
        except Exception as e:
            log_event("paper_export_analysis_summary_error", f"{e}")

        # metadata + summary + captions
        self.write_reproducibility_manifest()
        self.write_captions()

        # V1.6（P0-2 验收 6）：evidence_manifest.json + summary.json 透传进 ZIP——
        # quick run 场景两文件位于 run_dir 上一级（runs/quick_analysis/<run_id>/），
        # 复制到本输出目录（ZIP 内 results/ 子目录）后随 _zip_results 一并打包。
        # V1.6 fix（QA 回归问题 1 补充）：registry 传入时，此时 fig.* 证据已在
        # 上方图表/解读生成过程中注册——直接落盘完整 manifest，避免复制早于
        # _package 保存的旧版本（缺 fig.* 条目）。
        try:
            if self.registry is not None and len(self.registry) > 0:
                _mf_local = self.registry.save(self.run_dir)
                paths.setdefault("metadata", []).append(_mf_local)
            else:
                _mf_local = self.run_dir / "evidence_manifest.json"
                if not _mf_local.exists():
                    _mf_src = self.run_dir.parent / "evidence_manifest.json"
                    if _mf_src.exists():
                        shutil.copy2(_mf_src, _mf_local)
                        paths.setdefault("metadata", []).append(_mf_local)
        except Exception as e:
            log_event("paper_export_manifest_error", f"{e}")
        # V1.6 fix（QA 回归问题 2）：summary.json（智能摘要卡数据源）同法打包
        try:
            _sm_local = self.run_dir / "summary.json"
            if not _sm_local.exists():
                _sm_src = self.run_dir.parent / "summary.json"
                if _sm_src.exists():
                    shutil.copy2(_sm_src, _sm_local)
                    paths.setdefault("metadata", []).append(_sm_local)
        except Exception as e:
            log_event("paper_export_summary_error", f"{e}")

        top_features = []
        try:
            pipe = list(self.bundle["stage1_models"].values())[0]
            booster = pipe.named_steps["model"]
            s = pd.Series(booster.feature_importances_, index=self.bundle["stage1_features"])
            top_features = list(s.sort_values(ascending=False).items())
        except Exception:
            pass
        rng_rows = rng_df.to_dict("records") if len(rng_df) else []
        summary = self.write_results_summary(pareto_stats=stats, pareto_ranges=rng_rows,
                                             top_features=top_features)
        paths["summary"] = summary

        zip_path = self._zip_results(self.run_dir)
        paths["zip"] = zip_path
        paths["captions"] = self.run_dir / "Fig_Captions.md"
        log_event("paper_export", f"run={self.run_dir.name} figures={len(paths['figures'])} zip={zip_path.name}")
        return self.run_dir, zip_path, paths

    @staticmethod
    def _collect(paths, out):
        # 各 fig_* 方法返回 (path_list, stem) 或 (path_list, stem, extra)
        items = out[0]
        for p in items:
            sp = str(p)
            if sp.endswith(".png") or sp.endswith(".svg") or sp.endswith(".pdf"):
                paths["figures"].append(p)
            elif sp.endswith(".csv"):
                paths["data"].append(p)
            elif sp.endswith(".xlsx"):
                paths["tables"].append(p)
            elif sp.endswith(".json"):
                paths["metadata"].append(p)


def save_snapshot(df, schema, bundle, n_pareto_candidates=600):
    """研究快照（V1.3.1）：outputs/snapshots/Snapshot_<ts>/
    包含 dataset_info / model_info / metrics.xlsx / Pareto结果 / config 副本 /
    reproducibility manifest / README.txt。不复制 venv、缓存与临时文件。"""
    from .config import load_objectives
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = ROOT / "outputs" / "snapshots" / f"Snapshot_{ts}"
    (snap / "config").mkdir(parents=True, exist_ok=True)

    dataset_hash_str = hashlib.sha256(df.to_csv(index=True).encode("utf-8")).hexdigest()
    dataset_info = {
        "dataset_hash": dataset_hash_str,
        "dataset_version": "auto_" + dataset_hash_str[:8],
        "sample_count": int(len(df)),
        "batch_count": int(df["batch_id"].nunique()) if "batch_id" in df else None,
        "columns": list(df.columns),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    _dump_json(snap / "dataset_info.json", dataset_info)

    cv = bundle.get("chain_cv") or {}
    model_info = {
        "model_version": bundle.get("model_version"),
        "random_seed": bundle.get("seed"),
        "dataset_hash": bundle.get("dataset_hash"),
        "cv_method": cv.get("method", "none"),
        "targets": {s: list(bundle.get(f"{s}_models", {}).keys())
                    for s in ["stage1", "stage2", "stage3"]},
        "features": {s: bundle.get(f"{s}_features", [])
                     for s in ["stage1", "stage2", "stage3"]},
        "hyperparameters": {"xgboost": _XGB_HYPERPARAMS},
        "training_domain": bundle.get("training_domain"),
    }
    _dump_json(snap / "model_info.json", model_info)

    with pd.ExcelWriter(snap / "metrics.xlsx", engine="openpyxl") as w:
        for stage in ["stage1", "stage2", "stage3"]:
            rows = []
            for t, met in (cv.get("metrics") or {}).get(stage, {}).items():
                if stage == "stage3":
                    kind = bundle.get("stage3_models", {}).get(t, {}).get("kind", "-")
                else:
                    kind = "XGBoost"
                rows.append({"stage": stage, "target": t,
                             "R2": round(met["R2"], 4), "RMSE": round(met["RMSE"], 4),
                             "MAE": round(met["MAE"], 4), "model": kind})
            pd.DataFrame(rows).to_excel(w, sheet_name=stage, index=False)

    try:
        obj_cfg = load_objectives()
        fixed = {c: (float(s["min"]) + float(s["max"])) / 2
                 for c, s in schema["structure_inputs"].items()}
        front, _, _ = random_pareto_search(bundle, schema, obj_cfg, fixed_values=fixed,
                                           n_candidates=n_pareto_candidates, return_all=True)
        front.to_excel(snap / "Pareto结果.xlsx", index=False)
    except Exception as e:
        _dump_json(snap / "Pareto结果_错误.json", {"error": str(e)})

    for cfg in ["data_schema.yaml", "objectives.yaml", "data_dictionary.yaml"]:
        src = ROOT / "config" / cfg
        if src.exists():
            shutil.copy2(src, snap / "config" / cfg)

    exporter = PaperExporter(df, schema, bundle, demo=False, run_dir=snap / "_tmp_export")
    _dump_json(snap / "reproducibility_manifest.json",
               {**exporter.reproducibility_manifest(), "date": datetime.now().isoformat(timespec="seconds")})
    tmp = snap / "_tmp_export"
    if tmp.exists():
        shutil.rmtree(tmp)

    readme = ("研究快照（只读存档）\n"
              f"生成时间：{datetime.now().isoformat(timespec='seconds')}\n"
              f"模型版本：{model_info['model_version']}\n"
              f"数据版本：{dataset_info['dataset_version']}\n\n"
              "内容：dataset_info.json / model_info.json / metrics.xlsx / Pareto结果.xlsx / "
              "config 副本 / reproducibility_manifest.json\n"
              "说明：本快照用于科研追溯，不包含虚拟环境、缓存或临时文件；请勿手工修改。\n")
    (snap / "README.txt").write_text(readme, encoding="utf-8")
    log_event("snapshot", snap.name)
    return snap


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
