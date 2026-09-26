# -*- coding: utf-8 -*-
"""V1.5 一键科研分析引擎（快速模式）。

原则（spec 二十四）：自动识别但绝不盲目猜测——
  自动识别 → 数据安全检查 → 字段映射 → 高置信度自动运行 → 只让用户确认歧义字段。

隔离铁律（spec 三十一）：Quick Run 一律保存在 runs/quick_analysis/<run_id>/，
不覆盖正式 models/；【设为正式模型】必须由用户主动确认。

多 Sheet（spec 二十六）：README / Source_Metadata / Platform_* 视为元数据，
Dataset_01 / Dataset_02 … 每个工作表独立 Run，绝不自动合并不同文献数据。
"""
import json
import shutil
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ROOT
from .modes import MODES, load_schema_for_mode, suggest_cv
from .literature import build_runtime_schema, MELTING_ALIAS_SUGGESTIONS
from .evidence import EvidenceRegistry
from .paper_labels import cv_display_method, stage_scope_label, target_slug

QUICK_ROOT = ROOT / "runs" / "quick_analysis"
HISTORY_FILE = QUICK_ROOT / "run_history.json"
SUMMARY_NAME = "summary.json"
WORKSPACE_NAME = "workspace.json"


def save_quick_workspace(run_id, df, schema, bundle, objectives=None):
    """Keep the exact training input and role mapping alongside the isolated run."""
    from .research_utils import dataset_hash
    if dataset_hash(df) != str(bundle.get("dataset_hash", ""))[:16]:
        raise ValueError("快速运行的数据指纹与训练模型不一致")
    run_dir = QUICK_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Pickle is only read back from a locally created, history-listed run.
    df.to_pickle(run_dir / "input_data.pkl")
    (run_dir / WORKSPACE_NAME).write_text(json.dumps({
        "schema": schema, "objectives": objectives or {},
        "dataset_hash": bundle["dataset_hash"],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return run_dir / WORKSPACE_NAME


def load_quick_workspace(run_id):
    """Read a successful local run; reject legacy/mismatched snapshots explicitly."""
    from .research_utils import dataset_hash
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("无效的快速运行编号")
    record = next((r for r in load_history() if r.get("run_id") == run_id
                   and not r.get("failed")), None)
    if record is None:
        raise ValueError("该快速运行未完成或已不在运行历史中")
    run_dir = QUICK_ROOT / run_id
    if not (run_dir / WORKSPACE_NAME).exists() or not (run_dir / "input_data.pkl").exists():
        raise ValueError("该运行由旧版本创建，缺少原始数据快照；请重新运行智能分析")
    meta = json.loads((run_dir / WORKSPACE_NAME).read_text(encoding="utf-8"))
    df = pd.read_pickle(run_dir / "input_data.pkl")
    bundle = joblib.load(run_dir / "model.joblib")
    digest = bundle.get("dataset_hash")
    if not digest or meta.get("dataset_hash") != digest or dataset_hash(df) != digest[:16]:
        raise ValueError("快速运行的数据与模型指纹不一致，请重新运行智能分析")
    return {"df": df, "schema": meta["schema"], "bundle": bundle,
            "objectives": meta.get("objectives") or {}, "record": record}

META_SHEET_KEYWORDS = ["readme", "source_metadata", "platform_field_mapping",
                       "platform_test_plan", "使用说明", "说明", "字典", "字段", "metadata",
                       "文献总览", "论文结果核对", "参考文献"]


# ---------------- 逐行折外预测导出（要求 E） ----------------
def export_stage1_oof_csv(results_dir, df, bundle, prefix="QUICK_LIT_"):
    """为每个 Stage 1 目标导出逐行折外预测 CSV。

    文件：results/data/<prefix>Data_OOF_<目标slug>.csv
    列：row_index（0-based 原始行号）、fold（0–4，该行作为验证折的编号）、
        actual、predicted、residual = predicted − actual。
    数据全部来自 run_quick_analysis 训练时的 KFold / GroupKFold / LOOCV 切分
    （bundle["chain_cv"]["oof_predictions"] 与 ["oof_row_records"]），不做任何
    二次预测；actual 缺失的行不导出。返回写入的文件路径列表。
    """
    cv = bundle.get("chain_cv") or {}
    oof_all = cv.get("oof_predictions")
    stage1_oof = oof_all.get("stage1") if isinstance(oof_all, dict) else None
    if stage1_oof is None or len(stage1_oof) == 0:
        return []
    folds = cv.get("oof_row_records")
    if folds is None:
        return []
    data_dir = Path(results_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    row_pos = {idx: i for i, idx in enumerate(df.index)}
    written = []
    for target, pred in stage1_oof.items():
        if target not in df.columns:
            continue
        pred_v = pred.dropna()
        rows = []
        for idx, pv in pred_v.items():
            actual = pd.to_numeric(df.loc[idx, target], errors="coerce")
            fold = folds.loc[idx] if idx in folds.index else np.nan
            if pd.isna(actual) or pd.isna(pv) or pd.isna(fold):
                continue
            rows.append({"row_index": row_pos[idx], "fold": int(fold),
                         "actual": float(actual), "predicted": float(pv),
                         "residual": float(pv) - float(actual)})
        if not rows:
            continue
        out = data_dir / f"{prefix}Data_OOF_{target_slug(target)}.csv"
        pd.DataFrame(rows).sort_values("row_index").to_csv(
            out, index=False, encoding="utf-8-sig")
        written.append(out)
    return written

# 通用别名（字段名 → 规范字段），仅用于高置信度自动映射
GENERIC_ALIAS = {
    **{alias: f for f, aliases in MELTING_ALIAS_SUGGESTIONS.items() for alias in aliases},
    "arc_current": "arc_current_A", "current_a": "arc_current_A",
    "spray_distance": "spray_distance_mm", "standoff_distance_mm": "spray_distance_mm",
    "powder_feed": "powder_feed_g_min", "feed_rate_g_min": "powder_feed_g_min",
    "porosity": "porosity_pct", "bond_strength": "bond_strength_MPa",
    "hardness": "hardness_HV",
}

# Published APS process data: retain the original headers and units. Pressure in psi
# must never be silently interpreted as a gas flow in slpm. An experiment number is
# an identifier, not a tunable process variable.
LITERATURE_ROLE_ALIASES = {
    "ar压力(psi)": "x", "h2压力(psi)": "x",
    "ar_pressure_psi": "x", "h2_pressure_psi": "x",
    "电流(a)": "x", "喷距(mm)": "x", "stand_off_distance_mm": "x",
    "粒子温度(°c)": "states", "粒子速度(m/s)": "states",
    "particle_temperature_c": "states", "particle_velocity_m_s": "states",
}
IDENTIFIER_COLUMNS = {"喷涂序号", "spray_run_id", "experiment_id", "batch_id"}

ROLE_SECTIONS = {
    "structure_inputs": "structure_inputs",
    "process_inputs": "process_inputs",
    "process_states": "process_states",
    "defect_network": "defect_network",
    "performance_outputs": "performance_outputs",
    "material_inputs": "material_inputs",
    "melting_states": "melting_states",
}


# ---------------- Run History ----------------
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def append_history(record):
    QUICK_ROOT.mkdir(parents=True, exist_ok=True)
    hist = load_history()
    hist.append(record)
    HISTORY_FILE.write_text(json.dumps(hist[-200:], ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return record


def latest_runs(n=10):
    return list(reversed(load_history()))[:n]


def promote_to_formal(run_id, mode_id):
    """用户主动确认后，把 Quick Run 模型设为该模式正式模型。"""
    src = QUICK_ROOT / run_id / "model.joblib"
    dst = ROOT / "models" / mode_id / "latest_chain.joblib"
    if not src.exists():
        raise FileNotFoundError(f"Quick Run 模型不存在：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# ---------------- 数据准备检查（spec 六十七） ----------------
def data_readiness_issues(df, schema):
    """训练前数据安全检查；返回 (critical, warning) 中文问题列表，禁止直接 traceback。"""
    critical, warning = [], []
    n = len(df)
    if n == 0:
        return ["数据表为空"], []
    if n < 6:
        critical.append(f"样本量过少（n={n}），无法进行交叉验证")
    elif n < 10:
        warning.append(f"当前数据量极少（n={n}），结果仅适合作为软件功能测试或探索性分析")
    elif n <= 25:
        warning.append(f"样本量较小（n={n}），建议使用 LOOCV（平台已自动选择）")

    targets = []
    for sec in ["process_states", "defect_network", "performance_outputs", "melting_states"]:
        targets += [c for c in (schema.get(sec) or {}) if c in df.columns]
    usable = 0
    for t in targets:
        s = pd.to_numeric(df[t], errors="coerce")
        # train_chain requires >=20 measured target values; preflight must use
        # the same threshold so it never announces a trainable but empty model.
        if s.notna().sum() < 20:
            continue
        usable += 1
        if s.nunique() <= 1:
            warning.append(f"目标 {t} 为常数（无预测价值）")
    if usable == 0:
        crit = "未找到任何可训练目标（目标列缺失、非数值或有效值不足 20 条）"
        critical.append(crit)

    ins = [c for c in (schema.get("structure_inputs") or {}) if c in df.columns] + \
          [c for c in (schema.get("process_inputs") or {}) if c in df.columns]
    if not ins:
        critical.append("未找到任何输入变量 X")
    elif not any(pd.to_numeric(df[c], errors="coerce").notna().sum() >= 20
                 and pd.to_numeric(df[c], errors="coerce").nunique() > 1 for c in ins):
        critical.append("输入变量 X 均缺少足够的有效数值或没有变化")
    for c in ins:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().mean() > 0.6:
            warning.append(f"输入 {c} 缺失率超过 60%")
        elif s.nunique() <= 1:
            warning.append(f"输入 {c} 为常数（对模型无信息量）")
        if not pd.api.types.is_numeric_dtype(df[c]):
            warning.append(f"输入 {c} 为非数值列（已按分类处理，不进入模型特征）")
    if "batch_id" in df.columns and df["batch_id"].nunique() < 4:
        warning.append(f"独立批次仅 {df['batch_id'].nunique()} 个，分组交叉验证将回退 KFold/LOOCV")
    if "experiment_id" in df.columns and df["experiment_id"].duplicated().any():
        warning.append("experiment_id 存在重复，请核对是否为重复录入")
    return critical, warning


# ---------------- 字段自动识别（spec 二十七） ----------------
def _all_canonical_fields():
    fields = {}
    for mode_id in MODES:
        try:
            sch = load_schema_for_mode(mode_id)
        except Exception:
            continue
        for sec, cols in sch.items():
            if not isinstance(cols, dict):
                continue
            for c, spec in cols.items():
                fields.setdefault(c, {"role": sec, "categorical": bool(spec.get("categorical")),
                                      "modes": []})
                fields[c]["modes"].append(mode_id)
    return fields


def detect_mode(df):
    """根据列名与四种模式 Schema 的重合度判断研究模式（高置信度才自动，否则 literature）。"""
    cols = set(df.columns)
    scores = {}
    for mode_id in MODES:
        try:
            sch = load_schema_for_mode(mode_id)
        except Exception:
            continue
        known = set()
        for sec in ["structure_inputs", "process_inputs"]:
            known |= {c for c in (sch.get(sec) or {}) if not (sch[sec][c].get("categorical"))}
        hit = len(cols & known)
        scores[mode_id] = hit / max(1, len(known)) if known else 0
    best = max(scores, key=scores.get)
    return best if scores[best] >= 0.5 else "literature", scores


def infer_field_map(df):
    """自动字段映射：返回 schema 选择 + 歧义清单。

    高置信度（列名与规范字段精确一致，或命中别名表）→ 自动；
    其余数值列 → 歧义（可能为 X / Y / 材料），交由用户确认。
    """
    canon = {c.lower(): info for c, info in _all_canonical_fields().items()}
    sel = {"x": [], "states": [], "defects": [], "performance": [],
           "material": [], "melting_map": {}}
    ambiguous = {}
    used = set()
    for col in df.columns:
        key = str(col).lower().strip()
        if key in IDENTIFIER_COLUMNS:
            used.add(col)
            continue
        if key in LITERATURE_ROLE_ALIASES:
            sel[LITERATURE_ROLE_ALIASES[key]].append(col)
            used.add(col)
            continue
        if key in canon:
            role = canon[key]["role"]
            target = {"structure_inputs": "x", "process_inputs": "x",
                      "process_states": "states", "defect_network": "defects",
                      "performance_outputs": "performance",
                      "material_inputs": "material"}.get(role)
            if target == "x" and canon[key]["categorical"]:
                used.add(col)
                continue
            if target:
                sel[target].append(col)
                used.add(col)
            elif role == "melting_states":
                sel["melting_map"][key] = col
                used.add(col)
            continue
        alias = GENERIC_ALIAS.get(key)
        if alias and alias in df.columns:
            used.add(col)
            continue
        if alias and alias.lower() in canon:
            role = canon[alias.lower()]["role"]
            if role == "melting_states":
                sel["melting_map"][alias] = col
            else:
                target = {"structure_inputs": "x", "process_inputs": "x",
                          "process_states": "states", "defect_network": "defects",
                          "performance_outputs": "performance",
                          "material_inputs": "material"}.get(role)
                if target:
                    sel[target].append(alias if alias in df.columns else col)
            used.add(col)
            continue
        # 未知数值列 → 歧义
        if pd.api.types.is_numeric_dtype(df[col]):
            ambiguous[col] = "可能是输入 X / 目标 Y / 材料属性"
    return sel, ambiguous


def read_objective_directions(sheets_df):
    """从元数据表读取 optimize 方向（maximize/minimize/prediction_only）。没有 → 全部 prediction_only。"""
    directions = {}
    for name, df in (sheets_df or {}).items():
        low = str(name).lower()
        if not any(k in low for k in META_SHEET_KEYWORDS):
            continue
        for _, row in df.iterrows():
            cells = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            for i, cell in enumerate(cells):
                if cell in ("maximize", "minimize", "prediction_only"):
                    # 找同行里像字段名的单元格（含下划线/字母）
                    for other in cells:
                        if other not in ("maximize", "minimize", "prediction_only") \
                                and ("_" in other or other.replace(".", "").isalpha()) and len(other) > 3:
                            directions[other] = cell
                            break
    return directions


# ---------------- 一键编排（spec 二十五/三十/六十五） ----------------
def classify_sheets(sheet_names):
    data, meta = [], []
    for sn in sheet_names:
        low = str(sn).lower()
        if any(k in low for k in META_SHEET_KEYWORDS):
            meta.append(sn)
        else:
            data.append(sn)
    return data, meta


def run_quick_analysis(df, sheet_name="Data", run_id=None, confirm_map=None,
                       progress_cb=None, demo=False):
    """对单个数据表执行一键科研分析（分步容错，单步失败不丢失其他结果）。

    返回 summary dict：run_id / mode / steps(状态) / schema / bundle / paths / record。
    模型保存于 runs/quick_analysis/<run_id>/model.joblib，绝不写入正式 models/。
    """
    from .model_chain import train_chain_full, predict_chain
    from .paper_output import PaperExporter
    from . import insights as insight_mod
    from . import figure_analysis as fa

    run_id = run_id or ("QA_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = QUICK_ROOT / run_id
    results_dir = run_dir / "results"
    steps = []  # (名称, 状态, 原因)  状态: done/skip/fail

    # V1.6（P0-2）：证据注册表随 run 创建，沿调用链传递（生成即注册，非事后包装）
    registry = EvidenceRegistry(run_id, run_dir, demo=demo)
    manifest_path = None
    summary_path = None

    def step(name, fn, skippable=False, skip_reason=None):
        if skippable and skip_reason:
            steps.append((name, "skip", skip_reason))
            if progress_cb:
                progress_cb(name, "skip", skip_reason)
            return None
        if progress_cb:
            progress_cb(name, "running", "")
        try:
            out = fn()
            steps.append((name, "done", ""))
            if progress_cb:
                progress_cb(name, "done", "")
            return out
        except Exception as e:
            steps.append((name, "fail", str(e)))
            if progress_cb:
                progress_cb(name, "fail", str(e))
            from .research_utils import log_event
            log_event("quick_analysis_error", f"run={run_id} step={name}: {e}")
            return None

    # ② 数据准备检查
    mode_id, scores = detect_mode(df)
    sel, ambiguous = infer_field_map(df)
    if confirm_map:  # 用户确认歧义字段：{col: "x"/"performance"/"material"/"ignore"}
        for col, role in confirm_map.items():
            if role in ("x", "performance", "material", "defects", "states") and col in df.columns:
                sel[role].append(col)
            ambiguous.pop(col, None)

    schema = None

    def _build_schema():
        nonlocal schema
        if mode_id == "literature":
            schema = build_runtime_schema(df, sel)
        else:
            schema = load_schema_for_mode(mode_id)
        critical, warning = data_readiness_issues(df, schema)
        if critical:
            raise RuntimeError("数据准备检查未通过：" + "；".join(critical))
        return {"critical": critical, "warning": warning}

    check = step("数据读取与检查", lambda: None)  # ①
    check = step("数据准备检查", _build_schema)
    if check is None:
        return _finish(run_id, mode_id, steps, None, schema,
                       {"warnings": [], "unmapped_columns": list(ambiguous)},
                       sheet_name, demo, failed=True)

    # ④ 训练（Quick Run 模型目录，不覆盖正式模型）
    bundle = {}
    def _train():
        p = run_dir / "model.joblib"
        b = train_chain_full(df, schema, p, n_splits=None)
        bundle.update(b)
        # V1.6：训练成功后回填证据注册表的版本信息（meta 类溯源字段）
        registry.model_version = b.get("model_version")
        registry.data_version = ("auto_" + str(b.get("dataset_hash"))[:8]) \
            if b.get("dataset_hash") else None
        return b
    step("模型训练（保存于 runs/quick_analysis，不影响正式模型）", _train)
    if not bundle:
        return _finish(run_id, mode_id, steps, None, None, {}, sheet_name, demo, failed=True)

    directions = confirm_map.get("_directions") if isinstance(confirm_map, dict) else None
    objectives = {c: d for c, d in (directions or {}).items()
                  if d in ("maximize", "minimize")
                  and (c in bundle.get("stage3_models", {}) or
                       c in bundle.get("stage2_models", {}))}
    if step("保存运行数据与字段映射",
            lambda: save_quick_workspace(run_id, df, schema, bundle, objectives)) is None:
        return _finish(run_id, mode_id, steps, bundle, schema, {}, sheet_name, demo,
                       failed=True)

    # V1.8.1：结果文件名前缀统一计算一次（与 PaperExporter 的 _package 一致）
    prefix = "QUICK_" + MODES[mode_id]["prefix"].get("literature", "LIT_")

    # V1.8.1（要求 E）：逐行折外预测导出——每个 Stage 1 目标一份 OOF CSV
    # （row_index / fold / actual / predicted / residual），失败不阻断主流程。
    def _export_oof():
        return {"oof_csv": [str(p) for p in
                            export_stage1_oof_csv(results_dir, df, bundle, prefix)]}
    oof_export = step("逐行折外预测导出", _export_oof)

    # ⑥ 模型解释 + 熔融状态（信息汇总）
    def _explain():
        factors, stability = insight_mod.model_key_factors(bundle)
        return {"factors": {k: v.head(8).to_dict() for k, v in factors.items()},
                "stability": stability}
    explain = step("模型解释（特征重要性 / 跨折稳定性）", _explain)

    # ⑦ 熔融状态分析（未启用自动跳过）
    melt_info = {"enabled": bool(bundle.get("stage15_enabled"))}
    step("熔融状态分析", lambda: melt_info,
         skippable=not bundle.get("stage15_enabled"),
         skip_reason="当前数据缺少颗粒熔融状态标签，Stage 1.5 自动跳过")

    # ⑧ Pareto（仅当方向已定义；禁止自动认定优化方向）
    pareto_result = {}
    if not bundle.get("stage3_models") or len(objectives) < 2:
        # 默认全部 prediction_only：跳过 Pareto（spec 二十九）
        step("多目标优化（Pareto）", lambda: None, skippable=True,
             skip_reason=("缺少可训练的涂层性能目标；当前数据仅能验证过程层，无法计算涂层性能 Pareto 前沿"
                          if not bundle.get("stage3_models") else
                          "未定义至少两个已训练的涂层缺陷/性能目标的优化方向（prediction_only），未执行多目标优化"))
    else:
        def _pareto():
            from .optimize import random_pareto_search
            from .config import load_objectives
            obj_cfg = {"objectives": objectives, "constraints": {}}
            dom = bundle.get("training_domain") or {}
            fixed = {c: dom[c]["median"] for c in (schema.get("structure_inputs") or {})
                     if c in dom}
            front, stats = random_pareto_search(bundle, schema, obj_cfg,
                                                fixed_values=fixed,
                                                n_candidates=1200, return_stats=True)
            pareto_result.update({"front": front, "stats": stats})
            return stats
        step("多目标优化（Pareto）", _pareto)

    # ⑨ 数据洞察 + ⑩ 图表与结果包
    def _insights():
        from .context import build_context
        ctx = build_context(mode_id, "文献数据（快速分析）", df=df, bundle=bundle,
                            active_run_id=run_id)
        ctx["_demo"] = demo
        # V1.6（P0-2）：registry 传入洞察引擎——结论句取值即拼接 + 缺证降级
        md, xlsx = insight_mod.build_insight_report(ctx, df, schema, bundle,
                                                    results_dir / "insights",
                                                    registry=registry)
        return {"report": str(md), "evidence": str(xlsx)}
    insight_out = step("数据洞察", _insights)

    # V1.6（P0-2/P0-3）：成功路径落盘 summary.json + evidence_manifest.json（摘要卡数据源）。
    # 注意顺序（两层）：
    # ① 先 build_smart_summary（注册 qa.* 证据），再 registry.save 落盘，
    #    保证摘要卡每条结论都能在 manifest 中溯源（P0-3 验收 2）；
    # ② 落盘须在 _package 之前——PaperExporter.build_bundle 打包时从
    #    run_dir.parent 复制 evidence_manifest.json 进 ZIP（P0-2 验收 6）。
    manifest_path = None
    summary_path = None
    if bundle:
        try:
            smart = build_smart_summary(
                bundle, schema, {"explain": explain, "pareto": pareto_result},
                demo=demo, run_id=run_id, registry=registry, df=df, mode_id=mode_id)
            summary_path = run_dir / SUMMARY_NAME
            summary_path.write_text(json.dumps(smart, ensure_ascii=False, indent=1,
                                               default=str), encoding="utf-8")
        except Exception as e:
            steps.append(("智能摘要生成", "fail", str(e)))
            summary_path = None
        try:
            manifest_path = registry.save(run_dir)
        except Exception as e:
            steps.append(("证据清单落盘", "fail", str(e)))
            manifest_path = None

    def _package():
        # V1.8.1：prefix 已在上方统一计算（含 OOF CSV 导出），不再重复生成。
        # V1.6 fix（QA 回归问题 1）：registry 传入 PaperExporter——
        # 否则 build_bundle 模块 I 会以 registry=None 重新生成洞察报告，
        # 覆盖 _insights() 的证据化版本，且 fig.* 证据不注册。
        ex = PaperExporter(df, schema, bundle, lang="zh", formats=("png",),
                           demo=demo, prefix=prefix, run_dir=results_dir,
                           registry=registry,
                           objective_cfg={"objectives": objectives, "constraints": {}})
        run_d, zip_p, paths = ex.build_bundle(n_candidates=800)
        return {"run_dir": str(run_d), "zip": str(zip_p),
                "figures": len(paths.get("figures", [])),
                "analysis_dir": str(run_d / "analysis")}
    pkg = step("科研图表与结果输出", _package)

    # V1.6 fix（QA 回归问题 1 补充）：_package 期间 PaperExporter 会注册 fig.* 证据，
    # 此处再次落盘 run_dir 根的 manifest，使 fig.* 条目进入摘要卡数据源文件
    # （ZIP 内的完整 manifest 已由 PaperExporter 在 build_bundle 末尾落盘）。
    if bundle and manifest_path is not None:
        try:
            manifest_path = registry.save(run_dir)
        except Exception as e:
            steps.append(("证据清单更新", "fail", str(e)))

    return _finish(run_id, mode_id, steps, bundle, schema,
                   {"explain": explain, "insight": insight_out, "package": pkg,
                    "oof_export": oof_export, "pareto": pareto_result,
                    "warnings": (check or {}).get("warning", [])},
                   sheet_name, demo, failed=pkg is None,
                   manifest_path=manifest_path, summary_path=summary_path)


def _finish(run_id, mode_id, steps, bundle, schema, extras, sheet_name, demo,
            failed=False, manifest_path=None, summary_path=None):
    record = {
        "run_id": run_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "research_mode": mode_id,
        "data_source": "quick_analysis",
        "sheet": sheet_name,
        "dataset_version": ("auto_" + str(bundle.get("dataset_hash"))[:8]) if bundle else None,
        "model_version": bundle.get("model_version") if bundle else None,
        "targets": list((bundle or {}).get("stage3_models", {}).keys())[:8],
        "validation_method": ((bundle or {}).get("chain_cv") or {}).get("method"),
        "stage15": bool((bundle or {}).get("stage15_enabled")),
        "result_path": str(QUICK_ROOT / run_id),
        # V1.6（P0-2/P0-3）：增量字段——旧字段一律不动，向后兼容
        "evidence_manifest": str(manifest_path) if manifest_path else None,
        "summary": str(summary_path) if summary_path else None,
        "failed": failed,
    }
    if not failed:
        append_history(record)
    summary = {"run_id": run_id, "mode": mode_id, "steps": steps, "record": record,
               "bundle": bundle, "schema": schema, **extras}
    return summary


# ---------------- V1.6 智能分析摘要卡（P0-3） ----------------
def _slug(name):
    """证据 ID 语义 slug：小写 + 去除非字母数字（§7.2 禁止中文与空格）。"""
    return "".join(ch for ch in str(name) if ch.isalnum()).lower()


_STAGE_LABEL_SMART = {"stage1": "一级模型", "stage15": "Stage 1.5",
                      "stage2": "二级模型", "stage3": "三级模型"}


def build_smart_summary(bundle, schema, extras=None, *, demo=False, run_id=None,
                        registry=None, df=None, mode_id=None):
    """构建智能分析摘要卡数据（P0-3），结论句由 registry.statement 取值拼接（P0-2）。

    返回 dict（写入 runs/quick_analysis/<run_id>/summary.json）：
    header（Run 元信息）/ stages（各 Stage 目标 R²/RMSE/n）/ top_features /
    pareto / coverage / conclusions（3–6 条证据化结论）。
    registry=None 时结论区为空（旧调用路径兼容，不报错）。
    """
    extras = extras or {}
    bundle = bundle or {}
    mode_id = mode_id or "literature"
    mode_label = MODES.get(mode_id, {}).get("label", mode_id)
    cv = bundle.get("chain_cv") or {}
    metrics = cv.get("metrics") or {}
    oof = cv.get("oof_predictions") or {}
    cv_method = cv.get("method")
    # V1.8.1：结论句中的 CV 文案改为按实际执行方法生成的显示文案
    batches = int(df["batch_id"].nunique()) if (df is not None and "batch_id" in df.columns) else None
    cv_method_display = cv_display_method(cv_method, batches)
    data_version = ("auto_" + str(bundle.get("dataset_hash"))[:8]) \
        if bundle.get("dataset_hash") else None
    model_version = bundle.get("model_version")
    n = int(len(df)) if df is not None else None

    # 各 Stage 目标指标（真实计算值；n 取 OOF 有效样本数）
    def _oof_n(oof_all, stage, t):
        """OOF 有效样本数；oof 各 stage 可能为 DataFrame（列为目标）或 dict（值为 Series）。"""
        so = oof_all.get(stage) if hasattr(oof_all, "get") else None
        if so is None:
            return None
        try:
            if hasattr(so, "columns") and t in so.columns:      # DataFrame
                return int(so[t].dropna().shape[0])
            if hasattr(so, "get"):                              # dict of Series
                s = so.get(t)
                return int(s.dropna().shape[0]) if s is not None else None
        except Exception:
            return None
        return None

    stages = {}
    for stage in ["stage1", "stage15", "stage2", "stage3"]:
        rows = []
        for t, met in (metrics.get(stage) or {}).items():
            n_oof = _oof_n(oof, stage, t)
            row = {"target": t, "n": n_oof}
            if "R2" in met:
                row.update({"R2": float(met["R2"]), "RMSE": float(met["RMSE"]),
                            "MAE": float(met["MAE"])})
            else:
                row.update({"Accuracy": met.get("Accuracy"),
                            "Balanced_Accuracy": met.get("Balanced_Accuracy"),
                            "F1_macro": met.get("F1_macro")})
            rows.append(row)
        if rows:
            stages[stage] = rows

    # Top 5 特征重要性（一级模型首个目标）
    top_features = []
    try:
        t0 = list(bundle["stage1_models"].keys())[0]
        pipe = bundle["stage1_models"][t0]
        imp = pd.Series(pipe.named_steps["model"].feature_importances_,
                        index=bundle.get("stage1_features") or []).sort_values(ascending=False).head(5)
        top_features = [{"feature": str(f), "importance": float(v)} for f, v in imp.items()]
    except Exception:
        top_features = []

    pareto_stats = {}
    pareto = extras.get("pareto") or {}
    if isinstance(pareto, dict) and pareto.get("stats"):
        pareto_stats = pareto["stats"]

    coverage = None
    if df is not None:
        try:
            from .insights import coverage_text
            coverage = coverage_text(df, schema)
        except Exception:
            coverage = None

    # 结论区（3–6 条，全部 registry.statement 产出；required 缺失自动降级）
    conclusions = []
    if registry is not None:
        if n is not None:
            batch_text = ("本次分析共读取 {n} 条记录、{batches} 个独立喷涂批次"
                          "（数据版本 {data_version}）。" if batches is not None else
                          "本次分析共读取 {n} 条记录，未提供独立喷涂批次标识"
                          "（数据版本 {data_version}）。")
            txt = registry.statement(
                "qa.n_samples",
                batch_text,
                source_class="direct", computed_by="quick_analysis.run_quick_analysis",
                values={"n": n, "batches": batches,
                        "data_version": data_version or "未知"},
                required=("n", "batches") if batches is not None else ("n", "data_version"))
            conclusions.append({"eid": "qa.n_samples", "text": txt,
                                "source_class": "direct"})
        for stage in ["stage1", "stage2", "stage3"]:
            rows = stages.get(stage) or []
            if not rows or "R2" not in rows[0]:
                continue
            r0 = rows[0]
            eid = f"qa.{stage}_r2_{_slug(r0['target'])}"
            txt = registry.statement(
                eid,
                "在当前模型中，{stage_label}目标 {target} 的交叉验证 R²={r2:.3f}、"
                "RMSE={rmse:.3g}（n={n}，CV: {cv_method}，折外预测非训练集拟合值）。",
                source_class="model", computed_by="model_chain.train_chain_full",
                values={"stage_label": _STAGE_LABEL_SMART.get(stage, stage),
                        "target": r0["target"], "r2": r0["R2"], "rmse": r0["RMSE"],
                        "n": r0["n"] if r0["n"] is not None else n or 0,
                        "cv_method": cv_method or "未记录"},
                required=("r2", "rmse", "n", "cv_method"))
            conclusions.append({"eid": eid, "text": txt, "source_class": "model"})
        if top_features:
            t0_name = ""
            try:
                t0_name = list(bundle["stage1_models"].keys())[0]
            except Exception:
                t0_name = "一级模型首目标"
            f0 = top_features[0]
            txt = registry.statement(
                "qa.top_feature",
                "在当前模型中（CV: {cv_method}），对 {target} 影响最大的变量为 "
                "{feature}（重要性 {imp:.3f}）。",
                source_class="model", computed_by="insights.model_key_factors",
                values={"cv_method": cv_method_display, "target": t0_name,
                        "feature": f0["feature"], "imp": f0["importance"]},
                required=("feature", "imp", "cv_method"))
            conclusions.append({"eid": "qa.top_feature", "text": txt,
                                "source_class": "model"})
        if pareto_stats.get("n_pareto"):
            txt = registry.statement(
                "qa.pareto_count",
                "在当前模型与约束下（CV: {cv_method}），共搜索到 {n_pareto} 个 Pareto "
                "非支配方案（候选 {n_candidates}，满足约束 {n_feasible}）。",
                source_class="model", computed_by="optimize.random_pareto_search",
                values={"cv_method": cv_method_display,
                        "n_pareto": int(pareto_stats["n_pareto"]),
                        "n_candidates": int(pareto_stats.get("n_candidates", 0)),
                        "n_feasible": int(pareto_stats.get("n_feasible", 0))},
                required=("n_pareto", "cv_method"))
            conclusions.append({"eid": "qa.pareto_count", "text": txt,
                                "source_class": "model"})
        conclusions = conclusions[:6]  # 摘要卡结论区 3–6 条

    return {
        "run_id": run_id,
        "mode": mode_id,
        "mode_label": mode_label,
        "demo": bool(demo),
        "date": datetime.now().isoformat(timespec="seconds"),
        "header": {"n": n, "batches": batches, "data_version": data_version,
                   "model_version": model_version,
                   "cv_method": cv_method,
                   "cv_method_display": cv_method_display,
                   "stage_scope": stage_scope_label(
                       [s for s in ["stage1", "stage15", "stage2", "stage3"]
                        if bundle.get(f"{s}_models")]),
                   "stage15": bool(bundle.get("stage15_enabled"))},
        "stages": stages,
        "top_features": top_features,
        "pareto": ({"n_pareto": int(pareto_stats.get("n_pareto", 0)),
                    "n_candidates": int(pareto_stats.get("n_candidates", 0)),
                    "n_feasible": int(pareto_stats.get("n_feasible", 0))}
                   if pareto_stats else None),
        "coverage": coverage,
        "conclusions": conclusions,
        "failed": False,
    }


def load_smart_summary(run_id):
    """读取 runs/quick_analysis/<run_id>/summary.json；不存在/损坏 → None。"""
    p = QUICK_ROOT / str(run_id) / SUMMARY_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _unique_run_id(base):
    """生成不与历史/现存目录冲突的 run_id。

    同一秒内连续两次分析（如同一文件在 demo 勾选变化后再次运行）不得复用
    同一 run 目录互相覆盖——要求 F 的前提保障。
    """
    rid = base
    n = 2
    existing = {str(r.get("run_id")) for r in load_history()}
    while (QUICK_ROOT / rid).exists() or rid in existing:
        rid = f"{base}_{n}"
        n += 1
    return rid


def quick_analyze_file(file_or_path, run_prefix=None, progress_cb=None, demo=False,
                       confirm_maps=None):
    """对整个 Excel/CSV 执行快速分析：多数据 Sheet → 多个独立 Run（绝不合并）。"""
    from .literature import list_sheets, load_sheet
    name = str(getattr(file_or_path, "name", file_or_path)).lower()
    if name.endswith(".csv"):
        sheets = {"CSV": pd.read_csv(file_or_path)}
        sheet_names = ["CSV"]
    else:
        sheet_names = list_sheets(file_or_path)
        data_sheets, _meta = classify_sheets(sheet_names)
        sheets = {}
        meta_frames = {sn: pd.read_excel(file_or_path, sheet_name=sn) for sn in _meta}
        for sn in data_sheets:
            sheets[sn] = pd.read_excel(file_or_path, sheet_name=sn)
    directions = read_objective_directions(meta_frames if name.endswith((".xlsx", ".xls")) else {})
    results = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, (sn, df) in enumerate(sheets.items()):
        rid = _unique_run_id(f"{run_prefix or 'QA'}_{ts}_{i+1:02d}")
        confirm = dict((confirm_maps or {}).get(sn) or {})
        original_columns = {str(c).lower(): c for c in df.columns}
        sheet_directions = {original_columns[k.lower()]: v for k, v in directions.items()
                            if k.lower() in original_columns}
        confirm["_directions"] = {**sheet_directions, **(confirm.get("_directions") or {})}
        res = run_quick_analysis(df, sheet_name=sn, run_id=rid,
                                 confirm_map=confirm, progress_cb=progress_cb, demo=demo)
        results.append(res)
    return results
