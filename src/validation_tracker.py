# -*- coding: utf-8 -*-
"""V1.7 L2 · 推荐-验证闭环追踪（核心价值验证第二层）。

回答的问题：模型推荐的工艺点，做真实实验后能否落在预测区间内（推荐即命中）？

流程（对应指令集阶段二，按平台 Excel 数据流适配）：
  1. 【⑦ 逆向设计】Pareto 寻优完成后，一键导出「验证任务单」：
     - 格式与平台数据模板同构（Experiments 工作表 + 使用说明），可直接回灌判定，
       也可在补齐后经 ② 数据管理 并入训练数据；
     - 输入参数列填入 Top-N 推荐候选值；过程/缺陷/性能列留空待实验填写；
     - experiment_id 按 REC-v<版本号>-001 起编号；batch_id 预填版本号（并入数据
       后仍满足 batch 分组防泄漏要求）；
     - 同时把各候选的校准后预测均值 ± σ 存为 candidates_<版本号>.csv。
  2. 【⑥ 数据洞察】实验完成后上传填好实测值的任务单：
     - 按 experiment_id 匹配候选预测；
     - 逐目标判定 |实测 - 预测| ≤ k·σ_cal（k=1 / 2 双口径，σ 已按 L1 校准因子修正）；
     - 结果累积写入 validations.json，生成验证命中率报告。
  3. 判定完成后自动更新「核心价值验证总报告」（⑤ 汇总层）。

设计铁律（V1.6 继承）：
  - 不修改任何既有模块（不动清单 diff 保持为零）；
  - 判定与命中率全部由真实数值计算拼接，无匹配/缺实测的记录如实标记「待填写」，不编造；
  - 固定随机种子 42。
"""
from pathlib import Path
from datetime import datetime
import glob
import json

import numpy as np
import pandas as pd

from .config import ROOT
from .calibration import apply_calibration, load_factors, is_calibrated_for
from .status_binding import bind_identity, freshness

VAL_DIR = ROOT / "outputs" / "validation"
HISTORY_FILE = VAL_DIR / "validations.json"

# V1.8（方案 §2）：工程达标的默认目标规格（与 benchmark.DEFAULT_SPEC 同源；
# 界面可传入覆盖）。缺实测的目标按「未完成」处理，不算通过也不算失败。
DEFAULT_SPEC = {
    "porosity_pct": {"max": 4.0},
    "bond_strength_MPa": {"min": 55.0},
    "coupled_damage_rate": {"max": 0.6},
}


# ----------------------------------------------------------------------------
# 1) 导出验证任务单（⑦ 逆向设计 → 挂载）
# ----------------------------------------------------------------------------
def export_validation_sheet(front, schema, *, n_top=8, bundle=None, demo=False):
    """从 Pareto 前沿导出验证任务单。

    参数
    ----
    front : pd.DataFrame —— random_pareto_search 返回的前沿表（含输入列与 pred_ 列）
    schema : dict —— 当前模式 Schema（决定任务单列序）
    n_top : int —— 取前沿前 N 个候选（已按第一目标稳定排序）
    bundle : dict —— 当前模型（用于确定 Stage 3 实际建模的目标集合）

    返回 (xlsx_path, version, candidates_df)
    """
    if front is None or len(front) == 0:
        raise ValueError("Pareto 前沿为空，无法导出验证任务单。")
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    version = "v" + datetime.now().strftime("%Y%m%d_%H%M%S")

    top = front.head(int(n_top)).reset_index(drop=True)

    # Stage 3 实际建模的性能目标（任务单需为其留实测列、判定也只判这些目标）
    perf_targets = list((bundle or {}).get("stage3_models") or {})
    if not perf_targets:
        perf_targets = [c[5:] for c in top.columns if c.startswith("pred_")]

    # 任务单列序：meta + structure + process（填值）+ states + defects + performance（留空）
    # V1.8（方案 §2）：batch_id 不再预填模型版本号（模型版本不能冒充实验独立性标识），
    # 留空待实验者填【实际喷涂批次】；recommendation_run_id 记录推荐轮次身份。
    col_order, filled = [], {}
    for sec in ["meta", "structure_inputs", "process_inputs",
                "process_states", "melting_states", "defect_network",
                "performance_outputs", "material_inputs"]:
        for col in (schema.get(sec) or {}):
            if col in col_order:
                continue
            col_order.append(col)
            if sec in ("structure_inputs", "process_inputs") and col in top.columns:
                filled[col] = top[col].to_numpy()
    sheet = pd.DataFrame({c: filled.get(c, [None] * len(top)) for c in col_order})
    # 保证 experiment_id 列存在并填 REC 编号（置于首列）+ 推荐轮次身份列
    eid_col = "experiment_id"
    if eid_col not in sheet.columns:
        sheet.insert(0, eid_col, None)
    sheet[eid_col] = [f"REC-{version}-{i+1:03d}" for i in range(len(top))]
    sheet = sheet[[eid_col] + [c for c in sheet.columns if c != eid_col]]
    sheet.insert(1, "recommendation_run_id", version)

    # 使用说明 sheet（回灌操作指引）
    guide = pd.DataFrame({
        "步骤": [
            "1. 安排实验", "2. 填写实测值", "2b. 填写实际批次（重要）", "3. 上传判定",
            "4. 并入训练数据（可选）", "备注 A", "备注 B", "备注 C", "备注 D"],
        "说明": [
            f"本任务单包含 {len(top)} 个推荐候选工艺点（experiment_id 以 REC-{version}- 开头）。",
            "在 03/04 类列（射流/粒子状态、缺陷、涂层性能）填入实验实测值；输入参数列已预填推荐值，一般不修改。",
            "batch_id 列请填【实际喷涂批次号】（同一炉/同一次连续喷涂的候选填同一批次号；"
            "实际参数偏离推荐值时在 traverse_speed 等列如实填执行值，平台会同时保留推荐与执行记录）。",
            "回到平台「⑥ 数据洞察与实验反馈 → 推荐-验证闭环」上传本文件，自动分开判定："
            "区间覆盖（|实测-预测| ≤ k·σ_cal）与工程达标（实测值是否满足目标规格）。",
            "验证后的数据可并入正式训练集：在「② 数据管理」上传本文件即可（batch_id 填实际批次后满足分组防泄漏要求）。",
            "候选预测均值与校准后 σ 已存档于 outputs/validation/candidates_" + version + ".csv，判定以此为准（冻结预测）。",
            "当前为演示数据模式：所有预测与判定仅供功能验证。" if demo else "σ 已按 L1 校准因子修正（无校准文件时因子=1，界面会明示未校准）。",
            "严禁修改 experiment_id 编号规则，否则无法匹配预测记录。",
            "推荐身份（REC 编号/版本）与实验身份（实际批次）分开记录，模型版本不冒充实验独立性标识（V1.8）。"]})

    xlsx_path = VAL_DIR / f"验证任务单_{version}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        sheet.to_excel(w, sheet_name="Experiments", index=False)
        guide.to_excel(w, sheet_name="使用说明", index=False)

    # candidates 存档：REC 编号 → 输入 + 校准后预测均值 ± σ
    cand = pd.DataFrame({"experiment_id": sheet[eid_col]})
    for c in top.columns:
        if not c.startswith("pred_") and c in sheet.columns:
            cand[c] = top[c].to_numpy()
    for t in perf_targets:
        pc, sc = f"pred_{t}", f"{t}_std"
        if pc in top.columns:
            cand[pc] = top[pc].to_numpy()
        if sc in top.columns:
            cand[f"{t}_std_model"] = top[sc].to_numpy()
    # 校准后 σ（对齐 apply_calibration 的列名约定 <目标>_std）
    unc = top[[f"{t}_std" for t in perf_targets if f"{t}_std" in top.columns]].copy()
    unc_cal = apply_calibration(unc, load_factors())
    for c in unc_cal.columns:
        cand[c.replace("_std", "_std_cal")] = unc_cal[c].to_numpy()
    cand_path = VAL_DIR / f"candidates_{version}.csv"
    # V1.8：存档携带推荐时模型/数据版本（冻结预测的身份，重训后判定自动提示基线过期）
    cand_meta = {"recommendation_run_id": version,
                 "model_version": (bundle or {}).get("model_version"),
                 "dataset_hash": (bundle or {}).get("dataset_hash"),
                 "exported_at": datetime.now().isoformat(timespec="seconds"),
                 "demo": bool(demo)}
    cand.assign(**{f"__meta_{k}": json.dumps(v, ensure_ascii=False)
                   for k, v in cand_meta.items()}).to_csv(
        cand_path, index=False, encoding="utf-8-sig")

    return xlsx_path, version, cand


# ----------------------------------------------------------------------------
# 2) 回灌判定（⑥ 数据洞察 → 挂载）
# ----------------------------------------------------------------------------
def _load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"rounds": []}
    return {"rounds": []}


def _save_history(hist):
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def _candidate_index():
    """扫描全部 candidates_*.csv 建立 {rec_id: {目标: (pred, std_cal)}} 索引。"""
    index = {}
    for p in sorted(glob.glob(str(VAL_DIR / "candidates_*.csv"))):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "experiment_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            rec = {}
            for c in df.columns:
                if c.startswith("pred_") and not c.endswith("_std"):
                    t = c[5:]
                    sc = f"{t}_std_cal"
                    rec[t] = (row.get(c), row.get(sc) if sc in df.columns else None)
            index[str(row["experiment_id"])] = {"version": Path(p).stem.replace("candidates_", ""),
                                                "targets": rec,
                                                "row": {k: row.get(k) for k in df.columns
                                                        if k != "experiment_id"}}
    return index


def judge_validation(uploaded_df, *, demo=False, spec=None, bundle=None):
    """对回灌的验证任务单执行双口径判定（V1.8 方案 §2/§9）。

    口径一【区间覆盖】：|实测 - 预测| ≤ k·σ_cal（预测准确性，k=1/2）。
    口径二【工程达标】：实测值是否满足目标规格（spec，工程有效性）。
    两者分开统计——落进预测区间但性能不达标 ≠ 推荐成功；
    达标但落不进区间 = 模型待校准但工艺有效。
    联合达标：同一候选全部 spec 目标同时达标（缺实测 → 未完成，不算通过/失败）。

    返回报告 dict；同时更新 validations.json 与 验证命中率报告.md，并自动刷新总报告。
    """
    if uploaded_df is None or len(uploaded_df) == 0:
        raise ValueError("上传的验证任务单为空。")
    if "experiment_id" not in uploaded_df.columns:
        raise ValueError("上传文件缺少 experiment_id 列（须使用平台导出的验证任务单，勿改动列结构）。")
    spec = spec or DEFAULT_SPEC
    _cal_ok, _cal_note = is_calibrated_for(
        (bundle or {}).get("model_version"), (bundle or {}).get("dataset_hash"))

    index = _candidate_index()
    rec_rows = uploaded_df[uploaded_df["experiment_id"].astype(str).str.startswith("REC-")]
    if len(rec_rows) == 0:
        raise ValueError("未找到 REC- 开头的推荐记录行。请使用平台导出的验证任务单并在其中填写实测值。")

    records, versions = [], set()
    n_pending = 0
    for _, row in rec_rows.iterrows():
        rec_id = str(row["experiment_id"])
        info = index.get(rec_id)
        if info is None:
            n_pending += 1
            records.append({"experiment_id": rec_id, "matched": False,
                           "note": "未在候选存档中找到该编号（可能任务单由旧版本导出后被清理）。"})
            continue
        versions.add(info["version"])
        t_rec = {}
        for t, (pred, std) in info["targets"].items():
            if t not in uploaded_df.columns:
                continue
            true_v = pd.to_numeric(pd.Series([row.get(t)]), errors="coerce").iloc[0]
            entry = {"pred": None if pred is None or pd.isna(pred) else float(pred),
                     "std_cal": None if std is None or pd.isna(std) else float(std),
                     "true": None if pd.isna(true_v) else float(true_v),
                     "hit_1sigma": None, "hit_2sigma": None,
                     "meets_spec": None, "note": ""}
            if entry["true"] is None or entry["pred"] is None:
                entry["note"] = "预测或实测值缺失，待填写。"
                t_rec[t] = entry
                continue
            # 口径二：工程达标（实测值 vs 目标规格；spec 无该目标 → None 待定）
            if t in spec:
                s = spec[t]
                if "max" in s and "min" in s:
                    entry["meets_spec"] = bool(s["min"] <= entry["true"] <= s["max"])
                elif "max" in s:
                    entry["meets_spec"] = bool(entry["true"] <= s["max"])
                elif "min" in s:
                    entry["meets_spec"] = bool(entry["true"] >= s["min"])
            if std is None or pd.isna(std) or float(std) <= 1e-12:
                entry["note"] = "候选 σ 不可用（≤0），区间覆盖不可判定；工程达标已单独判定。"
                t_rec[t] = entry
                continue
            dev = abs(entry["true"] - entry["pred"])
            std_v = float(std)
            entry["abs_dev"] = round(dev, 6)
            entry["z"] = round((entry["true"] - entry["pred"]) / std_v, 4)
            entry["hit_1sigma"] = bool(dev <= 1.0 * std_v)
            entry["hit_2sigma"] = bool(dev <= 2.0 * std_v)
            t_rec[t] = entry
        # 联合达标：同一候选全部 spec 目标同时达标（缺实测 → None 未完成）
        spec_ts = [t for t in spec if t in (info["targets"] or {})]
        joint = None
        if spec_ts and all(t in t_rec and t_rec[t]["true"] is not None for t in spec_ts):
            joint = all(t_rec[t].get("meets_spec") is True for t in spec_ts)
        records.append({"experiment_id": rec_id, "matched": True,
                        "version": info["version"], "targets": t_rec,
                        "joint_spec_pass": joint})

    # ---- 汇总命中率（仅统计可判定记录：pred/σ/true 三者齐备）----
    def _rate(k):
        num = den = 0
        for r in records:
            if not r.get("matched"):
                continue
            for t, tr in (r.get("targets") or {}).items():
                if tr.get(f"hit_{k}sigma") is not None:
                    den += 1
                    num += bool(tr[f"hit_{k}sigma"])
        return (num / den, den) if den else (None, 0)

    rate1, n1 = _rate(1)
    rate2, n2 = _rate(2)
    # V1.8：工程达标口径（meets_spec）与联合达标率
    def _spec_rate():
        num = den = 0
        for r in records:
            if not r.get("matched"):
                continue
            for t, tr in (r.get("targets") or {}).items():
                if tr.get("meets_spec") is not None:
                    den += 1
                    num += bool(tr["meets_spec"])
        return (num / den, den) if den else (None, 0)
    spec_rate, n_spec = _spec_rate()
    joint_num = sum(1 for r in records if r.get("joint_spec_pass") is True)
    joint_den = sum(1 for r in records if r.get("joint_spec_pass") is not None)

    per_target = {}
    for r in records:
        if not r.get("matched"):
            continue
        for t, tr in (r.get("targets") or {}).items():
            if tr.get("hit_1sigma") is None and tr.get("meets_spec") is None:
                continue
            per_target.setdefault(t, {"n": 0, "hit1": 0, "hit2": 0,
                                      "spec_ok": 0, "n_spec": 0, "abs_z": []})
            d = per_target[t]
            if tr.get("hit_1sigma") is not None:
                d["n"] += 1
                d["hit1"] += bool(tr["hit_1sigma"])
                d["hit2"] += bool(tr["hit_2sigma"])
                d["abs_z"].append(abs(tr.get("z", 0.0)))
            if tr.get("meets_spec") is not None:
                d["n_spec"] += 1
                d["spec_ok"] += bool(tr["meets_spec"])
    worst_target = None
    if per_target:
        worst_target = max(per_target, key=lambda t: float(np.mean(per_target[t]["abs_z"]))
                           if per_target[t]["abs_z"] else -1)
        for t, d in per_target.items():
            d["hit_rate_1sigma"] = d["hit1"] / d["n"] if d["n"] else None
            d["hit_rate_2sigma"] = d["hit2"] / d["n"] if d["n"] else None
            d["spec_pass_rate"] = d["spec_ok"] / d["n_spec"] if d["n_spec"] else None
            d["mean_abs_z"] = float(np.mean(d["abs_z"])) if d["abs_z"] else None
            del d["abs_z"]

    report = {
        "judged_at": datetime.now().isoformat(timespec="seconds"),
        "version": sorted(versions)[0] if len(versions) == 1 else (sorted(versions) or [None])[0],
        "n_uploaded": int(len(rec_rows)), "n_matched": int(sum(r.get("matched") for r in records)),
        "n_pending": int(n_pending),
        "records": records,
        "n_judged_pairs": n1,
        "hit_rate_1sigma": round(rate1, 4) if rate1 is not None else None,
        "hit_rate_2sigma": round(rate2, 4) if rate2 is not None else None,
        "spec": {t: s for t, s in spec.items()},
        "spec_pass_rate": round(spec_rate, 4) if spec_rate is not None else None,
        "n_spec_pairs": n_spec,
        "joint_pass_count": joint_num, "joint_den": joint_den,
        "joint_pass_rate": round(joint_num / joint_den, 4) if joint_den else None,
        "calibration_note": "" if _cal_ok else _cal_note,
        "per_target": per_target,
        "worst_target": worst_target,
        "demo": bool(demo),
    }
    bind_identity(report, bundle=bundle, spec_version=str(sorted(spec.items())))

    # ---- 累积历史 + 落盘 ----
    hist = _load_history()
    hist["rounds"].append({k: report[k] for k in
                           ("judged_at", "version", "n_uploaded", "n_matched",
                            "n_judged_pairs", "hit_rate_1sigma", "hit_rate_2sigma",
                            "spec_pass_rate", "n_spec_pairs",
                            "joint_pass_count", "joint_den", "joint_pass_rate",
                            "per_target", "worst_target", "binding")})
    _save_history(hist)
    _write_hit_report(hist, report)
    report["rounds_history"] = hist["rounds"]

    # 判定完成后自动刷新核心价值总报告（轻量汇总，失败不阻塞判定结果）
    try:
        from .core_value_report import generate_core_value_report
        generate_core_value_report(trigger="L2 回灌判定完成")
    except Exception:
        pass
    return report


def _write_hit_report(hist, report):
    """验证命中率报告 md——所有数字由判定结果拼接。"""
    from .ui_labels import zh
    lines = ["# L2 · 验证命中率报告（推荐即命中）", "",
             f"- 判定时间：{report['judged_at']}",
             f"- 本轮上传：{report['n_uploaded']} 条 REC 记录；成功匹配 {report['n_matched']} 条；"
             + (f"{report['n_pending']} 条未匹配" if report["n_pending"] else "全部匹配") + "。",
             ""]
    if report["hit_rate_1sigma"] is not None:
        lines += [f"## 本轮判定（V1.8 双口径，分开解读）", "",
                  f"**口径一 · 区间覆盖**（预测准确性：|实测-预测| ≤ k·σ_cal）",
                  f"- 1σ 口径：**{report['hit_rate_1sigma']*100:.0f}%**（"
                  f"共 {report['n_judged_pairs']} 个目标-样本对）",
                  f"- 2σ 口径：**{report['hit_rate_2sigma']*100:.0f}%**", "",
                  f"**口径二 · 工程达标**（实测值是否满足目标规格）"]
        if report["spec_pass_rate"] is not None:
            lines += [f"- 单目标达标率：**{report['spec_pass_rate']*100:.0f}%**"
                      f"（{report['n_spec_pairs']} 对，规格 {report['spec']}）"]
        else:
            lines += ["- 暂无可判定记录（spec 目标的实测值待填写）。"]
        if report["joint_den"]:
            lines += [f"- 同一候选全部目标联合达标：**{report['joint_pass_count']}/"
                      f"{report['joint_den']}**（联合达标率 {report['joint_pass_rate']*100:.0f}%）"]
        else:
            lines += ["- 联合达标：未完成（存在缺测目标，按未完成处理，不算通过/失败）。"]
        if report.get("calibration_note"):
            lines += ["", f"> ⚠ 校准提示：{report['calibration_note']}"]
    else:
        lines += ["## 本轮命中率", "", "- 暂无可判定记录（预测或实测值缺失，待填写）。"]
    if report["per_target"]:
        lines += ["", "## 逐目标判定（覆盖与达标分开）", "",
                  "| 性能目标 | 覆盖判定数 | 1σ 命中率 | 2σ 命中率 | 平均 \\|z\\| | 达标判定数 | 工程达标率 |",
                  "|---|---|---|---|---|---|---|"]
        for t, d in report["per_target"].items():
            h1 = f"{d['hit_rate_1sigma']*100:.0f}%" if d.get("hit_rate_1sigma") is not None else "—"
            h2 = f"{d['hit_rate_2sigma']*100:.0f}%" if d.get("hit_rate_2sigma") is not None else "—"
            mz = f"{d['mean_abs_z']:.2f}" if d.get("mean_abs_z") is not None else "—"
            sp = f"{d['spec_pass_rate']*100:.0f}%" if d.get("spec_pass_rate") is not None else "—"
            lines.append(f"| {zh(t)} | {d['n']} | {h1} | {h2} | {mz} | {d.get('n_spec', 0)} | {sp} |")
    if report["worst_target"]:
        w = report["per_target"][report["worst_target"]]
        lines += ["", f"偏差最大的目标：**{zh(report['worst_target'])}**"
                      f"（平均 |z| = {w['mean_abs_z']:.2f}；建议优先在该参数区域补充实验）。"]
    if len(hist["rounds"]) > 1:
        lines += ["", "## 历史各轮命中率趋势", "",
                  "| 判定时间 | 版本 | 判定对数 | 1σ 命中率 | 2σ 命中率 |", "|---|---|---|---|---|"]
        for r in hist["rounds"]:
            h1 = f"{r['hit_rate_1sigma']*100:.0f}%" if r.get("hit_rate_1sigma") is not None else "—"
            h2 = f"{r['hit_rate_2sigma']*100:.0f}%" if r.get("hit_rate_2sigma") is not None else "—"
            lines.append(f"| {r['judged_at']} | {r.get('version') or '—'} | {r.get('n_judged_pairs', 0)} | {h1} | {h2} |")
    lines += ["", "> 判定基准：candidates 存档中的校准后预测均值 ± σ_cal（L1 校准因子已应用）；",
              "> σ 不可用或实测缺失的记录如实标记「待填写」，不参与命中率也不编造结果。"]
    if report["demo"]:
        lines += ["> 【演示数据】当前结论仅用于功能验证。"]
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    (VAL_DIR / "验证命中率报告.md").write_text("\n".join(lines), encoding="utf-8")


def pending_counts():
    """V1.8 首页工作台待办统计：已导出候选数 vs 已回灌判定数。"""
    n_exported, n_judged = 0, 0
    try:
        for p in sorted(glob.glob(str(VAL_DIR / "candidates_*.csv"))):
            try:
                df_ = pd.read_csv(p)
                n_exported += int(len(df_)) if "experiment_id" in df_.columns else 0
            except Exception:
                pass
    except Exception:
        pass
    try:
        hist = _load_history()
        for r in hist.get("rounds") or []:
            n_judged += int(r.get("n_matched") or 0)
    except Exception:
        pass
    return {"candidates_exported": n_exported, "judged": n_judged,
            "pending_validation": max(0, n_exported - n_judged)}


def validation_status(current_model_version=None, current_dataset_hash=None):
    """供状态面板/总报告：历史轮数、最新双口径命中率与时效性（V1.8）。"""
    hist = _load_history()
    rounds = hist.get("rounds") or []
    if not rounds:
        return {"done": False}
    last = rounds[-1]
    # V1.8：最新一轮若绑定身份则检查时效（推荐基线是旧模型时，覆盖率仅反映当时基线）
    fr_state = "unknown"
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            full = json.load(f)
        rounds_full = full.get("rounds") or []
        if rounds_full and isinstance(rounds_full[-1].get("binding"), dict):
            fr = freshness(rounds_full[-1], current_model_version, current_dataset_hash)
            fr_state = fr["state"]
    except Exception:
        pass
    return {"done": True, "n_rounds": len(rounds),
            "last_hit_1sigma": last.get("hit_rate_1sigma"),
            "last_spec_pass_rate": last.get("spec_pass_rate"),
            "last_joint_pass_rate": last.get("joint_pass_rate"),
            "last_judged_at": last.get("judged_at"),
            "freshness": fr_state}
