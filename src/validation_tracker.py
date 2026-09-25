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
from .calibration import apply_calibration, load_factors

VAL_DIR = ROOT / "outputs" / "validation"
HISTORY_FILE = VAL_DIR / "validations.json"


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
            elif sec == "meta" and col == "batch_id":
                filled[col] = [version] * len(top)      # 并入数据后仍可 batch 分组
    sheet = pd.DataFrame({c: filled.get(c, [None] * len(top)) for c in col_order})
    # 保证 experiment_id 列存在并填 REC 编号（置于首列）
    eid_col = "experiment_id"
    if eid_col not in sheet.columns:
        sheet.insert(0, eid_col, None)
    sheet[eid_col] = [f"REC-{version}-{i+1:03d}" for i in range(len(top))]
    sheet = sheet[[eid_col] + [c for c in sheet.columns if c != eid_col]]

    # 使用说明 sheet（回灌操作指引）
    guide = pd.DataFrame({
        "步骤": [
            "1. 安排实验", "2. 填写实测值", "3. 上传判定", "4. 并入训练数据（可选）",
            "备注 A", "备注 B", "备注 C"],
        "说明": [
            f"本任务单包含 {len(top)} 个推荐候选工艺点（experiment_id 以 REC-{version}- 开头）。",
            "在 03/04 类列（射流/粒子状态、缺陷、涂层性能）填入实验实测值；输入参数列已预填推荐值，一般不修改。",
            "回到平台「⑥ 数据洞察与实验反馈 → 推荐-验证闭环」上传本文件，自动判定推荐是否命中（1σ / 2σ 双口径）。",
            "验证后的数据可并入正式训练集：在「② 数据管理」上传本文件即可（batch_id 已预填版本号，满足分组防泄漏要求）。",
            "候选预测均值与校准后 σ 已存档于 outputs/validation/candidates_" + version + ".csv，判定以此为准。",
            "当前为演示数据模式：所有预测与判定仅供功能验证。" if demo else "σ 已按 L1 校准因子修正（无校准文件时因子=1）。",
            "严禁修改 experiment_id 编号规则，否则无法匹配预测记录。"]})

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
    cand.to_csv(cand_path, index=False, encoding="utf-8-sig")

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


def judge_validation(uploaded_df, *, demo=False):
    """对回灌的验证任务单执行命中判定。

    返回报告 dict：{"version", "n_judged", "records", "hit_rate_1sigma",
    "hit_rate_2sigma", "per_target", "worst_target", "rounds_history"}；
    同时更新 validations.json 与 验证命中率报告.md，并自动刷新核心价值总报告。
    """
    if uploaded_df is None or len(uploaded_df) == 0:
        raise ValueError("上传的验证任务单为空。")
    if "experiment_id" not in uploaded_df.columns:
        raise ValueError("上传文件缺少 experiment_id 列（须使用平台导出的验证任务单，勿改动列结构）。")

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
            if pd.isna(true_v) or pred is None or pd.isna(pred):
                t_rec[t] = {"pred": None if pred is None or pd.isna(pred) else float(pred),
                            "std_cal": None if std is None or pd.isna(std) else float(std),
                            "true": None if pd.isna(true_v) else float(true_v),
                            "hit_1sigma": None, "hit_2sigma": None,
                            "note": "预测或实测值缺失，待填写。"}
                continue
            std_v = float(std) if (std is not None and not pd.isna(std)) else None
            if not std_v or std_v <= 1e-12:
                t_rec[t] = {"pred": float(pred), "std_cal": std_v, "true": float(true_v),
                            "hit_1sigma": None, "hit_2sigma": None,
                            "note": "候选 σ 不可用（≤0），无法判定区间命中。"}
                continue
            dev = abs(float(true_v) - float(pred))
            t_rec[t] = {"pred": float(pred), "std_cal": std_v, "true": float(true_v),
                        "abs_dev": round(dev, 6), "z": round((float(true_v) - float(pred)) / std_v, 4),
                        "hit_1sigma": bool(dev <= 1.0 * std_v),
                        "hit_2sigma": bool(dev <= 2.0 * std_v)}
        records.append({"experiment_id": rec_id, "matched": True,
                        "version": info["version"], "targets": t_rec})

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
    per_target = {}
    for r in records:
        if not r.get("matched"):
            continue
        for t, tr in (r.get("targets") or {}).items():
            if tr.get("hit_1sigma") is None:
                continue
            per_target.setdefault(t, {"n": 0, "hit1": 0, "hit2": 0, "abs_z": []})
            d = per_target[t]
            d["n"] += 1
            d["hit1"] += bool(tr["hit_1sigma"])
            d["hit2"] += bool(tr["hit_2sigma"])
            d["abs_z"].append(abs(tr.get("z", 0.0)))
    worst_target = None
    if per_target:
        worst_target = max(per_target, key=lambda t: float(np.mean(per_target[t]["abs_z"])))
        for t, d in per_target.items():
            d["hit_rate_1sigma"] = d["hit1"] / d["n"] if d["n"] else None
            d["hit_rate_2sigma"] = d["hit2"] / d["n"] if d["n"] else None
            d["mean_abs_z"] = float(np.mean(d["abs_z"]))
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
        "per_target": per_target,
        "worst_target": worst_target,
        "demo": bool(demo),
    }

    # ---- 累积历史 + 落盘 ----
    hist = _load_history()
    hist["rounds"].append({k: report[k] for k in
                           ("judged_at", "version", "n_uploaded", "n_matched",
                            "n_judged_pairs", "hit_rate_1sigma", "hit_rate_2sigma",
                            "per_target", "worst_target")})
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
        lines += [f"## 本轮命中率", "",
                  f"- **1σ 口径：{report['hit_rate_1sigma']*100:.0f}%**（|实测-预测| ≤ 1×σ_cal；"
                  f"共 {report['n_judged_pairs']} 个目标-样本对）",
                  f"- **2σ 口径：{report['hit_rate_2sigma']*100:.0f}%**"]
    else:
        lines += ["## 本轮命中率", "", "- 暂无可判定记录（预测或实测值缺失，待填写）。"]
    if report["per_target"]:
        lines += ["", "## 逐目标命中率", "",
                  "| 性能目标 | 判定数 | 1σ 命中率 | 2σ 命中率 | 平均 |z| |", "|---|---|---|---|---|"]
        for t, d in report["per_target"].items():
            lines.append(f"| {zh(t)} | {d['n']} | {d['hit_rate_1sigma']*100:.0f}% "
                         f"| {d['hit_rate_2sigma']*100:.0f}% | {d['mean_abs_z']:.2f} |")
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


def validation_status():
    """供状态面板/总报告：历史轮数与最新命中率。"""
    hist = _load_history()
    rounds = hist.get("rounds") or []
    if not rounds:
        return {"done": False}
    last = rounds[-1]
    return {"done": True, "n_rounds": len(rounds),
            "last_hit_1sigma": last.get("hit_rate_1sigma"),
            "last_judged_at": last.get("judged_at")}
