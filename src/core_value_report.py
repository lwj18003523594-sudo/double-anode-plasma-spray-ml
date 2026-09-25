# -*- coding: utf-8 -*-
"""V1.7 · 核心价值验证总报告（L1→L4 汇总层）。

汇总四层验证产物，生成 outputs/reports/核心价值验证报告.md：
  L1 不确定度校准   ← models/calibration.json
  L2 推荐即命中     ← outputs/validation/validations.json
  L3 闭环效率       ← outputs/benchmark/效率对比结果.json
  L4 物理一致性     ← outputs/physics/物理一致性结果.json

一页式结论明确回答：平台当前通过到哪一层（L1→L4）、距离核心价值被证明还缺哪条
证据、下一步建议做什么实验。缺失的章节明确写「暂无证据」，不留空、不编造。

下一步建议内置指令集的纠偏逻辑：
  - L2 命中率 < 50%：查校准因子应用 / 补偏差最大目标的稀疏参数区域实验
  - L2 命中率 < 60%：κ 1.0→1.5 收紧悲观惩罚的建议
  - L3 无优势：初始样本 / 目标规格宽松 / 悲观强度逐项排查
  - L4 FAIL 多：先查实现，再查共线性与范围窄，给补实验建议
  - L1 覆盖率低：建议扩充该目标训练数据（平台 Stage 3 已是 Matern+White GP）

触发方式：手动（⑧ 结果输出按钮）+ L2 判定完成后自动刷新（轻量）。
"""
from pathlib import Path
from datetime import datetime
import json

from .config import ROOT

REPORT_PATH = ROOT / "outputs" / "reports" / "核心价值验证报告.md"


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _layer_verdicts(mode_label, demo=False):
    """逐层判定：返回 {layer: {done, passed, detail}}。"""
    out = {}

    # ---- L1 校准 ----
    cal = _load_json(ROOT / "models" / "calibration.json")
    if cal and (cal.get("targets") or {}):
        tg = {t: i for t, i in cal["targets"].items()
              if isinstance(i, dict) and "cov_1sigma" in i}
        if tg:
            good = [t for t, i in tg.items() if 0.55 <= i["cov_1sigma"] <= 0.80]
            out["L1"] = {"done": True, "passed": True if good else False,
                         "detail": f"{len(good)}/{len(tg)} 个目标校准良好；"
                                   + "；".join(f"{t}: 1σ={i['cov_1sigma']*100:.0f}%, k={i['factor']:.2f}"
                                               for t, i in list(tg.items())[:4])}
        else:
            out["L1"] = {"done": True, "passed": False, "detail": "已运行但无有效目标（样本不足）。"}
    else:
        out["L1"] = {"done": False, "passed": None, "detail": "暂无证据（未运行 L1 校准）。"}

    # ---- L2 命中 ----
    val = _load_json(ROOT / "outputs" / "validation" / "validations.json")
    rounds = (val or {}).get("rounds") or []
    if rounds:
        judged = [r for r in rounds if r.get("hit_rate_1sigma") is not None]
        if judged:
            last = judged[-1]
            r1 = last["hit_rate_1sigma"]
            out["L2"] = {"done": True, "passed": r1 >= 0.6,
                         "detail": f"共 {len(judged)} 轮判定；最新 1σ 命中率 {r1*100:.0f}%"
                                   + (f"（2σ {last['hit_rate_2sigma']*100:.0f}%）"
                                      if last.get("hit_rate_2sigma") is not None else "")}
        else:
            out["L2"] = {"done": True, "passed": False,
                         "detail": f"有 {len(rounds)} 轮上传但无可判定记录（实测值待填写）。"}
    else:
        out["L2"] = {"done": False, "passed": None,
                     "detail": "暂无证据（未导出验证任务单或未回灌实测值）。"}

    # ---- L3 效率 ----
    bench = _load_json(ROOT / "outputs" / "benchmark" / "效率对比结果.json")
    if bench and bench.get("feasible") and bench.get("saved_experiments") is not None:
        sv = bench["saved_experiments"]
        out["L3"] = {"done": True, "passed": sv > 0,
                     "detail": f"模型引导 vs 随机：节省 {sv:.0f} 次实验"
                               f"（引导中位 {bench['guided_median_steps']:.0f} vs "
                               f"随机中位 {bench['random_median_steps']:.0f}，"
                               f"{bench['n_seeds']} 种子）"}
    elif bench and bench.get("feasible") is False:
        out["L3"] = {"done": True, "passed": False,
                     "detail": "已运行但无法开展竞赛（数据集无达标点 / 规格过严）。"}
    else:
        out["L3"] = {"done": False, "passed": None, "detail": "暂无证据（未运行 L3 效率基准）。"}

    # ---- L4 物理 ----
    phys = _load_json(ROOT / "outputs" / "physics" / "物理一致性结果.json")
    if phys and (phys.get("summary") or {}):
        s = phys["summary"]
        if s.get("n_checkable"):
            ratio = s["pass_ratio"]
            out["L4"] = {"done": True, "passed": ratio >= 0.7,
                         "detail": f"可检验 {s['n_checkable']} 条：PASS {s['PASS']} ｜ "
                                   f"WEAK {s['WEAK']} ｜ FAIL {s['FAIL']}"
                                   f"（通过率 {ratio*100:.0f}%）"}
        else:
            out["L4"] = {"done": True, "passed": None, "detail": "已运行但无可检验条目。"}
    else:
        out["L4"] = {"done": False, "passed": None,
                     "detail": "暂无证据（未运行 L4 物理一致性检验）。"}
    return out


def _next_steps(verdicts, demo=False):
    """自动生成下一步建议（内置指令集纠偏逻辑）。"""
    tips = []
    L1, L2, L3, L4 = verdicts["L1"], verdicts["L2"], verdicts["L3"], verdicts["L4"]

    if not L1["done"]:
        tips.append("在「③ 模型训练 → L1 不确定度校准」运行校准（约 1–2 分钟），先回答不确定度是否可信。")
    elif not L1["passed"]:
        tips.append("L1 校准存在覆盖偏差的目标：σ 已按校准因子自动修正；若修正后覆盖率仍低，"
                   "建议优先扩充该目标的训练数据（Stage 3 已是 Matern+White GP，核函数无需更换）。")

    if not L2["done"]:
        tips.append("在「⑦ 逆向设计」导出验证任务单并安排喷涂实验；实测值回填后在"
                    "「⑥ 数据洞察 → 推荐-验证闭环」上传判定（推荐即命中，L2）。")
    elif L2["passed"] is False:
        r1_txt = ""
        val = _load_json(ROOT / "outputs" / "validation" / "validations.json")
        rounds = [r for r in ((val or {}).get("rounds") or [])
                  if r.get("hit_rate_1sigma") is not None]
        if rounds:
            last = rounds[-1]
            r1 = last["hit_rate_1sigma"]
            wt = last.get("worst_target")
            if r1 < 0.5 and wt:
                from .ui_labels import zh
                tips.append(f"L2 命中率不达标（1σ {r1*100:.0f}%）：检查校准因子是否已应用；"
                             f"偏差最大的目标是 {zh(wt)}，建议在该目标的稀疏参数区域补充实验。")
            elif r1 < 0.6:
                tips.append(f"L2 命中率略低于 60%（1σ {r1*100:.0f}%）：建议把逆向设计候选的"
                            "悲观惩罚从 κ=1.0 收紧到 1.5 重新寻优，并优先验证不确定度小的候选点。")
        if not any("L2" in t for t in tips):
            tips.append("L2 存在待填写记录：补齐验证任务单中的实测值后重新上传判定。")

    if not L3["done"]:
        tips.append("在「⑥ 数据洞察 → L3 闭环效率基准」运行重放竞赛（演示数据约 2–3 分钟，"
                    "可先用 3 个种子快速预览），拿到「模型引导节省 N 次实验」的核心价值数字。")
    elif L3["passed"] is False:
        tips.append("L3 未跑赢随机基线：逐项排查——初始样本是否 ≥15 且覆盖主参数区、"
                    "目标规格是否过松（达标点占比 >40% 时收紧到前 20% 分位水平重跑）、"
                    "悲观惩罚 κ 可从 1.0 试 1.5。")

    if not L4["done"]:
        tips.append("在「⑤ 模型解析 → L4 物理一致性检验」运行检验，确认模型规律符合喷涂物理。")
    elif L4["passed"] is False:
        tips.append("L4 FAIL 条目偏多：先确认判定实现无误；再检查 FAIL 项是否源于变量共线性或"
                    "变化范围太窄（报告已附针对性补实验建议），按建议补充两端实验点。")

    if not tips:
        tips.append("四层验证证据齐备。建议将「核心价值验证报告」随论文结果包一并归档，"
                    "并持续经 L2 闭环积累多轮命中率（历史趋势会进入报告）。")
    if demo:
        tips.append("【演示数据】当前全部结论仅用于功能验证；真实数据下请重新运行四层验证。")
    return tips


def generate_core_value_report(trigger="manual"):
    """生成总报告；返回 (path, verdicts)。"""
    verdicts = _layer_verdicts("")
    done_layers = [k for k, v in verdicts.items() if v["done"]]
    passed_layers = [k for k, v in verdicts.items() if v.get("passed")]

    # 「通过到哪层」：按 L1→L4 顺序取连续通过段
    deepest = None
    for k in ["L1", "L2", "L3", "L4"]:
        if verdicts[k].get("passed"):
            deepest = k
        else:
            break

    lines = ["# 核心价值验证报告（L1 → L4）", "",
             f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}"
             f"（触发：{trigger}）",
             "- 四层验证体系：L1 不确定度校准 → L2 推荐即命中 → L3 闭环效率 → L4 物理一致性",
             ""]
    lines += ["## 一页式结论", ""]
    if deepest:
        lines.append(f"- **平台核心价值当前通过到：{deepest}**（连续通过 L1→{deepest}）。")
    else:
        lines.append("- 平台核心价值尚未有连续通过的层级，请按下方「下一步建议」逐层补齐证据。")
    lines += [f"- 已运行：{('、'.join(done_layers) if done_layers else '无')}；"
              f"通过：{('、'.join(passed_layers) if passed_layers else '无')}。", "",
              "## 逐层证据状态", "",
              "| 层 | 验证问题 | 状态 | 关键证据 |", "|---|---|---|---|"]
    q = {"L1": "模型给出的不确定度是否可信（校准性）",
         "L2": "推荐的工艺点做实验能否落在预测区间内（推荐即命中）",
         "L3": "模型引导寻优是否比随机试错省实验（闭环效率）",
         "L4": "模型学到的规律是否符合喷涂物理（物理一致性）"}
    for k in ["L1", "L2", "L3", "L4"]:
        v = verdicts[k]
        status = ("✅ 通过" if v.get("passed") else
                  ("⚠️ 已运行未通过" if v["done"] else "— 未运行"))
        lines.append(f"| {k} | {q[k]} | {status} | {v['detail']} |")

    lines += ["", "## 下一步建议", ""]
    for i, t in enumerate(_next_steps(verdicts), 1):
        lines.append(f"{i}. {t}")

    # ---- 详细数据分节（引用各层产物，不重复计算）----
    lines += ["", "---", "## 附：各层详细数据入口", "",
              "- L1：outputs/calibration/校准报告.md（因子存 models/calibration.json）",
              "- L2：outputs/validation/验证命中率报告.md（判定记录 validations.json）",
              "- L3：outputs/benchmark/效率对比报告.md（含对比曲线）",
              "- L4：outputs/physics/物理一致性报告.md（逐条判定与 FAIL 分析）",
              "", "> 各层数值均由真实计算生成；缺失层明确标注「暂无证据」，不编造。"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH, verdicts


def core_value_status():
    """轻量状态（首页徽标用，不读大文件）。"""
    _, verdicts = generate_core_value_report.__wrapped__() if hasattr(
        generate_core_value_report, "__wrapped__") else (None, None)
    # 直接逐层轻量读取（避免在 status 路径重复生成报告）
    v = _layer_verdicts("")
    deepest = None
    for k in ["L1", "L2", "L3", "L4"]:
        if v[k].get("passed"):
            deepest = k
        else:
            break
    return {"layers": v, "deepest": deepest}
