# -*- coding: utf-8 -*-
"""V1.7 L3 · 闭环效率基准对比（核心价值验证第三层——平台核心价值的主证据）。

回答的问题：模型引导寻优是否比随机试错用更少的实验次数达到目标性能？

方法（对应指令集阶段三，按平台三级链式架构适配）：
  用当前数据集做「重放模拟」竞赛——同一初始子集出发，两种策略补点：
    A. 模型引导：每轮用当前已标注子集训练三级链模型，对未标注样本池预测
       （均值 + L1 校准后 σ），按「满足工程约束 + 悲观惩罚（mean ∓ κ·σ_cal）+ 综合
       目标最优」选点加入训练集；
    B. 随机基线：每轮随机补点。
  真实标签全部取自数据集本身（重放模拟的铁律：严禁用模型预测值冒充真值）。

  记录：累计实验次数、是否已找到满足目标规格的点（按真实标签判定）、
  已发现可行点集的归一化超体积（蒙特卡洛估计）。

  核心输出：「达到同等目标规格，模型引导比随机基线平均节省 N 次实验
  （多种子的中位数与区间）」。N ≤ 0 时必须逐项分析原因（初始样本太少 /
  目标规格太宽松 / 数据量不足），不许只给数字不给解释。

目标规格（指令集预填，界面可改）：
  porosity_pct ≤ 4.0  且  bond_strength_MPa ≥ 55  且  coupled_damage_rate ≤ 0.6

设计铁律（V1.6 继承）：不修改任何既有模块；随机种子 42；结论句由真实数值拼接。
"""
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from .config import ROOT
from .model_chain import train_chain, predict_chain
from .calibration import load_factors

BENCH_DIR = ROOT / "outputs" / "benchmark"
RESULT_FILE = BENCH_DIR / "效率对比结果.json"
SEED = 42
KAPPA = 1.0          # 悲观惩罚系数（指令集纠偏项允许调到 1.5）

DEFAULT_SPEC = {
    "porosity_pct": {"max": 4.0},
    "bond_strength_MPa": {"min": 55.0},
    "coupled_damage_rate": {"max": 0.6},
}


def _satisfies(row, spec):
    """真实标签是否满足目标规格（全部目标）。"""
    for t, s in spec.items():
        v = row.get(t)
        if v is None or pd.isna(v):
            return False
        if "max" in s and v > s["max"]:
            return False
        if "min" in s and v < s["min"]:
            return False
    return True


def _norm_hypervolume(points, spec, data, rng, n_mc=3000):
    """已发现可行点集的归一化超体积（MC 估计，∈ [0, 1]）。

    盒定义：每个目标从规格边界（最劣可接受）到数据集最优值方向；
    无可行点时返回 0.0。跨目标量纲无关（逐目标按盒宽归一）。
    """
    if points is None or len(points) == 0:
        return 0.0
    boxes = {}
    for t, s in spec.items():
        vals = pd.to_numeric(data[t], errors="coerce").dropna()
        if len(vals) == 0:
            return 0.0
        if "max" in s:      # minimize：盒 [spec_max, 数据最小值]
            lo, hi = float(s["max"]), float(vals.min())
            if hi >= lo:    # 数据最优不优于规格 → 无达标空间
                return 0.0
            boxes[t] = (lo, hi, "min")
        else:               # maximize：盒 [数据最大值, spec_min]
            lo, hi = float(vals.max()), float(s["min"])
            if lo <= hi:
                return 0.0
            boxes[t] = (hi, lo, "max")   # 统一为 (lower, upper)，maximize 的理想在 lower…见下
    # 采样（统一坐标：u ∈ [0,1]，0 = 规格边界、1 = 数据最优方向）
    n_obj = len(boxes)
    qs = rng.random((n_mc, n_obj))
    # 把已发现点也映射到 u 坐标：minimize 目标 u=(spec_max - v)/(spec_max - data_min)
    pts_u = []
    for _, p in points.iterrows():
        u = []
        ok = True
        for t, (lo, hi, kind) in boxes.items():
            v = p.get(t)
            if v is None or pd.isna(v):
                ok = False
                break
            if kind == "min":
                ui = (lo - v) / (lo - hi)      # v 越小 u 越大
            else:
                ui = (v - lo) / (hi - lo)      # v 越大 u 越大
            if ui < 0:                          # 不满足规格的点不构成体积
                ok = False
                break
            u.append(min(ui, 1.5))
        if ok:
            pts_u.append(u)
    if not pts_u:
        return 0.0
    P = np.asarray(pts_u)
    dominated = np.zeros(n_mc, dtype=bool)
    for u in P:
        dominated |= np.all(qs <= u + 1e-12, axis=1)
    return float(dominated.mean())


def _guided_pick(pool, perf, unc, spec, data, n_batch, kappa=KAPPA):
    """模型引导选点：悲观惩罚评分（向量化）。

    悲观值：minimize → mean + κ·σ_cal；maximize → mean - κ·σ_cal。
    悲观可行点按「深入规格盒的平均深度」取最大；无悲观可行点时按
    「悲观值到规格的归一化距离」取最小（贴近规格的探索）。
    返回 None 表示模型尚未就绪（spec 目标缺列），调用方应回退随机选点。
    """
    idx = pool.index
    # 小样本时 Stage 3 目标可能未达建模门槛（有效值 < 20）——如实回退随机
    if any(t not in perf.columns for t in spec):
        return None
    depth = np.zeros(len(idx))
    dist = np.zeros(len(idx))
    for t, s in spec.items():
        mean = perf[t].to_numpy(float)
        std = (unc[t + "_std"].to_numpy(float) if t + "_std" in unc
               else np.zeros(len(idx)))
        pess = mean + kappa * std if "max" in s else mean - kappa * std
        vals = pd.to_numeric(data[t], errors="coerce").dropna()
        if "max" in s:
            lo = float(s["max"])
            span = max(lo - float(vals.min()), 1e-9)
            depth += np.clip((lo - pess) / span, -1, 2)
            dist += np.clip((pess - lo) / span, 0, None)
        else:
            lo = float(s["min"])
            span = max(float(vals.max()) - lo, 1e-9)
            depth += np.clip((pess - lo) / span, -1, 2)
            dist += np.clip((lo - pess) / span, 0, None)
    depth /= len(spec)
    dist /= len(spec)
    feasible = dist <= 1e-12
    score = np.where(feasible, depth, -dist)      # 可行优先，其次深度/贴近度
    order = np.lexsort((dist, -score))
    take = order[:n_batch]
    return idx[take]


def _simulate_strategy(df, schema, spec, data, *, labeled_idx, pool_idx, guided,
                       n_batch, factors, seed, rng, progress_cb=None, tag=""):
    """单策略单种子重放。返回 {"steps": int|None, "curve": [(n, hv)]}。"""
    labeled = set(labeled_idx)
    pool = set(pool_idx)
    steps = None
    curve = []
    n_total = len(df)

    def _hv():
        pts = df.loc[list(labeled)]
        pts = pts[[bool(_satisfies(r, spec)) for _, r in pts.iterrows()]]
        return _norm_hypervolume(pts if len(pts) else None, spec, data, rng)

    curve.append((len(labeled), _hv()))
    if any(_satisfies(df.loc[i], spec) for i in labeled):
        return {"steps": len(labeled), "curve": curve}

    rounds = 0
    max_rounds = int(np.ceil((n_total - len(labeled)) / max(n_batch, 1))) + 2
    while pool and rounds < max_rounds:
        rounds += 1
        if guided:
            sub = df.loc[list(labeled)]
            b = train_chain(sub, schema, model_path=None, seed=seed)
            pool_df = df.loc[sorted(pool)]
            _s, _d, perf, unc = predict_chain(pool_df, b, return_uncertainty=True)
            # σ 乘 L1 校准因子
            for c in list(unc.columns):
                t = c[:-4] if c.endswith("_std") else None
                if t and t in factors and factors[t] > 0:
                    unc[c] = unc[c] * factors[t]
            chosen = _guided_pick(pool_df, perf, unc, spec, data, n_batch,
                                 kappa=KAPPA)
            if chosen is None:
                # 模型未就绪（初始样本下 Stage 3 未达建模门槛）——该轮回退随机
                chosen = pd.Index(rng.choice(sorted(pool), size=min(n_batch, len(pool)),
                                             replace=False))
        else:
            chosen = rng.choice(sorted(pool), size=min(n_batch, len(pool)), replace=False)
            chosen = pd.Index(chosen) if not isinstance(chosen, pd.Index) else chosen
        labeled |= set(chosen)
        pool -= set(chosen)
        curve.append((len(labeled), _hv()))
        if steps is None and any(_satisfies(df.loc[i], spec) for i in chosen):
            steps = len(labeled)
            break
        if progress_cb:
            progress_cb(f"{tag} 第 {rounds} 轮（已标注 {len(labeled)}）", None)
    # 循环耗尽仍未在「新加入点」中检出的兜底复查（初始已检过，这里覆盖整集变化）
    if steps is None and any(_satisfies(df.loc[i], spec) for i in labeled):
        # 曲线里找首个含达标点的累计实验数
        steps = len(labeled)
    return {"steps": steps, "curve": curve}


def run_benchmark(df, schema, *, spec=None, n_init=15, n_batch=5, n_seeds=5,
                   seed=SEED, progress_cb=None):
    """执行闭环效率重放竞赛。

    返回 dict（含曲线、节省次数 N 与解释），并落盘结果 json、对比曲线图与报告。
    """
    if df is None or len(df) < n_init + n_batch * 2:
        raise ValueError(f"数据量不足（需至少 {n_init + n_batch * 2} 行，当前 {len(df) if df is not None else 0}）。")
    spec = spec or DEFAULT_SPEC
    spec_targets = list(spec.keys())
    missing = [t for t in spec_targets if t not in df.columns]
    if missing:
        raise ValueError(f"目标规格引用的列不在当前数据中：{missing}。请在界面调整目标规格。")

    # 重放子集：spec 目标真实标签齐全的行（真实标签只来自数据库）
    sub = df.copy()
    for t in spec_targets:
        sub[t] = pd.to_numeric(sub[t], errors="coerce")
    sub = sub.dropna(subset=spec_targets).reset_index(drop=False)
    if len(sub) < n_init + n_batch * 2:
        raise ValueError(f"三目标真实标签齐全的样本不足（{len(sub)} 行，需 ≥ {n_init + n_batch * 2}）。")
    data = sub  # 超体积盒以重放子集的数据范围为准

    def _cb(msg, frac):
        if progress_cb:
            try:
                progress_cb(msg, frac)
            except Exception:
                pass

    n_ok = int(sum(_satisfies(r, spec) for _, r in sub.iterrows()))
    if n_ok == 0:
        result = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                  "spec": {t: s for t, s in spec.items()}, "n_replay": int(len(sub)),
                  "n_target_points": 0, "feasible": False,
                  "n_init": int(n_init), "n_batch": int(n_batch), "n_seeds": int(n_seeds),
                  "guided_steps": [], "random_steps": [],
                  "guided_median_steps": None, "random_median_steps": None,
                  "saved_experiments": None, "n_ok_ratio": 0.0,
                  "explanation": ("当前数据集中没有任何满足目标规格的样本点，"
                                  "重放竞赛无法开展（策略双方都永远无法『找到』达标点）。"
                                  "建议：① 核对目标规格是否过严（可参考数据集前 20% 分位水平放宽）；"
                                  "② 若规格本身合理，说明当前工艺域尚无达标经验，"
                                  "这正是模型外推探索的价值场景，应先扩充边界区域实验。")}
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _write_report(result)
        return result

    factors = load_factors()
    g_steps, r_steps, g_curves, r_curves = [], [], [], []
    # 初始子集从「非达标点」中抽取（模拟尚未找到好工艺的起点；
    # 达标点留给补点阶段发现——「从未找到 → 找到」的公平竞赛）
    _ok_idx = [i for i in sub.index if _satisfies(sub.loc[i], spec)]
    _not_ok_idx = [i for i in sub.index if i not in set(_ok_idx)]
    if len(_not_ok_idx) < n_init:
        raise ValueError(f"非达标样本不足（{len(_not_ok_idx)} 行 < 初始 {n_init}），"
                         "无法构造不含达标点的初始子集；请下调初始样本数或放宽目标规格。")
    master = np.random.default_rng(seed)
    for s_i in range(int(n_seeds)):
        rng = np.random.default_rng(seed + 1000 * (s_i + 1))
        init_pick = rng.choice(_not_ok_idx, size=n_init, replace=False)
        rest = [i for i in sub.index if i not in set(init_pick)]
        labeled_idx = pd.Index(init_pick)
        pool_idx = pd.Index(rest)
        _cb(f"种子 {s_i+1}/{n_seeds}：模型引导策略重放", (s_i) / n_seeds)
        g = _simulate_strategy(sub, schema, spec, data, labeled_idx=list(labeled_idx),
                               pool_idx=list(pool_idx), guided=True, n_batch=n_batch,
                               factors=factors, seed=seed, rng=rng, tag=f"种子{s_i+1}·引导")
        _cb(f"种子 {s_i+1}/{n_seeds}：随机基线策略重放", (s_i + 0.5) / n_seeds)
        rr = _simulate_strategy(sub, schema, spec, data, labeled_idx=list(labeled_idx),
                                pool_idx=list(pool_idx), guided=False, n_batch=n_batch,
                                factors=factors, seed=seed, rng=rng, tag=f"种子{s_i+1}·随机")
        g_curves.append(g["curve"]); g_steps.append(g["steps"])
        r_curves.append(rr["curve"]); r_steps.append(rr["steps"])

    g_arr = np.array([x if x is not None else np.nan for x in g_steps], dtype=float)
    r_arr = np.array([x if x is not None else np.nan for x in r_steps], dtype=float)
    g_med = float(np.nanmedian(g_arr)) if np.any(~np.isnan(g_arr)) else None
    r_med = float(np.nanmedian(r_arr)) if np.any(~np.isnan(r_arr)) else None
    saved = (r_med - g_med) if (g_med is not None and r_med is not None) else None
    feasible_all = not (np.any(np.isnan(g_arr)) or np.any(np.isnan(r_arr)))

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "spec": {t: s for t, s in spec.items()},
        "n_replay": int(len(sub)), "n_target_points": n_ok,
        "n_init": int(n_init), "n_batch": int(n_batch), "n_seeds": int(n_seeds),
        "seed": int(seed), "kappa": KAPPA,
        "calibration_factors_applied": bool(factors),
        "guided_steps": [None if np.isnan(x) else int(x) for x in g_arr],
        "random_steps": [None if np.isnan(x) else int(x) for x in r_arr],
        "guided_median_steps": g_med, "random_median_steps": r_med,
        "saved_experiments": saved, "feasible": True,
        "guided_curves": g_curves, "random_curves": r_curves,
        "n_ok_ratio": round(n_ok / len(sub), 4),
    }
    explanation = []
    if saved is not None and saved > 0:
        explanation.append(
            f"达到目标规格（{' 且 '.join(_spec_text(spec))}），模型引导策略平均需要 "
            f"{g_med:.0f} 次实验（中位数，{n_seeds} 个种子），随机基线需要 {r_med:.0f} 次——"
            f"模型引导平均节省 {saved:.0f} 次实验。")
    elif saved is not None:
        explanation.append(
            f"本次重放中模型引导未跑赢随机基线（引导 {g_med:.0f} 次 vs 随机 {r_med:.0f} 次）。"
            "排查：① 初始样本是否太少（建议 ≥15 且覆盖主要参数区域）；"
            "② 目标规格是否过松（当前数据集达标点占比 "
            f"{n_ok/len(sub)*100:.0f}%，若 >40% 规格偏松，建议收紧到前 20% 水平重跑）；"
            "③ 悲观惩罚是否过强（κ=1.0，可试 1.5 加强置信）。")
    if not feasible_all:
        explanation.append(
            "部分种子在池耗尽前未找到达标点（初始子集或数据规模限制），中位数统计已忽略这些种子。")
    result["explanation"] = " ".join(explanation)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    _plot_curves(g_curves, r_curves, spec, result)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _write_report(result)
    _cb("效率基准完成", 1.0)
    return result


def _spec_text(spec):
    parts = []
    for t, s in spec.items():
        if "max" in s:
            parts.append(f"{t} ≤ {s['max']:g}")
        else:
            parts.append(f"{t} ≥ {s['min']:g}")
    return parts


def _curve_stats(curves):
    """把各种子变长曲线对齐到统一横轴（累计实验数网格）后求均值±标准差。"""
    max_n = max(n for c in curves for n, _ in c)
    xs = sorted({n for c in curves for n, _ in c})
    grid = np.arange(xs[0], xs[-1] + 1)
    Y = np.full((len(curves), len(grid)), np.nan)
    for i, c in enumerate(curves):
        arr = np.asarray(c, dtype=float)
        base = arr[0, 0]
        idx_map = {int(n): v for n, v in arr}
        last = 0.0
        for j, x in enumerate(grid):
            if int(x) in idx_map:
                last = idx_map[int(x)]
            Y[i, j] = last
    return grid, np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)


def _plot_curves(g_curves, r_curves, spec, result):
    """落盘 matplotlib 论文级对比曲线（89mm 单栏，统一色板）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plot_style as ps
    from . import paper_style as psty

    psty.apply_paper_style()
    fig, ax = plt.subplots(figsize=psty.fig_single(2.4))
    for curves, color, label in ((r_curves, ps.GRAY_MUTED, "随机基线"),
                                 (g_curves, ps.BLUE_SIGNAL, "模型引导")):
        grid, mean, std = _curve_stats(curves)
        ax.plot(grid, mean, "-", lw=1.4, color=color, label=label)
        ax.fill_between(grid, np.clip(mean - std, 0, None), mean + std,
                        color=color, alpha=0.15, lw=0)
    ax.set_xlabel("累计实验次数")
    ax.set_ylabel("归一化超体积（MC 估计）")
    ax.legend(fontsize=7, frameon=False)
    sv = result.get("saved_experiments")
    psty.add_stat_box(ax, f"seeds={result['n_seeds']} ｜ 初始 n={result['n_init']}"
                          + (f" ｜ 节省 {sv:.0f} 次实验" if sv is not None else ""), loc="lower right")
    out = BENCH_DIR / "闭环效率对比.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_report(result):
    from .ui_labels import zh
    lines = ["# L3 · 闭环效率基准对比报告（核心价值主证据）", "",
             f"- 生成时间：{result['generated_at']}",
             f"- 目标规格：{' 且 '.join(_spec_text(result['spec']))}",
             f"- 重放样本：{result['n_replay']} 行（三目标真实标签齐全）；"
             f"其中满足规格 {result['n_target_points']} 行（{result.get('n_ok_ratio', 0)*100:.0f}%）",
             f"- 设置：初始 {result['n_init']} 个样本，每轮补 {result['n_batch']} 个，"
             f"{result['n_seeds']} 个随机种子，κ={result.get('kappa', 1.0)}，seed={result['seed']}",
             f"- 校准因子应用：{'是（L1）' if result.get('calibration_factors_applied') else '否（无校准文件，因子=1）'}",
             ""]
    if not result.get("feasible"):
        lines += ["## 结果", "", f"**无法开展竞赛**：{result['explanation']}"]
        (BENCH_DIR / "效率对比报告.md").write_text("\n".join(lines), encoding="utf-8")
        return
    g, r = result["guided_median_steps"], result["random_median_steps"]
    lines += ["## 核心数字", "",
              f"- 模型引导：达标所需实验数中位数 **{g:.0f}**（各种子：{result['guided_steps']}）",
              f"- 随机基线：达标所需实验数中位数 **{r:.0f}**（各种子：{result['random_steps']}）",
              (f"- **节省 N = {result['saved_experiments']:.0f} 次实验**"
               if result["saved_experiments"] is not None else "- 节省次数不可计算（存在未达成种子）"),
              "", "## 解释", "", result["explanation"], "",
              "![闭环效率对比](闭环效率对比.png)", "",
              "## 真实迭代收敛曲线", ""]
    # 真实迭代：历史模型版本样本量序列（各版本训练数据快照未存档，超体积无法回算）
    try:
        import glob as _g
        from .config import ROOT as _R
        mans = sorted(_g.glob(str(_R / "models" / "history" / "*" / "manifest.json")))
        if len(mans) >= 2:
            rows = []
            for p in mans:
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                rows.append((m.get("training_time", ""), m.get("sample_count")))
            lines += ["历史各模型版本的训练样本量：" + " → ".join(str(n) for _, n in rows),
                      "",
                      "> 注：历史版本未保存训练数据快照，无法回算各版本超体积；",
                      "> 真实闭环超体积曲线建议自本版本起经「L2 验证任务单」闭环逐步积累。"]
        else:
            lines += ["暂无证据（历史模型版本不足 2 个，无法绘制迭代曲线）。"]
    except Exception:
        lines += ["暂无证据（历史版本信息不可读）。"]
    lines += ["", "> 真实标签全部取自数据集（重放模拟铁律：严禁以模型预测值冒充真值）；",
              "> 超体积为蒙特卡洛估计（逐目标按规格盒归一化，跨量纲可比）。"]
    if result.get("n_target_points", 0) and result.get("n_replay", 1):
        pass
    (BENCH_DIR / "效率对比报告.md").write_text("\n".join(lines), encoding="utf-8")


def benchmark_status():
    try:
        with open(RESULT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {"done": True, "saved": data.get("saved_experiments"),
                "generated_at": data.get("generated_at"),
                "feasible": data.get("feasible")}
    except Exception:
        return {"done": False}
