# -*- coding: utf-8 -*-
"""V1.3.1 科研细节与稳健性工具集

- dataset_hash：数据指纹（模型过期检测）
- training_domain：训练数据覆盖范围统计（min/max/median/Q1/Q3）
- classify_support：输入值相对训练域的范围分级（覆盖良好/接近边界/外推）
- fmt_value / fmt_metric：统一数字显示格式（仅显示层，不改底层数据）
- friendly_error：用户可读的中文错误提示（详细 traceback 写日志）
- log_event：outputs/logs/ 每日日志

全部 pathlib，macOS / Windows 通用；不修改任何模型逻辑。
"""
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

LOG_DIR = ROOT / "outputs" / "logs"

SUPPORT_GOOD = "数据覆盖良好"
SUPPORT_EDGE = "接近数据边界"
SUPPORT_EXTRAP = "存在外推风险"


# ---------------- 数据指纹与过期检测 ----------------
def dataset_hash(df) -> str:
    """与 train_chain_full 相同算法的数据指纹（sha256 前 16 位）。"""
    return hashlib_sha256(df.to_csv(index=True).encode("utf-8"))[:16]


def hashlib_sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def model_staleness(bundle, df):
    """返回 (is_stale, model_hash, current_hash)。
    bundle 无 dataset_hash（旧模型）时返回 (None, None, current)。"""
    current = dataset_hash(df)
    model_hash = bundle.get("dataset_hash")
    if model_hash is None:
        return None, None, current
    # bundle 内保存完整 64 位哈希，统一截取前 16 位比较
    return str(model_hash)[:16] != current, str(model_hash)[:16], current


# ---------------- 训练数据覆盖范围 ----------------
def training_domain(df, input_cols):
    """统计每个输入特征的 min/max/median/Q1/Q3（来源于训练数据，非设备极限）。"""
    domain = {}
    for c in input_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        domain[c] = {
            "min": float(s.min()), "max": float(s.max()),
            "median": float(s.median()),
            "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
            "n": int(len(s)),
        }
    return domain


def save_domain_files(domain, model_path):
    """训练后保存 training_domain.json（部署目录 + 历史版本目录）。"""
    model_path = Path(model_path)
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path.parent / "training_domain.json", "w", encoding="utf-8") as f:
            json.dump(domain, f, ensure_ascii=False, indent=2)
        hist = model_path.parent / "history"
        if hist.exists():
            for ver_dir in hist.iterdir():
                if ver_dir.is_dir():
                    with open(ver_dir / "training_domain.json", "w", encoding="utf-8") as f:
                        json.dump(domain, f, ensure_ascii=False, indent=2)
    except Exception:
        log_event("domain_save_error", traceback.format_exc())


def classify_support(value, dom):
    """相对训练域的分级：good(≤Q1~Q3 内) / edge(min~max 内) / extrap(超出)。
    dom 为 None（旧模型）时返回 'unknown'。"""
    if dom is None:
        return "unknown"
    v = float(value)
    if v < dom["min"] or v > dom["max"]:
        return "extrap"
    if v < dom["q1"] or v > dom["q3"]:
        return "edge"
    return "good"


SUPPORT_COLOR = {"good": "#2ecc71", "edge": "#f1c40f", "extrap": "#e74c3c", "unknown": "#95a5a6"}
SUPPORT_LABEL_EN = {"good": "Within data coverage", "edge": "Near data boundary",
                    "extrap": "Extrapolation risk", "unknown": "Unknown (legacy model)"}


def support_label(status, lang="zh"):
    if lang == "en":
        return SUPPORT_LABEL_EN.get(status, status)
    return {"good": SUPPORT_GOOD, "edge": SUPPORT_EDGE, "extrap": SUPPORT_EXTRAP,
            "unknown": "未知（旧模型）"}.get(status, status)


def pareto_row_support(row_params, domain):
    """一组工艺参数的综合数据支持状态：任意外推 > 任一接近边界 > 覆盖良好。"""
    if not domain:
        return "unknown"
    statuses = []
    for col, dom in domain.items():
        if col in row_params and row_params[col] is not None and not pd.isna(row_params[col]):
            statuses.append(classify_support(row_params[col], dom))
    if not statuses:
        return "unknown"
    if "extrap" in statuses:
        return "extrap"
    if "edge" in statuses:
        return "edge"
    return "good"


# ---------------- 统一数字显示格式（仅显示层） ----------------
def _decimals_for(col):
    c = col.lower()
    if c.endswith("_a"):
        return 1
    if "flow" in c or c.endswith("slpm") or "g_min" in c or "feed" in c:
        return 1
    if c.endswith("_k") or c.endswith("_c"):
        return 1
    if "velocity" in c or "speed" in c or c.endswith("m_s"):
        return 1
    if c.endswith("pct"):
        return 2
    if c.endswith("mpa"):
        return 2
    if c.endswith("hv"):
        return 1
    if "ratio" in c or "index" in c:
        return 3
    return 2


def fmt_value(col, value):
    """按字段类别统一小数位；返回 float（显示用），不改动底层数据。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(v):
        return None
    return round(v, _decimals_for(col))


def fmt_metric(v, kind="R2"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if kind == "R2":
        return round(float(v), 4)
    a = abs(float(v))
    sig = 3 if a >= 100 else 4
    return float(f"{float(v):.{sig}g}")


# ---------------- 中文错误提示 ----------------
def friendly_error(exc, df=None, schema=None) -> str:
    """把常见异常映射为用户可读中文；原始 traceback 写入日志。"""
    try:
        log_event("error", "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass
    name = type(exc).__name__
    msg = str(exc)
    if isinstance(exc, KeyError):
        return f"当前数据缺少对应字段（{msg}），因此无法运行该模型。请检查数据表头是否与模板一致。"
    if isinstance(exc, ValueError) and ("NaN" in msg or "inf" in msg):
        missing_info = ""
        if df is not None and schema is not None:
            try:
                rows = []
                for sec in ["process_states", "defect_network", "performance_outputs"]:
                    for col in schema.get(sec, {}):
                        if col in df.columns:
                            s = pd.to_numeric(df[col], errors="coerce")
                            if s.isna().any():
                                rows.append(f"{col}: 缺失 {int(s.isna().sum())} 条 / "
                                            f"实际可用 {int(s.notna().sum())} 条")
                missing_info = "；".join(rows[:6])
            except Exception:
                pass
        base = "数据中存在缺失值，无法完成该计算。"
        if missing_info:
            base += f" 缺失情况：{missing_info}。未测指标可先留空，有效样本不足 20 条的目标不会参与训练。"
        return base
    if isinstance(exc, ZeroDivisionError):
        return "计算过程中出现除零（可能某列全为常数或全缺失），请检查数据。"
    return f"操作未完成（{name}）。详情已写入日志；请检查数据格式后重试。"


# ---------------- 日志 ----------------
def log_event(event, detail=""):
    """按天写日志到 outputs/logs/，不在界面显示。失败静默。"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {event}" + (f" | {detail}" if detail else "")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass


# ---------------- 重复实验统计 ----------------
def replicate_stats(df, group_col="sample_id", min_dup=2):
    """重复测量统计：n / mean / std / CV%。
    优先 sample_id（+replicate_id），否则回退 experiment_id。
    返回 DataFrame 或 None（无重复）。不修改原始数据。"""
    if group_col not in df.columns:
        return None
    gcol = [group_col]
    if group_col == "sample_id" and "replicate_id" in df.columns:
        gcol.append("replicate_id")
        # 按 试样+重复 编号分组时，每组通常只有 1 条；重复测量应为同 sample_id 多行
        gcol = [group_col]
    num = df.select_dtypes("number")
    if "batch_id" in num.columns:
        num = num.drop(columns=["batch_id"])
    grouped = num.groupby(df[gcol[0]])
    rows = []
    for key, g in grouped:
        if len(g) < min_dup:
            continue
        for c in g.columns:
            s = g[c].dropna()
            if len(s) < min_dup:
                continue
            mean = float(s.mean())
            std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            cv = (std / abs(mean) * 100) if mean else np.nan
            rows.append({group_col: key, "字段": c, "n": int(len(s)),
                         "mean": mean, "std": std,
                         "CV%": round(cv, 2) if pd.notna(cv) else None})
    if not rows:
        return None
    return pd.DataFrame(rows)
