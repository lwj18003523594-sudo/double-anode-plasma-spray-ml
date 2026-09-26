# -*- coding: utf-8 -*-
"""V1.8 · 全局身份与状态绑定（P0：状态铁律的机制层）。

方案 §10 要求：每个结果自身携带上下文（model_version / dataset_version）；
切换数据或重训模型后，旧结果自动判为「历史结果」，不得冒充当前状态。

状态语义铁律（方案 §10 原文）：
    未知不是 0，未运行不是失败，过期不是通过。

三类结果身份（freshness）：
    fresh   —— 结果的绑定版本与当前模型/数据一致（可显示为当前状态）
    stale   —— 绑定版本存在但与当前不匹配（显示为「历史结果（针对 模型X/数据Y）」）
    unknown —— 结果文件缺绑定信息（V1.7 及更早的旧文件；显示「旧版结果，待重跑核验」）

用法：
    from src.status_binding import bind_identity, freshness
    meta = bind_identity(bundle_or_none, dataset_hash_or_none)   # 写入结果 json
    state = freshness(saved_meta, current_model_version, current_dataset_hash)
"""
from datetime import datetime

UNKNOWN_TEXT = "未知"


def current_identity(bundle=None, dataset_hash=None):
    """从当前模型 bundle 与数据 hash 提取身份对。

    bundle 为 None（未加载模型）时 model_version 为 None——表示「当前无模型」，
    此时任何结果都不能称为 fresh。
    """
    mv = None
    dh = None
    if isinstance(bundle, dict):
        mv = bundle.get("model_version")
        dh = bundle.get("dataset_hash")
    if dataset_hash:
        dh = dataset_hash
    return {"model_version": mv, "dataset_hash": dh,
            "bound_at": datetime.now().isoformat(timespec="seconds")}


def bind_identity(result_dict, bundle=None, dataset_hash=None, **extra):
    """把身份写入结果 dict（原地修改并返回）。

    extra 记录附加上下文（如 schema_version / rule_version / spec）。
    结果落盘后即携带完整绑定信息。
    """
    ident = current_identity(bundle, dataset_hash)
    ctx = {"model_version": ident["model_version"],
           "dataset_hash": ident["dataset_hash"],
           "bound_at": ident["bound_at"]}
    ctx.update(extra)
    result_dict["binding"] = ctx
    return result_dict


def freshness(saved, current_model_version=None, current_dataset_hash=None):
    """判定已保存结果的时效性。

    saved: 结果 json（dict）或其中提取出的 binding 字段。
    返回 {"state": "fresh"|"stale"|"unknown", "note": 展示用说明}
    """
    b = saved.get("binding") if isinstance(saved, dict) else None
    if not isinstance(b, dict):
        return {"state": "unknown",
                "note": "旧版结果（无版本绑定），建议在当前数据/模型下重跑核验。"}
    mv, dh = b.get("model_version"), b.get("dataset_hash")
    parts, mismatch = [], False
    if current_model_version is not None:
        if mv is None:
            mismatch = True
            parts.append(f"结果未记录模型版本（当前 {current_model_version}）")
        elif mv != current_model_version:
            mismatch = True
            parts.append(f"针对模型 {mv}（当前 {current_model_version}）")
    if current_dataset_hash:
        cur8 = str(current_dataset_hash)[:8]
        if dh is None:
            mismatch = True
            parts.append(f"结果未记录数据版本（当前 auto_{cur8}）")
        elif str(dh)[:8] != cur8:
            mismatch = True
            parts.append(f"针对数据 auto_{str(dh)[:8]}（当前 auto_{cur8}）")
    if mismatch:
        return {"state": "stale",
                "note": "历史结果（" + "；".join(parts) + "），不代表当前数据/模型状态。"}
    return {"state": "fresh", "note": ""}


def fmt_n(value, label="n"):
    """样本量展示：None/不可得 → 「未知」，绝不填 0 冒充统计（方案 §2 诊断③）。"""
    if value is None:
        return f"{label}=未知"
    try:
        v = int(value)
        return f"{label}={v}" if v > 0 else f"{label}=未知"
    except (TypeError, ValueError):
        return f"{label}=未知"
