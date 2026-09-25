# -*- coding: utf-8 -*-
"""V1.5 CurrentContext：全局唯一研究状态对象（纯函数，便于测试）。

所有页面统一从 build_context() 构建的单一状态源读取：
研究模式 / 数据来源 / 数据集版本 / 记录数 / 批次 / 模型状态 / 验证方式 / Stage 1.5。

铁律（spec 六）：
- 切换研究模式或数据来源后，若当前模式没有加载数据：
  dataset_loaded=False、model_trained=False → 界面显示「未加载 / 等待数据」；
- 禁止任何跨模式状态串用（左侧"文献数据"、结果页"真实数据"之类的不一致）。
"""
from datetime import datetime


def build_context(research_mode, data_source, df=None, bundle=None, model_file_exists=False,
                  active_run_id=None, source_kind=None):
    """构建 CurrentContext 字典。

    df=None → dataset_loaded=False；bundle=None 且无模型文件 → model_trained=False。
    hash/完整版本号保留在 context 内（供【详细信息】与 metadata 使用），普通界面只显示短标签。
    """
    from .modes import MODES

    mode_cfg = MODES.get(research_mode) or {}
    dataset_hash = None
    dataset_version = None
    record_count = batch_count = sample_count = None
    if df is not None:
        try:
            import pandas as _pd
            record_count = int(len(df))
            if "batch_id" in df.columns:
                batch_count = int(df["batch_id"].nunique())
            if "sample_id" in df.columns:
                s = df["sample_id"].dropna()
                if len(s):
                    sample_count = int(s.nunique())
            from .research_utils import dataset_hash as _dh
            dataset_hash = _dh(df)
            dataset_version = "auto_" + dataset_hash[:8]
        except Exception:
            pass

    model_version = model_id = None
    validation_method = None
    stage15_enabled = False
    model_trained = False
    if bundle is not None:
        model_trained = True
        model_version = bundle.get("model_version")
        model_id = bundle.get("model_version")
        cv = bundle.get("chain_cv") or {}
        validation_method = cv.get("method") if cv and cv.get("method") != "none" else None
        stage15_enabled = bool(bundle.get("stage15_enabled"))
    elif model_file_exists:
        model_trained = True  # 模型文件存在（具体版本在加载后补全）

    return {
        "research_mode": research_mode,
        "research_mode_label": mode_cfg.get("label", research_mode),
        "data_source": data_source,
        "data_source_kind": source_kind,
        "dataset_id": dataset_version,
        "dataset_name": None,          # 由上传层填写文件名
        "dataset_hash": dataset_hash,  # 完整 hash 仅进【详细信息】/metadata
        "dataset_version": dataset_version,
        "dataset_loaded": df is not None,
        "record_count": record_count,
        "batch_count": batch_count,
        "sample_count": sample_count,
        "model_id": model_id,
        "model_version": model_version,
        "model_trained": model_trained,
        "validation_method": validation_method,
        "stage15_enabled": stage15_enabled,
        "active_run_id": active_run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def context_status_items(ctx):
    """普通状态栏的 7 项显示（spec 六十：不显示长 hash/完整版本号）。"""
    mode_label = ctx.get("research_mode_label", "-")
    source = ctx.get("data_source", "-")
    if not ctx.get("dataset_loaded"):
        return [
            f"<b>研究模式</b>：{mode_label}",
            f"<b>数据来源</b>：{source}",
            "<b>数据状态</b>：未加载",
            "<b>数据记录数</b>：—",
            "<b>独立批次</b>：—",
            "<b>模型状态</b>：等待数据",
            "<b>验证方式</b>：—",
        ]
    validation = ctx.get("validation_method")
    validation_disp = ("GroupKFold" if "GroupKFold" in str(validation)
                       else ("LOOCV" if "LOOCV" in str(validation)
                             else ("KFold" if "KFold" in str(validation) else "—")))
    if ctx.get("model_trained"):
        model_status = "已训练"
    else:
        model_status = "未训练"
    return [
        f"<b>研究模式</b>：{mode_label}",
        f"<b>数据来源</b>：{source}",
        "<b>数据状态</b>：已加载",
        f"<b>数据记录数</b>：{ctx.get('record_count')}",
        f"<b>独立批次</b>：{ctx.get('batch_count') if ctx.get('batch_count') is not None else '—'}",
        f"<b>模型状态</b>：{model_status}",
        f"<b>验证方式</b>：{validation_disp}",
    ]


def context_detail_lines(ctx):
    """【详细信息】内容：完整 hash / 模型版本 / 时间戳（不进普通界面）。"""
    return [
        f"研究模式：{ctx.get('research_mode_label')}（{ctx.get('research_mode')}）",
        f"数据来源：{ctx.get('data_source')}",
        f"数据集版本：{ctx.get('dataset_version') or '—'}",
        f"dataset_hash：{ctx.get('dataset_hash') or '—'}",
        f"模型版本：{ctx.get('model_version') or '—'}",
        f"Stage 1.5：{'已启用' if ctx.get('stage15_enabled') else '未启用'}",
        f"验证方式：{ctx.get('validation_method') or '—'}",
        f"active_run_id：{ctx.get('active_run_id') or '—'}",
        f"状态时间：{ctx.get('timestamp')}",
    ]


# ---------------- V1.6：状态徽标辅助（P0-7 / P1-2，与 ③⑧ 同源只读） ----------------
def context_cv_short(ctx) -> str:
    """CV 方式短标签：GroupKFold / LOOCV / KFold / —（界面显示专用）。"""
    validation = ctx.get("validation_method")
    s = str(validation or "")
    if "GroupKFold" in s:
        return "GroupKFold"
    if "LOOCV" in s:
        return "LOOCV"
    if "KFold" in s:
        return "KFold"
    return "—"


def context_pill_text(ctx) -> str:
    """Tab 顶部 status-pill 文本（全部读 build_context 产物，禁止二次计算）。

    示例：『✓ 数据已加载 · n=45 · 数据版本 auto_xxxx · ✓ 模型已训练 · CV: GroupKFold』
    首页 / ③ / ⑧ 使用同一 CTX，显示结果完全一致（P1-2 验收）。
    """
    parts = []
    if ctx.get("dataset_loaded"):
        n = ctx.get("record_count")
        parts.append("✓ 数据已加载" + (f" · n={n}" if n is not None else ""))
        if ctx.get("dataset_version"):
            parts.append(f"数据版本 {ctx['dataset_version']}")
    else:
        parts.append("数据未加载")
    if ctx.get("model_trained"):
        parts.append("✓ 模型已训练")
        parts.append(f"CV: {context_cv_short(ctx)}")
        if ctx.get("model_version"):
            parts.append(f"模型版本 {str(ctx['model_version']).split('_')[-1]}")
    else:
        parts.append("模型未训练")
    return " ｜ ".join(parts)
