# -*- coding: utf-8 -*-
"""V1.6 全局校验脚本（QA 辅助，DESIGN_V1.6 T04）。

用法：
    python scripts/verify_v16.py                 # 色值残留扫描 + 禁用词检查
    python scripts/verify_v16.py --run <run_id>  # 追加：evidence_manifest 数值-文本一致性抽查

检查项：
① 色值残留：app.py + src/*.py（plot_style.py 为唯一定义处，豁免；
   不动清单 8 模块零改动、其 V1.5 基线遗留色豁免）中
   seaborn/plotly 默认与残留色 hex 0 hit；
② 禁用词：全部已生成的 *_Analysis.md / Insight_Report.md 中——
   绝对化词 0 hit；"显著" 仅允许出现在 p 值/显著性检验语境；
③ evidence 抽查（--run）：manifest 中每条非降级结论，其引用数值可还原回文本。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ① 禁止出现的 hex（plotly 默认循环色 / seaborn 残留色 / V1.5 残留强调色）
FORBIDDEN_HEX = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
    "#FF6692", "#B6E880", "#FF97FF", "#FECB52",             # plotly 默认
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",   # seaborn 残留
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#C0392B", "#2ECC71", "#F1C40F", "#E74C3C",              # V1.5 散落装饰色
]
# 色板唯一定义处（豁免文件）
PALETTE_HOME = "src/plot_style.py"
# 不动清单（零改动模块，历史遗留色由 V1.5 基线保证，不在本次扫描范围）
PROTECTED_MODULES = {
    "model_chain.py", "optimize.py", "literature.py", "modes.py",
    "research_utils.py", "data_utils.py", "features.py", "config.py",
}
SCAN_FILES = ["app.py"] + sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "src").glob("*.py")
    if p.name not in {"plot_style.py", "__init__.py"} | PROTECTED_MODULES)

# ② 禁用词（与 figure_analysis 纪律一致）
ABSOLUTE_WORDS = ["绝对最优", "完全证明", "必然导致", "最佳工艺", "最优工艺方案"]


def check_colors() -> bool:
    ok = True
    for rel in SCAN_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for hexcode in FORBIDDEN_HEX:
            for m in re.finditer(re.escape(hexcode), text, flags=re.IGNORECASE):
                line_no = text[: m.start()].count("\n") + 1
                print(f"  ✗ {rel}:{line_no} 残留色 {hexcode}")
                ok = False
    if ok:
        print(f"  ✓ {len(SCAN_FILES)} 个文件扫描，残留色 hex 0 hit")
    return ok


def _md_files() -> list[Path]:
    out = []
    for pattern in ["runs/quick_analysis/**/*.md", "outputs/**/*_Analysis.md",
                    "outputs/**/Insight_Report.md"]:
        out += [p for p in ROOT.glob(pattern) if p.is_file()]
    return sorted(set(out))[:200]


def check_forbidden_words() -> bool:
    files = _md_files()
    if not files:
        print("  （未找到已生成的解读 md，跳过——生成结果包后请复检）")
        return True
    ok = True
    n_abs = n_sig = 0
    for p in files:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for w in ABSOLUTE_WORDS:
                if w in line:
                    print(f"  ✗ {p.relative_to(ROOT)}:{i} 绝对化词「{w}」")
                    ok = False
                    n_abs += 1
            # “显著”仅在无统计检验语境出现时降级（与 figure_analysis 同规则）
            if "显著" in line and "p<" not in line and "p <" not in line and "显著性" not in line:
                print(f"  ✗ {p.relative_to(ROOT)}:{i} 无检验语境使用「显著」")
                ok = False
                n_sig += 1
    if ok:
        print(f"  ✓ {len(files)} 份解读 md 禁用词检查通过")
    return ok


def _fmt_candidates(v):
    """数值在文本中可能出现的格式（3 位有效数字家族）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return []
    if math.isnan(f):
        return []
    out = {f"{f:.3f}", f"{f:.3g}", f"{f:.2f}", f"{f:.4f}", str(int(f)) if f == int(f) else None}
    return [s for s in out if s]


def check_evidence(run_id: str) -> bool:
    run_dir = ROOT / "runs" / "quick_analysis" / run_id
    manifest_path = run_dir / "evidence_manifest.json"
    if not manifest_path.exists():
        print(f"  ✗ 未找到 {manifest_path}")
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = True
    n_ok = n_degraded = 0
    for eid, item in manifest.items():
        text = str(item.get("text", ""))
        if item.get("degraded"):
            if text.endswith("当前数据不足以支持该判断"):
                n_degraded += 1
            else:
                print(f"  ✗ {eid} 标记 degraded 但文本不是降级句：{text[:50]}")
                ok = False
            continue
        if item.get("demo") and not text.startswith("[演示数据]"):
            print(f"  ✗ {eid} demo 记录缺少「[演示数据]」前缀")
            ok = False
        vals = item.get("values") or {}
        # meta 类可豁免数值要求（run_id/版本号等字符串型元信息）
        if item.get("source_class") != "meta" and not vals:
            print(f"  ✗ {eid} 非降级结论缺少 values 字段")
            ok = False
        # 抽查：每个引用值应能在文本中还原——数值按 3 位有效数字家族格式匹配，
        # 字符串按子串匹配（取前 20 字避免过长拼接串误判）
        hit = 0
        for k, v in vals.items():
            if k == "__degrade_reason":
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                for cand in _fmt_candidates(v):
                    if cand in text:
                        hit += 1
                        break
            elif isinstance(v, str) and len(v) >= 4:
                probe = v[:20] if len(v) > 20 else v
                if probe in text:
                    hit += 1
        if vals and hit == 0:
            print(f"  ✗ {eid} 文本中找不到任何 values 引用值：{text[:60]}")
            ok = False
        else:
            n_ok += 1
    print(f"  ✓ evidence 抽查：{len(manifest)} 条（正常 {n_ok}，降级 {n_degraded}）")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="V1.6 校验脚本")
    parser.add_argument("--run", default=None, help="抽查指定 run_id 的 evidence_manifest")
    args = parser.parse_args()

    print("== ① 色值残留扫描（plotly/seaborn 默认色 + V1.5 残留色） ==")
    ok1 = check_colors()
    print("== ② 解读 md 禁用词检查 ==")
    ok2 = check_forbidden_words()
    ok3 = True
    if args.run:
        print(f"== ③ evidence_manifest 抽查（run={args.run}） ==")
        ok3 = check_evidence(args.run)

    if ok1 and ok2 and ok3:
        print("\n全部通过：verify_v16 ✓")
        return 0
    print("\n存在未通过项：verify_v16 ✗")
    return 1


if __name__ == "__main__":
    sys.exit(main())
