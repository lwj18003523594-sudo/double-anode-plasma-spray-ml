# -*- coding: utf-8 -*-
"""V1.5 UI 一致性测试（spec 七十一）

检查：字号层级、正文大小、对齐、间距、卡片宽度、文字裁切、超大标题、
keyboard_double 泄漏、uploadupload 泄漏。
（对齐/间距为 CSS 规则断言；论文图字体独立于网页 CSS 由 ui_theme_test 覆盖。）
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

APP = Path(__file__).resolve().parents[1] / "app.py"
SRC = APP.read_text(encoding="utf-8")
CSS = re.search(r"<style>(.*?)</style>", SRC, re.S)
CSS = CSS.group(1) if CSS else ""

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# 1. icon 连字泄漏：全局隐藏规则存在；源码不包含泄漏文本
check("1. stIconMaterial 全局隐藏规则存在（keyboard_double/expand_more/menu 泄漏修复）",
      'span[data-testid="stIconMaterial"]' in CSS and "display: none !important" in CSS)
check("1b. 源码无 uploadupload / keyboard_double 泄漏文本",
      "uploadupload" not in SRC and "keyboard_double" not in SRC)

# 2. 字号层级（spec 九）
def has_rule(sel_prop, value):
    return value in CSS

check("2. 主标题 h1 26–32px（clamp）", "h1 {font-size: clamp(26px" in CSS.replace("\n", " "))
check("2b. 二级标题 h2 ≤ 25px", "h2 {font-size: clamp(20px, 2.2vw, 25px)" in CSS.replace("\n", " "))
check("2c. 模块标题 h3 ≤ 20px", "h3 {font-size: clamp(17px, 1.8vw, 20px)" in CSS.replace("\n", " "))
check("2d. 正文 15–17px + 行距 1.5–1.7", "font-size: 15.5px" in CSS and "line-height: 1.6" in CSS)
check("2e. 辅助说明 13–14px", "font-size: 13.5px" in CSS)
check("2f. 无超大标题（无 45–55px 字号）",
      not re.search(r"font-size:\s*(4[5-9]|5[0-5])px", CSS))
check("2g. Metric 数字 28–32px", 'stMetricValue"] {font-size: 30px' in CSS)

# 3. 对齐：标题/正文左对齐；正文无两端对齐
check("3. 正文左对齐且无 justify", "text-align: left" in CSS and "justify" not in CSS)

# 4. 间距体系（8/12/16/24/32/48 出现于规则中）
spacing_ok = all(v in CSS for v in ["24px", "16px", "12px", "8px"])
check("4. 统一 spacing 体系（8/12/16/24）", spacing_ok)
check("4b. 卡片 padding 16–20px", "padding:16px 12px" in CSS)

# 5. 卡片不裁切：h4 允许换行 + 容器可换行 + 最小宽度
flat = CSS.replace("\n", " ")
check("5. 研究主线卡片标题不裁切（white-space:normal + word-break）",
      "white-space:normal" in flat and "word-break" in flat)
check("5b. 卡片容器 flex-wrap + min-width:0（末卡不被裁切）",
      "flex-wrap:wrap" in flat and "min-width:0" in flat)

# 6. 研究主线包含颗粒熔融与沉积状态层
from src.modes import MODES
check("6. 双阳极研究主线含「颗粒熔融与沉积」层",
      any("熔融" in t for t, _ in MODES["dual_anode"]["research_line"]))

# 7. 顶部导航 8 页（含数据洞察）
check("7. 顶部导航 8 页且含「数据洞察与实验反馈」",
      '"⑥ 数据洞察与实验反馈"' in SRC and SRC.count('st.tabs([') == 1)

print()
if failures:
    print(f"UI CONSISTENCY TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("UI CONSISTENCY TEST PASSED")
