# -*- coding: utf-8 -*-
"""UI 主题测试（V1.5 spec 七十二，V1.6 T03 起随 CSS 变量化架构更新）

检查：中文优先宋体、英文/数字 Times New Roman、淡蓝+淡樱花粉主题、
无高饱和主色、Windows/macOS 字体回退、论文图未受网页 CSS 影响。
V1.6 T03 起：hex 色板唯一定义处为 src/plot_style.py；app.py 经 _CSS_VARS 注入
CSS 变量、主体一律 var(--*) 引用。本测试对两处分别断言。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
PLOT_STYLE = (ROOT / "src" / "plot_style.py").read_text(encoding="utf-8")


def _extract_css(src: str) -> str:
    """提取页面 CSS：V1.6 T03 起 CSS 主体在 _CSS_VARS/_CSS_BODY 字符串常量中；
    兼容旧版内联 <style>...</style> 块。"""
    parts = []
    for var in ("_CSS_VARS", "_CSS_BODY"):
        m = re.search(var + r'\s*=\s*f?"""(.*?)"""', src, re.S)
        if m:
            parts.append(m.group(1))
    if parts:
        return "\n".join(parts)
    m = re.search(r"<style>(.*?)</style>", src, re.S)
    return m.group(1) if m else ""


CSS = _extract_css(APP)
failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# 1. 字体栈：Times New Roman（英文/数字）在前，宋体系（中文）随后
m = re.search(r'font-family:\s*"Times New Roman",\s*"Songti SC",\s*"SimSun",\s*STSong', CSS)
check("1. 字体栈 = Times New Roman + 宋体系（Songti SC/SimSun/STSong）", bool(m))
check("1b. 不使用 PingFang / Microsoft YaHei / Arial 作为主要字体",
      "PingFang" not in CSS and "Microsoft YaHei" not in CSS and "Arial" not in CSS
      and "Arial" not in PLOT_STYLE)
check("1c. 不下载/打包字体文件（无 @font-face / url( 字体）",
      "@font-face" not in CSS and not re.search(r"url\([^)]*\.(ttf|otf|woff)", CSS))
# Windows（SimSun）与 macOS（Songti SC）回退都存在
check("1d. Windows 回退 SimSun 与 macOS 回退 Songti SC 同时在栈中",
      "SimSun" in CSS and "Songti SC" in CSS and "STSong" in CSS)

# 2. 淡蓝 + 淡樱花粉主题：hex 唯一定义处 src/plot_style.py（V1.6 T03），
#    app.py CSS 主体经 var(--*) 引用同源变量
palette = {
    "页面背景 #FAFBFC（--gray-paper）": ("#FAFBFC", "var(--gray-paper)"),
    "极淡蓝 #EAF3FB（--blue-faint）": ("#EAF3FB", "var(--blue-faint)"),
    "强调蓝 #A9C9E5（--blue-light）": ("#A9C9E5", "var(--blue-light)"),
    "信号蓝 #3C6E9F（--blue-signal，V1.6 主色）": ("#3C6E9F", "var(--blue-signal)"),
    "强调粉 #E8B8C0（--pink-light）": ("#E8B8C0", "var(--pink-light)"),
    "主粉 #C4707F（--pink-accent，V1.6 P0-1）": ("#C4707F", "var(--pink-accent)"),
    "主文字 #2F3440（--gray-ink）": ("#2F3440", "var(--gray-ink)"),
    "次级文字 #7A8290（--gray-muted）": ("#7A8290", "var(--gray-muted)"),
    "边框 #E5E9EF（--gray-border）": ("#E5E9EF", "var(--gray-border)"),
}
for name, (hexv, varv) in palette.items():
    check(f"2. {name}", hexv in PLOT_STYLE and (varv in CSS or varv in APP))

# 3. 无高饱和主色（亮红/亮蓝/霓虹/彩虹渐变）
high_sat = ["#FF0000", "#ff6b6b", "#00FF00", "#0000FF", "#FF1493", "#00FFFF",
            "neon", "rainbow", "linear-gradient(rainbow"]
check("3. 无高饱和主色/霓虹/彩虹渐变", not any(h.lower() in CSS.lower() for h in high_sat))

# 4. 按钮规范（V1.6 P0-1）：普通白底浅灰边框 + hover 极淡蓝；primary 唯一实底强调（粉）
flat = CSS.replace("\n", " ")
check("4. 普通按钮白底浅灰边框（#FFFFFF / --gray-border）",
      "background:#FFFFFF; color:var(--gray-ink); border:1px solid var(--gray-border);" in flat)
check("4b. 按钮 hover 极淡蓝（--blue-faint）",
      "button:hover" in flat and "background:var(--blue-faint)" in flat)
check("4c. primary 按钮唯一实底强调（--pink-accent 实底白字，V1.6 P0-1）",
      'stBaseButton-primary"]' in flat and "background:var(--pink-accent) !important" in flat
      and "color:#FFFFFF !important" in flat)

# 5. 研究主线卡片：颜色只用于顶部色带（淡蓝/淡粉交替），卡片主体接近白色
check("5. 卡片主体白色 + 3px 顶部色带（淡蓝 --blue-light / 淡粉 --pink-light 交替）",
      "background:#FFFFFF; border:1px solid var(--gray-border);" in flat
      and "border-top:3px solid var(--blue-light)" in flat
      and "border-top-color:var(--pink-light)" in flat)

# 6. 论文图字体独立：paper_style.py 不受网页 CSS 影响
PS = (ROOT / "src" / "paper_style.py").read_text(encoding="utf-8")
check("6. paper_style.py 独立管理论文图字体（含 SimSun/STSong/serif，未被 app CSS 覆盖）",
      ("SimSun" in PS or "STSong" in PS or "serif" in PS)
      and "stIconMaterial" not in PS and "<style>" not in PS)
check("6b. 网页字体设置不进入 paper_output（无 plotly 模板注入）",
      "pio.templates" not in PS)

# 7. plotly 网页图表统一字体（英文 Times + 中文宋体），唯一定义处 plot_style.py WEB_FONT
check("7. plot_style.py 定义 plotly 网页字体（Times New Roman + Songti SC）",
      'Times New Roman, "Songti SC", SimSun' in PLOT_STYLE
      and "tpl.layout.font.family = WEB_FONT" in PLOT_STYLE)

print()
if failures:
    print(f"UI THEME TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("UI THEME TEST PASSED")
