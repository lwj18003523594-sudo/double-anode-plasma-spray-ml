# -*- coding: utf-8 -*-
"""V1.5 UI 主题测试（spec 七十二）

检查：中文优先宋体、英文/数字 Times New Roman、淡蓝+淡樱花粉主题、
无高饱和主色、Windows/macOS 字体回退、论文图未受网页 CSS 影响。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
CSS = re.search(r"<style>(.*?)</style>", APP, re.S).group(1)

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# 1. 字体栈：Times New Roman（英文/数字）在前，宋体系（中文）随后
m = re.search(r'font-family:\s*"Times New Roman",\s*"Songti SC",\s*"SimSun",\s*"STSong"', CSS)
check("1. 字体栈 = Times New Roman + 宋体系（Songti SC/SimSun/STSong）", bool(m))
check("1b. 不使用 PingFang / Microsoft YaHei / Arial 作为主要字体",
      "PingFang" not in CSS and "Microsoft YaHei" not in CSS and "Arial" not in CSS)
check("1c. 不下载/打包字体文件（无 @font-face / url( 字体）",
      "@font-face" not in CSS and not re.search(r"url\([^)]*\.(ttf|otf|woff)", CSS))
# Windows（SimSun）与 macOS（Songti SC）回退都存在
check("1d. Windows 回退 SimSun 与 macOS 回退 Songti SC 同时在栈中",
      "SimSun" in CSS and "Songti SC" in CSS and "STSong" in CSS)

# 2. 淡蓝 + 淡樱花粉主题（调色板在 CSS 规则与内联样式中使用）
for name, hexes in [("页面背景 #FAFBFC", ["#FAFBFC"]),
                    ("极淡蓝 #EAF3FB/#F2F7FC", ["#EAF3FB", "#F2F7FC"]),
                    ("强调蓝 #A9C9E5", ["#A9C9E5"]),
                    ("蓝色文字 #5C86AC", ["#5C86AC"]),
                    ("极淡樱花粉 #FBEDEF/#FFF5F6", ["#FBEDEF", "#FFF5F6"]),
                    ("强调粉 #E8B8C0", ["#E8B8C0"]),
                    ("粉色文字 #B77C87", ["#B77C87"]),
                    ("主文字 #2F3440 / 次级 #7A8290 / 边框 #E5E9EF",
                     ["#2F3440", "#7A8290", "#E5E9EF"])]:
    check(f"2. {name}", all(h in APP for h in hexes))

# 3. 无高饱和主色（亮红/亮蓝/霓虹/彩虹渐变）
high_sat = ["#FF0000", "#ff6b6b", "#00FF00", "#0000FF", "#FF1493", "#00FFFF",
            "neon", "rainbow", "linear-gradient(rainbow"]
check("3. 无高饱和主色/霓虹/彩虹渐变", not any(h.lower() in CSS.lower() for h in high_sat))

# 4. 按钮规范：普通白底浅灰边框 + hover 极淡蓝；primary 极淡蓝
flat = CSS.replace("\n", " ")
check("4. 普通按钮白底浅灰边框（#FFFFFF / #E5E9EF）",
      "background:#FFFFFF; color:#2F3440; border:1px solid #E5E9EF" in flat)
check("4b. 按钮 hover 极淡蓝 #EAF3FB", "button:hover" in flat and "#EAF3FB" in flat)
check("4c. primary 按钮极淡蓝（#EAF3FB 背景）",
      'stBaseButton-primary"]' in flat and "background:#EAF3FB !important" in flat)

# 5. 研究主线卡片：颜色只用于顶部色带（淡蓝/淡粉交替），卡片主体接近白色
check("5. 卡片主体白色 + 3px 顶部色带（淡蓝 #A9C9E5 / 淡粉 #E8B8C0 交替）",
      "background:#FFFFFF; border:1px solid #E5E9EF;" in flat
      and "border-top:3px solid #A9C9E5" in flat and "border-top-color:#E8B8C0" in flat)

# 6. 论文图字体独立：paper_style.py 不受网页 CSS 影响
PS = (ROOT / "src" / "paper_style.py").read_text(encoding="utf-8")
check("6. paper_style.py 独立管理论文图字体（含 SimSun/STSong/serif，未被 app CSS 覆盖）",
      ("SimSun" in PS or "STSong" in PS or "serif" in PS)
      and "stIconMaterial" not in PS and "<style>" not in PS)
check("6b. 网页字体设置不进入 paper_output（无 plotly 模板注入）",
      "pio.templates" not in PS)

# 7. plotly 网页图表统一字体（英文 Times + 中文宋体），且不影响 matplotlib
check("7. app.py 设置 plotly 网页字体（Times New Roman + Songti SC）",
      'Times New Roman, "Songti SC", SimSun' in APP)

print()
if failures:
    print(f"UI THEME TEST FAILED — {len(failures)} 项: {failures}")
    sys.exit(1)
print("UI THEME TEST PASSED")
