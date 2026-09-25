# 双阳极等离子喷涂智能工艺设计平台 · V1.6 发布说明

发布日期：2026-09-25 ｜ 基线：V1.5（commit 8fcb67d）｜ 升级方式：原地覆盖，无需迁移

---

## 一、本次升级主题

V1.6 不新增研究功能，聚焦三件事：**好用**（首页智能操作区）、**可信**（全程证据溯源 +
反套话红线）、**好看**（全站统一 Nature 论文级图表）。V1.5 的 13 项功能全部保留，零删减。

## 二、变更清单

### 1. 新增模块

| 模块 | 作用 |
|------|------|
| `src/plot_style.py` | 全站唯一色板定义处（中性灰/信号蓝/强调粉三家族 + 分类循环色）；Plotly Nature 模板；统一图入口（`parity_chart` / `importance_chart` / `nature_layout`）；统计注脚（`stat_note` / `add_stat_note`）；Demo 水印；图表下载三件套 |
| `src/evidence.py` | 证据注册表 `EvidenceRegistry`：结论文本唯一合法出口 `statement()`（模板取值即拼接、缺证降级、model 类强制 CV 限定、Demo 前缀统一添加）；`evidence_manifest.json` 读写；证据徽标 UI 组件 |
| `scripts/verify_v16.py` | 发布前全局校验：① 禁用色值扫描 ② 解读文本绝对化措辞检查 ③ 证据清单数值-文本一致性抽查（`--run <run_id>`） |

### 2. 首页智能操作区（P0-1 / P0-3）

- 两步路径：**拖入 Excel → 点「智能分析」**，进度步骤条实时显示八步流程；
- 三态展示：空态引导 / 就绪态 / 结果态摘要卡；
- 摘要卡四区：元信息（run_id、数据/模型版本、CV 方式）→ 证据区（各 Stage 指标表 +
  Top 特征重要性图，每条结论挂证据徽标）→ 结论区（可溯源结论句列表）→ 行动区
  （run_id 一键复制 + 结果 ZIP 下载 + 下一步指引）；
- 失败如实展示：检查项失败原因直接给出，不吞异常。

### 3. 全程可溯源（P0-2）

- `run_quick_analysis` 内部创建 `EvidenceRegistry` 并沿调用链传入
  `insights.build_insight_report` / `figure_analysis.analyze_figure`；
- 每个 run 落盘 `evidence_manifest.json`（10 条证据：ins.* 5 条 + qa.* 5 条）与
  `summary.json`，**均随结果 ZIP 打包**（位于 ZIP 内 `results/` 目录）；
- 洞察报告与图表解读末尾自动追加「附：证据溯源」附录；
- history record 增量字段 `evidence_manifest` / `summary`（旧字段不动，向后兼容）。

### 4. 全 Tab 四段式重组（P1-1 / P1-2 / P0-7）

- 每个 Tab 统一为：导语 + status-pill → 主证据区（首屏）→ 次级证据区（expander）→ 操作区；
- 首页 / ③ 训练 / ⑧ 输出三处指标**同源去重**：n / 数据版本 / 模型版本 / CV 方式全部读
  同一 `build_context()` 产物，禁止二次计算；
- ⑥ 数据洞察五个结论容器与 ⑦ 逆向设计均挂证据徽标 + 计算方式说明。

### 5. 图表 Nature 化（P0-4 / P0-5 / P1-4 / P1-5）

- 网页图（Plotly）：全部走 `plot_style` 统一入口，色板 / 字体 / 网格 / 图例位置统一；
- 论文图（matplotlib）：`paper_style` 改为 import 同源色板常量，新增
  89mm（单栏）/ 120mm（中幅）/ 183mm（双栏）版式 helper 与 `add_stat_box` 统计框；
- `paper_output` 约 30 处散落色值替换为同源常量；每张图注脚含 n / R² / RMSE / CV 方式；
  Demo 图自动加粉色水印；
- 网页图配「下载 PNG / 下载数据 CSV」三件套，与论文图导出对齐。

### 6. 全局 CSS 变量化（P1-3）

- 14 个 CSS 变量从 `plot_style` 常量生成（`:root {--gray-ink: ...}`），界面配色与图表同源；
- 全站仅智能操作区两键为实底 primary 强调按钮，其余按钮降为白底描边，视觉焦点唯一。

### 7. 收尾

- `README_先看我.md` 增补 V1.6 使用说明与开发者自检入口；
- `requirements.txt` 新增 `kaleido>=0.2.1`（Plotly 静态导出引擎）；
- `VERSION` → V1.6。

## 三、兼容性承诺

- **不动清单零改动**：`model_chain / optimize / literature / modes / research_utils /
  data_utils / features / config` 八模块 diff 为零（OOF/CV 训练逻辑、优化搜索、
  文献 Schema、模式隔离原样保留）；
- **隔离铁律不变**：Quick Run 只写 `runs/quick_analysis/<run_id>/`，不覆盖 `models/`；
  `promote_to_formal` 仍须用户主动确认；
- **旧调用兼容**：所有新函数参数带默认值（`registry=None` 等），V1.5 调用路径零修改可运行；
  V1.5 旧 run 在摘要卡显示「旧版运行，无溯源数据」提示，不影响查看；
- 13 项功能不减：4 研究模式 / 8 Tab / XGBoost 三级链 / 单点预测 / 数据解析 /
  数据洞察 / 逆向优化 / 结果输出 / 快速分析 / 八段式图表解读 / 文献 Schema /
  数据安全检查 / 运行日志，逐项核对通过。

## 四、自验证记录

| 任务 | commit | 验证 |
|------|--------|------|
| T01 底层 | 79c4c54 | imports 无副作用；缺证/NaN 降级；Demo 前缀 + model 类 CV 检查；manifest 读写往返；色板无残留色 |
| T02 中层 | 8d0a324 | demo 全流程 run：manifest 10 条 + summary 5 结论，数值逐条对照；旧调用兼容；缺证降级 |
| T03 表层 | 416b276 | Streamlit AppTest 全页执行 0 异常（8 Tab）；指标同源；唯一 primary 按钮 |
| T04 导出层 | 516da81 | `verify_v16.py` 全绿（10 文件色值 0 hit / 200 份 md 禁用词通过 / 证据抽查 10 条通过）；demo ZIP 实测含 `evidence_manifest.json` |
| T05 收尾 | 本次 | README/VERSION/requirements/发布说明更新；全局校验复跑全绿 |

## 五、已知限制

1. **Streamlit 无法编程切换 Tab**：摘要卡行动区为「下一步指引 + run_id 复制」降级方案，
   不能一键跳转到指定 Tab（受 Streamlit 组件能力限制，见设计文档 §8）；
2. **图表语言双轨**：论文导出图支持中/英双语，网页图当前为中文（与 V1.5 一致）；
3. **`research_utils` 中数据支持程度红绿灯色**沿用 V1.5 语义色（绿/黄/红），
   未并入统一色板——该模块属不动清单，语义色与装饰色板职责不同；
4. **旧 run 无证据清单**：V1.5 生成的历史 run 无 manifest，界面按「旧版运行」提示；
5. **macOS 中文字体回退**：论文图导出在无 STSong 字体的环境回退至系统默认字体，
   不影响数据与版式。

## 六、快速回归（升级后建议执行）

```
python scripts/verify_v16.py                     # 色值/禁用词
python scripts/verify_v16.py --run <最新run_id>  # 证据抽查
```

用 Demo 数据走一遍「智能分析 → 摘要卡 → ZIP 下载」，确认 ZIP 内含
`evidence_manifest.json` 与 `summary.json` 即回归通过。
