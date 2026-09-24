# SYNC_GUIDE — Windows / macOS 跨平台同步说明

## 核心原则

**Windows 和 macOS 共用同一套源码（同一个 Git 仓库），但各自使用自己的 `.venv` 虚拟环境。**

- 源码、配置（`config/`）、测试（`scripts/`）、数据模板与 Demo 数据通过 Git 同步；
- `.venv/` 属于各机器的本地运行环境，**绝不提交、绝不跨系统复制**
  （Windows 的 `.venv` 含 `.venv\Scripts\`，macOS 的含 `.venv/bin/`，二者互不兼容）；
- 模型文件（`models/`）、输出产物（`outputs/`、`runs/`）由各机器用相同数据训练/生成，
  不进版本库（`.gitignore` 已排除）。

## 首次在 macOS 上部署本仓库

1. `git clone <仓库地址>` 后，在项目根目录执行：
   ```
   python3.12 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   ```
   （或直接双击 `01_首次安装.command`）
2. 如 xgboost 报 OpenMP 错误：`brew install libomp`。
   **不要把 `libomp.dylib` 提交进仓库**（已在 `.gitignore` 中排除）。
3. 启动：双击 `02_启动平台.command`，浏览器访问 http://localhost:8501。

## 首次在 Windows 上部署本仓库

1. `git clone <仓库地址>` 后，双击 `01_首次安装_Windows.bat`
   （需已安装 Python 3.11+ 并勾选 Add to PATH）。
2. 如 xgboost 报 libxgboost.dll / OpenMP 错误，安装
   [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。
3. 启动：双击 `02_启动平台_Windows.bat`。

## 日常同步流程

```
# 修改代码 / 数据后（任一机器）
git add -A
git commit -m "说明本次修改"
git push

# 另一台机器
git pull
```

- 两台机器可同时运行各自的 Streamlit 实例，互不影响；
- 训练得到的模型与论文图表在各自机器的 `models/`、`outputs/` 中，
  如需跨机器比较，请通过网盘/U盘单独传输，不要走 Git。

## 修改代码前的回归验证

提交前建议在本地跑一遍测试（Windows 与 macOS 命令相同，仅 python 路径不同）：

```
# Windows
.venv\Scripts\python.exe scripts/smoke_test.py
.venv\Scripts\python.exe scripts/leakage_test.py
.venv\Scripts\python.exe scripts/output_test.py
.venv\Scripts\python.exe scripts/usability_test.py
# 以及 V1.4/V1.5 新增测试（literature/mode_switch/cascade/melting/optional/ui_*/quick/insight/context/figure）
.venv\Scripts\python.exe scripts/v132_check.py

# macOS
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/v132_check.py
...
```

## 版本号

仓库根目录 `VERSION` 文件记录当前正式版本（当前为 **V1.5**）。
每次正式版本升级时同步更新该文件。

## 环境要求（两平台一致）

- Python 3.11–3.13（详见 requirements.txt）
- 依赖安装一律使用本机 `.venv`，禁止升级全局 Python 包
