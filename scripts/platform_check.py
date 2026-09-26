# -*- coding: utf-8 -*-
"""平台环境自检脚本（macOS / Windows 通用）

检查项：
1. 操作系统与 CPU 架构
2. Python 版本（要求 >= 3.10）
3. 是否运行在虚拟环境中（.venv 或其他 venv）
4. 关键依赖是否可导入及版本（streamlit/pandas/numpy/sklearn/xgboost/plotly/joblib/openpyxl/yaml）
5. xgboost 原生库能否真实加载（小型拟合冒烟）
6. 项目关键文件/目录是否齐全

用法：python scripts/platform_check.py
返回码：0 = 全部通过；1 = 存在失败项
"""
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def in_venv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


# 1. 操作系统
system = platform.system()
arch = platform.machine()
check("操作系统识别", system in ("Darwin", "Windows", "Linux"), f"{system} / {arch}")

# 2. Python 版本
ver = sys.version_info
check("Python 版本 >= 3.10", ver >= (3, 10), f"{platform.python_version()}")

# 3. 虚拟环境
project_venv = (ROOT / ".venv").exists()
check("运行于虚拟环境", in_venv(),
      f"sys.prefix={sys.prefix}" if in_venv() else
      ("检测到项目 .venv 目录存在但当前未激活" if project_venv else "未检测到虚拟环境（建议先运行安装脚本）"))

# 4. 关键依赖
deps = ["streamlit", "pandas", "numpy", "sklearn", "xgboost",
        "plotly", "joblib", "openpyxl", "yaml"]
for mod_name in deps:
    try:
        mod = importlib.import_module(mod_name)
        ver_str = getattr(mod, "__version__", "unknown")
        check(f"依赖可导入: {mod_name}", True, ver_str)
    except Exception as e:
        check(f"依赖可导入: {mod_name}", False, repr(e))

# 5. xgboost 原生库真实加载（Windows 上若失败需装 VC++ 运行库）
try:
    import numpy as np
    from xgboost import XGBRegressor
    X = np.random.default_rng(0).random((20, 3))
    y = X[:, 0] * 2 + 1
    m = XGBRegressor(n_estimators=5, max_depth=2)
    m.fit(X, y)
    m.predict(X[:2])
    check("xgboost 原生库加载与小型拟合", True)
    print("      提示: 若在其他机器报 OpenMP/libxgboost 错误 ——")
    print("            Windows: 安装 VC++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe)")
    print("            macOS:   安装 libomp (brew install libomp)")
except Exception as e:
    check("xgboost 原生库加载与小型拟合", False, repr(e))

# 6. 项目文件完整性
required = [
    "app.py", "requirements.txt",
    "config/data_schema.yaml", "config/objectives.yaml", "config/data_dictionary.yaml",
    "src/config.py", "src/data_utils.py", "src/features.py",
    "src/model_chain.py", "src/optimize.py", "src/ui_labels.py",
    "scripts/smoke_test.py", "scripts/leakage_test.py",
    "data/demo/DEMO_双阳极喷涂数据.xlsx",
    "data/template/真实数据录入模板.xlsx",
]
for rel in required:
    check(f"文件存在: {rel}", (ROOT / rel).exists())

# app.py reads these methods on every rerun. A partial copy of app.py alone
# can pass dependency checks yet crash as soon as a Quick Run is selected.
try:
    sys.path.insert(0, str(ROOT))
    from src import quick_analysis as quick_module
    for name in ("latest_runs", "quick_analyze_file", "load_smart_summary",
                 "save_quick_workspace", "load_quick_workspace"):
        check(f"智能分析模块接口: {name}", callable(getattr(quick_module, name, None)),
              f"{quick_module.__file__}")
except Exception as e:
    check("智能分析模块可导入", False, repr(e))

# 结果
print()
if failures:
    print(f"PLATFORM CHECK FAILED — {len(failures)} 项未通过: {failures}")
    sys.exit(1)
print(f"PLATFORM CHECK PASSED — {system} 环境就绪")
