@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ============================================
echo  双阳极等离子喷涂智能工艺设计平台
echo  Windows 首次安装
echo ============================================

rem 优先使用 Python 3.12 / 3.11
set "PYEXE="
py -3.12 -c "import sys" >nul 2>nul && set "PYEXE=py -3.12"
if not defined PYEXE ( py -3.11 -c "import sys" >nul 2>nul && set "PYEXE=py -3.11" )
if not defined PYEXE (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYEXE=python"
  )
)
if not defined PYEXE (
  echo 未检测到 Python 3.11/3.12，请先安装 Python 并勾选 Add Python to PATH。
  pause
  exit /b 1
)

echo 使用解释器: %PYEXE%
echo 正在创建 Windows 专用虚拟环境 .venv ...
%PYEXE% -m venv .venv
if errorlevel 1 ( echo [错误] 创建虚拟环境失败。 & pause & exit /b 1 )

echo 正在升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo 正在安装 requirements.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo [错误] 依赖安装失败，请检查网络后重试。 & pause & exit /b 1 )

echo.
echo 基础依赖检查 ...
".venv\Scripts\python.exe" scripts\platform_check.py
echo.
echo 首次安装完成。之后请双击 02_启动平台_Windows.bat 启动平台。
echo 如 xgboost 报 OpenMP/libxgboost 错误，请安装 VC++ 运行库：
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
pause
