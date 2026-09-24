@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 尚未完成 Windows 首次安装，请先运行 01_首次安装_Windows.bat
  pause
  exit /b 1
)
echo 正在启动平台，浏览器访问 http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
pause
