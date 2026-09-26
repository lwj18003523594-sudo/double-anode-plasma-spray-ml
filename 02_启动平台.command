#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "尚未安装环境，请先双击 01_首次安装.command"
  read -p "按回车关闭..."
  exit 1
fi
source .venv/bin/activate
if ! python -c 'from src import quick_analysis as q; assert all(callable(getattr(q,n,None)) for n in ("latest_runs","quick_analyze_file","load_smart_summary","save_quick_workspace","load_quick_workspace"))' >/dev/null 2>&1; then
  echo "平台程序文件版本不一致，请从 GitHub 下载完整 main ZIP，双击其中的 04_同步源码.command。"
  read -r -p "按回车关闭..." _
  exit 1
fi
streamlit run app.py
