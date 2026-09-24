#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "尚未安装环境，请先双击 01_首次安装.command"
  read -p "按回车关闭..."
  exit 1
fi
source .venv/bin/activate
streamlit run app.py
