#!/bin/bash
cd "$(dirname "$0")"
echo "============================================"
echo " 双阳极喷涂 ML 平台 - 首次安装"
echo "============================================"
if ! command -v python3 >/dev/null 2>&1; then
  echo "未检测到 Python 3。请先安装 Python 3.11 或 3.12，然后重新双击本文件。"
  read -p "按回车退出..."
  exit 1
fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "安装完成。以后无需再次运行本文件。"
echo "请双击：02_启动平台.command"
read -p "按回车关闭..."
