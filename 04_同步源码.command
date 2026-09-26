#!/bin/bash
# 在最新 GitHub ZIP 解压目录双击。只把程序文件同步至 Mac 的现有安装目录。
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$HOME/Downloads/DoubleAnode_ML_Foolproof}"

echo "新代码：$SOURCE"
echo "现有平台：$TARGET"
if [ "$SOURCE" = "$TARGET" ]; then
  echo "请先从 GitHub 下载完整 main ZIP 并解压，从新文件夹运行本脚本。"
  read -r -p "按回车关闭..." _
  exit 1
fi

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$TARGET/.venv/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "找不到 Python 3；请先运行现有平台的 01_首次安装.command。"
  read -r -p "按回车关闭..." _
  exit 1
fi

"$PYTHON_BIN" "$SOURCE/scripts/safe_code_update.py" --target "$TARGET"

# 只停止从本安装目录启动、占用 8501 端口的 Streamlit 进程。
# 端口被其他程序使用时不做任何操作。
if command -v lsof >/dev/null 2>&1; then
  for PID in $(lsof -nP -tiTCP:8501 -sTCP:LISTEN 2>/dev/null || true); do
    CMD="$(ps -p "$PID" -o command= 2>/dev/null || true)"
    case "$CMD" in
      *"$TARGET/.venv/"*"streamlit"*"app.py"*)
        echo "关闭旧版平台进程 $PID，下一次启动将载入完整新源码。"
        kill -TERM "$PID" 2>/dev/null || true
        ;;
    esac
  done
fi

echo "请双击现有平台桌面按钮或 02_启动平台.command 重新打开。"
read -r -p "按回车关闭..." _
