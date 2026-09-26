#!/bin/bash
# 双阳极等离子喷涂平台 — macOS 桌面启动器
# 逻辑：平台已运行 → 直接打开浏览器；未运行 → 后台启动并等待就绪后再打开。
# 由桌面 App（osacompile）或终端直接调用。

PROJECT="/Users/liangwenjie/Downloads/DoubleAnode_ML_Foolproof"
PORT=8501
URL="http://localhost:${PORT}"
if [ ! -f "$PROJECT/app.py" ] || [ ! -x "$PROJECT/.venv/bin/python" ]; then
  osascript -e 'display alert "平台目录或 Python 环境不存在" message "请检查下载目录中的 DoubleAnode_ML_Foolproof，并完成首次安装。"' 2>/dev/null || true
  exit 1
fi
cd "$PROJECT" || exit 1
if ! "$PROJECT/.venv/bin/python" -c 'from src import quick_analysis as q; assert all(callable(getattr(q,n,None)) for n in ("latest_runs","quick_analyze_file","load_smart_summary","save_quick_workspace","load_quick_workspace"))' >/dev/null 2>&1; then
  osascript -e 'display alert "平台程序文件未完整更新" message "app.py 与 src 模块版本不一致。请从 GitHub 下载完整 main ZIP，并运行其中的 04_同步源码.command 后再启动。"' 2>/dev/null || true
  exit 1
fi
LOG_DIR="$PROJECT/outputs/logs"
mkdir -p "$LOG_DIR"

check_health() {
  local code
  code=$(curl -s --noproxy '*' -o /dev/null -w "%{http_code}" --max-time 2 "$URL/_stcore/health" 2>/dev/null)
  [ "$code" = "200" ]
}

# 已在运行：直接打开浏览器，避免端口冲突
if check_health; then
  open "$URL"
  exit 0
fi

nohup "$PROJECT/.venv/bin/python" -m streamlit run app.py \
  --server.headless true --server.port "$PORT" --browser.gatherUsageStats false \
  >> "$LOG_DIR/streamlit_app.log" 2>&1 &

# 等待就绪（最多 30 秒）
i=0
while [ $i -lt 30 ]; do
  sleep 1
  if check_health; then
    break
  fi
  i=$((i + 1))
done

open "$URL"
