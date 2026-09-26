#!/bin/bash
# 双阳极等离子喷涂平台 — macOS 桌面启动器
# 逻辑：平台已运行 → 直接打开浏览器；未运行 → 后台启动并等待就绪后再打开。
# 由桌面 App（osacompile）或终端直接调用。

PROJECT="/Users/liangwenjie/Downloads/DoubleAnode_ML_Foolproof"
PORT=8501
URL="http://localhost:${PORT}"
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

cd "$PROJECT" || exit 1
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
