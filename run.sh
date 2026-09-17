#!/usr/bin/env bash
# 一条命令跑起来：创建虚拟环境 → 安装依赖 → 启动网页界面
#   ./run.sh                 默认端口 8760
#   ./run.sh --port 9000     换端口
#   ./run.sh --no-browser    不自动开浏览器
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "[1/2] 创建虚拟环境 .venv ..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[错误] 找不到 python3，请先安装 Python 3.10+"
    exit 1
  fi
  python3 -m venv .venv
  "$PY" -m pip install --upgrade pip --quiet
else
  echo "[1/2] 已有虚拟环境 .venv，跳过创建"
fi

# 依赖装过了就不再重装（首次会下载 PyTorch 等，比较慢）
if ! "$PY" -c "import langgraph, langchain_openai, faiss, sentence_transformers" >/dev/null 2>&1; then
  echo "[2/2] 安装依赖（首次运行需要几分钟，会下载 PyTorch 与嵌入模型）..."
  "$PY" -m pip install -r requirements.txt
else
  echo "[2/2] 依赖已就绪"
fi

echo
echo "启动中，稍等片刻浏览器会自动打开 ..."
exec "$PY" main.py ui "$@"
