@echo off
rem 一条命令跑起来：创建虚拟环境 → 安装依赖 → 启动网页界面
rem   run.bat                 默认端口 8760
rem   run.bat --port 9000     换端口
rem   run.bat --no-browser    不自动开浏览器
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo [1/2] 创建虚拟环境 .venv ...
  where python >nul 2>nul
  if errorlevel 1 (
    echo [错误] 找不到 python，请先安装 Python 3.10+ 并加入 PATH：https://www.python.org/downloads/
    exit /b 1
  )
  python -m venv .venv
  if errorlevel 1 exit /b 1
  "%PY%" -m pip install --upgrade pip --quiet
) else (
  echo [1/2] 已有虚拟环境 .venv，跳过创建
)

rem 依赖装过了就不再重装（首次会下载 PyTorch 等，比较慢）
"%PY%" -c "import langgraph, langchain_openai, faiss, sentence_transformers" >nul 2>nul
if errorlevel 1 (
  echo [2/2] 安装依赖（首次运行需要几分钟，会下载 PyTorch 与嵌入模型）...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试
    exit /b 1
  )
) else (
  echo [2/2] 依赖已就绪
)

echo.
echo 启动中，稍等片刻浏览器会自动打开 ...
"%PY%" main.py ui %*
