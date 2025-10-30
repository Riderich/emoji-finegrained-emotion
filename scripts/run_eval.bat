@echo off
REM 评估脚本（Windows）
REM 说明：该脚本用于启动最小评估流程，占位示例。

REM 激活虚拟环境（如已激活可忽略）
IF EXIST .\.venv\Scripts\activate.bat (
  call .\.venv\Scripts\activate.bat
) ELSE (
  echo [提示] 未检测到 .venv，建议先创建並激活虛擬環境。
)

REM 运行评估主程序
python -m src.eval.main %*

REM 结束
exit /b 0