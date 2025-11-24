#!/usr/bin/env bash
# 中文说明：Linux 版训练脚本，占位示例；将参数原样传递给训练主程序
# 使用示例：
#   ./scripts/run_train.sh --config configs/train.yaml --epochs 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 可选：conda 环境名（若提供将优先使用）
ENV_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;
    --help|-h) echo "用法: $0 [--env ENV] [训练参数...]"; exit 0;;
    *) break;;
  esac
done

run_py() {
  # 中文说明：若提供ENV_NAME且系统存在conda，则使用该环境运行；否则使用系统python
  if command -v conda >/dev/null 2>&1 && [[ -n "${ENV_NAME}" ]]; then
    conda run -n "${ENV_NAME}" python "$@"
  else
    python "$@"
  fi
}

cd "${REPO_ROOT}"
echo "[step] 启动训练：传参=$*"
run_py -m src.train.main "$@"
echo "[done] 训练流程结束。"

