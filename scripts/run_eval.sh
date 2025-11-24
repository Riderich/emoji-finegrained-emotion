#!/usr/bin/env bash
# 中文说明：Linux 版评估脚本，占位示例；将参数原样传递给评估主程序
# 使用示例：
#   ./scripts/run_eval.sh --config configs/eval.yaml --model-path checkpoints/latest.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 可选：conda 环境名（若提供将优先使用）
ENV_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;
    --help|-h) echo "用法: $0 [--env ENV] [评估参数...]"; exit 0;;
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
echo "[step] 启动评估：传参=$*"
run_py -m src.eval.main "$@"
echo "[done] 评估流程结束。"

