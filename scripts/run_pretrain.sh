#!/usr/bin/env bash
# 中文说明：Linux 版检索训练入口脚本（Top-1 / Top-5）。
# 说明：当前 Python 训练入口已内置参数与数据路径，脚本仅负责调用。
# 用法示例：
#   1) 赋权：chmod +x scripts/run_pretrain.sh
#   2) 运行：scripts/run_pretrain.sh --env your_conda_env

set -euo pipefail

# 中文说明：定位脚本所在目录与项目根目录（支持从任意位置运行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================= 可选配置（仅用于选择conda环境） =========================
ENV_NAME=""    # 中文说明：conda 环境名；若提供则优先使用 conda run -n

# ========================= 参数解析 =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;                    # 中文说明：指定 conda 环境名
    --help|-h)
      echo "用法: $0 [--env CONDA_ENV]"; exit 0;;      # 中文说明：仅支持 --env 选择环境
    *) echo "[warn] 未识别参数: $1（训练超参请直接修改 Python 脚本）"; shift;;
  esac
done

# ========================= 运行辅助函数 =========================
run_py() {
  # 中文说明：若提供ENV_NAME且系统存在conda，则使用该环境运行；否则使用系统python
  if command -v conda >/dev/null 2>&1 && [[ -n "${ENV_NAME}" ]]; then
    conda run -n "${ENV_NAME}" python "$@"
  else
    python "$@"
  fi
}

# ========================= 执行训练入口 =========================
cd "${REPO_ROOT}"

# 中文说明：提示设备信息（如存在 nvidia-smi 则打印 GPU 信息）
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] GPU 可用：$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
else
  echo "[info] 未检测到 GPU；训练将以 CPU/无AMP 运行。"
fi

echo "[step] 启动检索训练入口：python -m src.train.contrastive.run_pretrain"
run_py -m src.train.contrastive.run_pretrain
echo "[done] 训练流程结束。检查点位于 ./checkpoints/（best/last）。"
