#!/usr/bin/env bash
# 中文说明：Linux 版数据准备脚本，下载 OpenMoji 面部 Emoji 并进行图像增强
# 使用方法示例：
#   ./scripts/prepare_data.sh --env cuda_env \
#     --src "data/emoji_images/openmoji" --dst "data/emoji_images/openmoji_aug"

set -euo pipefail

# 中文说明：定位脚本所在目录与项目根目录（支持从任意位置运行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================= 默认配置（可通过命令行参数覆盖） =========================
ENV_NAME=""                                 # 可选：conda 环境名；若提供将优先用 conda run -n
SRC_DIR="${REPO_ROOT}/data/emoji_images/openmoji"       # 下载目录（OpenMoji）
DST_DIR="${REPO_ROOT}/data/emoji_images/openmoji_aug"   # 增强输出目录

# ========================= 参数解析 =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;
    --src) SRC_DIR="$2"; shift 2;;
    --dst) DST_DIR="$2"; shift 2;;
    --help|-h)
      echo "用法: $0 [--env ENV] [--src DIR] [--dst DIR]"; exit 0;;
    *) echo "[warn] 未识别参数: $1"; shift;;
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

# ========================= 执行流程 =========================
echo "[step] 下载 OpenMoji 到 ${SRC_DIR}"
mkdir -p "${SRC_DIR}"
cd "${REPO_ROOT}"
run_py -m src.data.download_openmoji

echo "[step] 图像增强：src=${SRC_DIR} → dst=${DST_DIR}"
mkdir -p "${DST_DIR}"
run_py -m src.data.augment_images --src "${SRC_DIR}" --dst "${DST_DIR}"

echo "[done] OpenMoji 下载与增强完成。"

