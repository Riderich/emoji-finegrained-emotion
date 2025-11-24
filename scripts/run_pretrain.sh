#!/usr/bin/env bash
# 中文说明：Linux 版自监督预训练入口，包含断点续训与检查点保存
# 你可以通过参数覆盖默认路径与训练设置
# 使用示例：
#   ./scripts/run_pretrain.sh --env cuda_env --epochs 3 --batch-size 32

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================= 默认配置 =========================
ENV_NAME=""
CSV_DIR="${REPO_ROOT}/data/vendor/crawl/cleaned"                  # 清洗后的CSV目录
EXTRA_COMBINED=""                                                 # 额外合并文件（可留空）
CACHE_DIR="${REPO_ROOT}/data/vendor/emote_cache"                  # 表情资源缓存目录
EMOJI_MAP="${REPO_ROOT}/data/vendor/bilibili_emojiall_map.json"   # 表情名称映射JSON
NAME_IMAGE_MAP="${REPO_ROOT}/data/vendor/bilibili_image_name_map.csv" # 名称到图片映射CSV
PREFER_LOCAL_EMOJI=1                                              # 优先使用本地表情（布尔）
LOCAL_ONLY=1                                                      # 仅本地资源（布尔）
TEXT_SENTINEL=1                                                   # 启用文本哨兵（布尔）
SENTINEL_TOKEN="[EMOJI]"                                         # 哨兵占位符
TEXT_FIELD="sentence"                                            # 文本字段名
EPOCHS=1
BATCH_SIZE=16
NUM_WORKERS=0
RESUME=1                                                          # 断点续训（布尔）
SAVE_DIR="${REPO_ROOT}/checkpoints"                               # 检查点输出目录

# ========================= 参数解析 =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;
    --csv-dir) CSV_DIR="$2"; shift 2;;
    --extra-combined) EXTRA_COMBINED="$2"; shift 2;;
    --cache-dir) CACHE_DIR="$2"; shift 2;;
    --emoji-map) EMOJI_MAP="$2"; shift 2;;
    --name-image-map) NAME_IMAGE_MAP="$2"; shift 2;;
    --sentinel-token) SENTINEL_TOKEN="$2"; shift 2;;
    --text-field) TEXT_FIELD="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --save-dir) SAVE_DIR="$2"; shift 2;;
    --no-prefer-local-emoji) PREFER_LOCAL_EMOJI=0; shift 1;;
    --no-local-only) LOCAL_ONLY=0; shift 1;;
    --no-text-sentinel) TEXT_SENTINEL=0; shift 1;;
    --no-resume) RESUME=0; shift 1;;
    --help|-h)
      echo "用法: $0 [--env ENV] [--csv-dir DIR] [--cache-dir DIR] [--emoji-map FILE] [--name-image-map FILE] [--epochs N] [--batch-size N] [--num-workers N] [--save-dir DIR] [--sentinel-token TOK] [--text-field NAME] [--no-prefer-local-emoji] [--no-local-only] [--no-text-sentinel] [--no-resume]"; exit 0;;
    *) echo "[warn] 未识别参数: $1"; shift;;
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

# ========================= 构建参数与运行 =========================
cd "${REPO_ROOT}"
ARGS=( -m src.train.contrastive.run_pretrain
  --csv-dir "${CSV_DIR}"
  --extra-combined "${EXTRA_COMBINED}"
  --cache-dir "${CACHE_DIR}"
  --emoji-map "${EMOJI_MAP}"
  --name-image-map "${NAME_IMAGE_MAP}"
  --sentinel-token "${SENTINEL_TOKEN}"
  --text-field "${TEXT_FIELD}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
)

# 中文说明：布尔选项按默认开启；如指定 --no-* 参数则关闭对应功能
if [[ ${PREFER_LOCAL_EMOJI} -eq 1 ]]; then ARGS+=( --prefer-local-emoji ); fi
if [[ ${LOCAL_ONLY} -eq 1 ]]; then ARGS+=( --local-only ); fi
if [[ ${TEXT_SENTINEL} -eq 1 ]]; then ARGS+=( --text-sentinel ); fi
if [[ ${RESUME} -eq 1 ]]; then ARGS+=( --resume ); fi

echo "[step] 启动自监督预训练：${ARGS[*]}"
run_py "${ARGS[@]}"
echo "[done] 预训练流程结束。"

