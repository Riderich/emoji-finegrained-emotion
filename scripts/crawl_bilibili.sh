#!/usr/bin/env bash
# 中文说明：一键运行的Bash脚本（Linux），实现：批量获取BV → BV→AID后抓取评论 → 清洗与合并

set -euo pipefail

# 中文说明：定位脚本所在目录与项目根目录（确保可从任意位置运行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================= 默认配置（可通过命令行参数覆盖） =========================
# 中文说明：可选conda环境名（若提供将优先使用 conda run -n）
ENV_NAME=""
# 中文说明：不再从脚本传入 SESSDATA；Python 内部会从 src/data/cookies.json 读取完整 Cookie

# 中文说明：关键词文件与输出路径（均为绝对路径，便于跨目录执行）
KEYWORDS_FILE="${REPO_ROOT}/data/vendor/keywords_bilibili.txt"
OUT_SEED="${REPO_ROOT}/data/vendor/crawl/seed_bvids.csv"

# 中文说明：抓取评论输出目录与映射文件
PER_DIR="${REPO_ROOT}/data/vendor/crawl/by_bvid"
CLEAN_DIR="${REPO_ROOT}/data/vendor/crawl/cleaned"
COMBINED_OUT="${REPO_ROOT}/data/vendor/crawl/cleaned_combined.csv"
EMOJI_MAP="${REPO_ROOT}/data/vendor/bilibili_emojiall_map.json"

# 中文说明：搜索/热门与爬取参数
# 获取模式：search=按关键词搜索；popular=按热门榜分页（更贴近“视频热门度”）
FETCH_MODE="popular"
FETCH_MAX_PAGES=5        # search: 每关键词页数；popular: 热门榜页数
FETCH_ORDER="pubdate"    # 仅在search模式下生效：pubdate/view
FETCH_SLEEP=1.2          # 获取阶段请求间歇秒数（内部含随机抖动，建议≥1.0）
FETCH_MIN_REPLY=0        # 按评论数过滤（>=该值保留；0表示不过滤）
FETCH_PS=20              # popular模式每页数量

# 中文说明：默认抓取页数上限为 800；遇到 -400（max offset exceeded）时提前停止该BV
CRAWL_MAX_PAGES=800      # 每个BV抓取评论页数上限（推荐：800页）
CRAWL_SLEEP=1.6          # 评论抓取分页间歇秒数（含随机抖动，建议≥1.2）
MIN_PER_BVID=500         # 每个BV至少写出N条（按你的设想：至少500条）

# 中文说明：调试/诊断选项（可选）
DUMP_RAW=""             # 若提供路径，则对每个BV转储原始响应到该TXT并退出抓取
PRINT_FIRST=0            # 若>0，则打印首批评论文本条数并退出抓取
SKIP_FETCH=0             # 若=1，跳过Step1，直接使用现有种子CSV

# ========================= 参数解析 =========================
# 中文说明：支持覆盖主要参数，示例：
# ./crawl_bilibili.sh --env cuda_env --sessdata "xxx" --fetch-max-pages 8 --crawl-max-pages 4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;                        # 指定conda环境名
    --keywords-file) KEYWORDS_FILE="$2"; shift 2;;         # 指定关键词文件
    --seed-out|--out-seed) OUT_SEED="$2"; shift 2;;        # 指定BV种子输出CSV
    --fetch-mode) FETCH_MODE="$2"; shift 2;;               # 获取模式：search/popular
    --fetch-max-pages) FETCH_MAX_PAGES="$2"; shift 2;;     # 搜索页数
    --fetch-order) FETCH_ORDER="$2"; shift 2;;             # 搜索排序
    --fetch-sleep) FETCH_SLEEP="$2"; shift 2;;             # 搜索间歇
    --fetch-min-reply) FETCH_MIN_REPLY="$2"; shift 2;;     # 按评论数过滤
    --fetch-ps) FETCH_PS="$2"; shift 2;;                   # popular模式每页数量
    --crawl-max-pages) CRAWL_MAX_PAGES="$2"; shift 2;;     # 评论页数
    --crawl-sleep) CRAWL_SLEEP="$2"; shift 2;;             # 评论间歇
    --min-per-bvid) MIN_PER_BVID="$2"; shift 2;;           # 每BV最小条数
    --per-dir) PER_DIR="$2"; shift 2;;                     # 每BV输出目录
    --clean-dir) CLEAN_DIR="$2"; shift 2;;                 # 清洗输出目录
    --combined-out) COMBINED_OUT="$2"; shift 2;;           # 合并输出CSV
    --emoji-map) EMOJI_MAP="$2"; shift 2;;                 # 映射文件路径
    --dump-raw) DUMP_RAW="$2"; shift 2;;                    # 为抓取阶段开启原始响应转储并退出
    --print-first) PRINT_FIRST="$2"; shift 2;;              # 打印首批评论文本条数并退出
    --skip-fetch) SKIP_FETCH=1; shift 1;;                    # 跳过Step1（使用现有种子）
    --help|-h)
      echo "用法: $0 [--env ENV] [--fetch-mode search|popular] [--keywords-file FILE] [--fetch-max-pages N] [--fetch-min-reply N] [--crawl-max-pages N] ...";
      exit 0;;
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

# ========================= 开始执行流程 =========================
echo "[step1] 获取BV：mode=${FETCH_MODE} 输出=${OUT_SEED} (skip=${SKIP_FETCH})"
mkdir -p "$(dirname "${OUT_SEED}")"

# 中文说明：切换到项目根（内部src包位于该目录），以便用 -m 运行模块
cd "${REPO_ROOT}"

if [[ "${SKIP_FETCH}" -eq 0 ]]; then
  if [[ "${FETCH_MODE}" == "search" ]]; then
    # 中文说明：关键词搜索模式
    FETCH_ARGS=( -m src.data.fetch_bilibili_bv --mode search --keywords-file "${KEYWORDS_FILE}" --max-pages "${FETCH_MAX_PAGES}" --order "${FETCH_ORDER}" --sleep-seconds "${FETCH_SLEEP}" --min-reply "${FETCH_MIN_REPLY}" --out "${OUT_SEED}" )
  else
    # 中文说明：热门榜模式（更贴近视频热门度）；不需要关键词文件
    FETCH_ARGS=( -m src.data.fetch_bilibili_bv --mode popular --max-pages "${FETCH_MAX_PAGES}" --ps "${FETCH_PS}" --sleep-seconds "${FETCH_SLEEP}" --min-reply "${FETCH_MIN_REPLY}" --out "${OUT_SEED}" )
  fi
  # 中文说明：无需传入 SESSDATA；Python 会自动读取 cookies.json
  run_py "${FETCH_ARGS[@]}"
else
  echo "[info] 跳过Step1，使用现有种子CSV：${OUT_SEED}"
fi

echo "[step2] 按BV转换为AID后抓取评论：bvids-file=${OUT_SEED} per-dir=${PER_DIR} 映射=${EMOJI_MAP}"
mkdir -p "${PER_DIR}"

CRAWL_ARGS=( -m src.data.crawl_bilibili --root "." --bvids-file "${OUT_SEED}" --max-pages "${CRAWL_MAX_PAGES}" --sleep-seconds "${CRAWL_SLEEP}" --emoji-map "${EMOJI_MAP}" --min-per-bvid "${MIN_PER_BVID}" --per-bvid-output-dir "${PER_DIR}" )
# 中文说明：上述 "min-per-bvid" 按“写出行数”统计（每个表情一行、含Unicode与[doge]类占位）。
# 如需严格按“评论条数”统计并提前停止，可后续新增参数（例如 --min-comments-per-bvid）。
 # 中文说明：无需传入 SESSDATA；Python 会自动读取 cookies.json

# 中文说明：若指定了 DUMP_RAW 或 PRINT_FIRST，则进入诊断流程并在抓取阶段退出
if [[ -n "${DUMP_RAW}" ]]; then
  echo "[step2:diagnose] 原始响应转储（BV→AID）：dump-raw=${DUMP_RAW}"
  CRAWL_DIAG=( "${CRAWL_ARGS[@]}" --dump-raw "${DUMP_RAW}" )
  run_py "${CRAWL_DIAG[@]}"
  echo "[done] 诊断转储完成。"
  exit 0
elif [[ "${PRINT_FIRST}" -gt 0 ]]; then
  echo "[step2:diagnose] 打印首批评论文本：count=${PRINT_FIRST}"
  CRAWL_DIAG=( "${CRAWL_ARGS[@]}" --print-first-messages "${PRINT_FIRST}" )
  run_py "${CRAWL_DIAG[@]}"
  echo "[done] 首批评论打印完成。"
  exit 0
else
  run_py "${CRAWL_ARGS[@]}"
fi

echo "[step3] 清洗列并合并：src-dir=${PER_DIR} out-dir=${CLEAN_DIR} combined=${COMBINED_OUT}"
mkdir -p "${CLEAN_DIR}"
run_py -m src.data.clean_crawled_csv --src-dir "${PER_DIR}" --out-dir "${CLEAN_DIR}" --combined-out "${COMBINED_OUT}"

echo "[done] 全流程完成。清洗后的CSV目录：${CLEAN_DIR}；合并输出：${COMBINED_OUT}"
