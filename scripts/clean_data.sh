#!/usr/bin/env bash
# 中文说明：Linux 版 CSV 清洗与整合脚本
# 功能：筛选列、按用户+内容去重，并可生成合并汇总CSV
# 使用示例：
#   ./scripts/clean_data.sh --env cuda_env \
#     --src-dir "data/vendor/crawl/by_bvid" \
#     --out-dir "data/vendor/crawl/cleaned" \
#     --combined-out "data/vendor/crawl/cleaned_combined.csv"

set -euo pipefail

# 中文说明：定位脚本所在目录与项目根目录（支持从任意位置运行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================= 默认配置（可通过命令行参数覆盖） =========================
ENV_NAME=""
SRC_DIR="${REPO_ROOT}/data/vendor/crawl/by_bvid"       # 原始CSV目录
OUT_DIR="${REPO_ROOT}/data/vendor/crawl/cleaned"       # 清洗后输出目录
COMBINED_OUT="${REPO_ROOT}/data/vendor/crawl/cleaned_combined.csv"  # 合并汇总CSV
PATTERN="*.csv"                                         # 匹配模式

# ========================= 参数解析 =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;
    --src-dir) SRC_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --combined-out) COMBINED_OUT="$2"; shift 2;;
    --pattern) PATTERN="$2"; shift 2;;
    --help|-h)
      echo "用法: $0 [--env ENV] [--src-dir DIR] [--out-dir DIR] [--combined-out FILE] [--pattern GLOB]"; exit 0;;
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
echo "[step] 清洗与整合：src=${SRC_DIR} → out=${OUT_DIR} combined=${COMBINED_OUT} pattern=${PATTERN}"
mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
run_py -m src.data.clean_crawled_csv --src-dir "${SRC_DIR}" --out-dir "${OUT_DIR}" --combined-out "${COMBINED_OUT}" --pattern "${PATTERN}"

echo "[done] 数据清洗与合并完成。输出目录：${OUT_DIR}；合并：${COMBINED_OUT}"

