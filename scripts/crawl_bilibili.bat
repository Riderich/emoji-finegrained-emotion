@echo off
REM 批量获取B站BV号并在 BV→AID 后爬取评论，随后清洗为训练集可用CSV

IF "%CONDA_DEFAULT_ENV%"=="" (
  echo [提示] 建议激活或使用 conda 运行指定环境：cuda_env
)

REM 中文说明：不再在脚本层传入 SESSDATA；Python 会在运行时读取 src\data\cookies.json 中的完整 Cookie

REM 第一步：获取 BV 种子文件（支持热门榜或关键词搜索）
set OUT_SEED=emoji-finegrained-emotion\data\vendor\crawl\seed_bvids.csv
set FETCH_MODE=popular
set FETCH_MIN_REPLY=0
set FETCH_MAX_PAGES=5
set FETCH_SLEEP=1.2
set FETCH_PS=20
set KEYWORDS_FILE=emoji-finegrained-emotion\data\vendor\keywords_bilibili.txt

IF /I "%FETCH_MODE%"==search (
  echo [step1] 获取BV（search）：关键词文件=%KEYWORDS_FILE% 输出=%OUT_SEED%
  conda run -n cuda_env python -m src.data.fetch_bilibili_bv ^
    --mode search ^
    --keywords-file "%KEYWORDS_FILE%" ^
    --max-pages %FETCH_MAX_PAGES% ^
    --order pubdate ^
    --sleep-seconds %FETCH_SLEEP% ^
    --min-reply %FETCH_MIN_REPLY% ^
    --out "%OUT_SEED%" || goto :error
) ELSE (
  echo [step1] 获取BV（popular）：pages=%FETCH_MAX_PAGES% ps=%FETCH_PS% 输出=%OUT_SEED%
  conda run -n cuda_env python -m src.data.fetch_bilibili_bv ^
    --mode popular ^
    --max-pages %FETCH_MAX_PAGES% ^
    --ps %FETCH_PS% ^
    --sleep-seconds %FETCH_SLEEP% ^
    --min-reply %FETCH_MIN_REPLY% ^
    --out "%OUT_SEED%" || goto :error
)

REM 第二步：按 BV 转换为 AID 后批量抓取评论，按表情映射过滤，分别输出与汇总
set ROOT=emoji-finegrained-emotion
set PER_DIR=emoji-finegrained-emotion\data\vendor\crawl\by_bvid
set EMOJI_MAP=emoji-finegrained-emotion\data\vendor\bilibili_emojiall_map.json
conda run -n cuda_env python -m src.data.crawl_bilibili ^
  --root "%ROOT%" ^
  --bvids-file "%OUT_SEED%" ^
  REM 中文说明：默认抓取页数上限调整为 800；遇到 -400（max offset exceeded）时提前停止该BV
  --max-pages 800 ^
  --sleep-seconds 1.6 ^
  --emoji-map "%EMOJI_MAP%" ^
  --min-per-bvid 500 ^
  --per-bvid-output-dir "%PER_DIR%" || goto :error

REM 第三步：清洗列并合并输出供训练使用
set CLEAN_DIR=emoji-finegrained-emotion\data\vendor\crawl\cleaned
set COMBINED_OUT=emoji-finegrained-emotion\data\vendor\crawl\cleaned_combined.csv
conda run -n cuda_env python -m src.data.clean_crawled_csv ^
  --src-dir "%PER_DIR%" ^
  --out-dir "%CLEAN_DIR%" ^
  --combined-out "%COMBINED_OUT%" || goto :error

echo [完成] B站数据爬取与清洗流程已完成。
exit /b 0

:error
echo [错误] 脚本执行失败，请检查依赖与网络连接。
exit /b 1
