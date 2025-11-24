@echo off
REM CSV清洗脚本：筛选列并按用户+内容去重
REM 说明：运行后在 data\vendor\crawl\cleaned 生成清洗后的各文件；可选生成合并文件

python -m src.data.clean_crawled_csv ^
  --src-dir "emoji-finegrained-emotion\data\vendor\crawl\by_bvid" ^
  --out-dir "emoji-finegrained-emotion\data\vendor\crawl\cleaned" ^
  --combined-out "emoji-finegrained-emotion\data\vendor\crawl\cleaned_combined.csv"

IF %ERRORLEVEL% NEQ 0 (
  echo [错误] 清洗执行失败，请检查依赖与源目录。
  exit /b 1
)

echo [完成] 数据清洗已完成。
exit /b 0