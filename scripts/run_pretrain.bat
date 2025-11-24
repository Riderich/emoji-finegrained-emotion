@echo off
REM 自监督预训练入口：包含断点续训与检查点保存
REM 你可以根据需要调整参数（数据路径、批大小、轮数等）

python -m src.train.contrastive.run_pretrain ^
  --csv-dir "emoji-finegrained-emotion\data\vendor\crawl\cleaned" ^
  --extra-combined "" ^
  --cache-dir "emoji-finegrained-emotion\data\vendor\emote_cache" ^
  --emoji-map "emoji-finegrained-emotion\data\vendor\bilibili_emojiall_map.json" ^
  --name-image-map "emoji-finegrained-emotion\data\vendor\bilibili_image_name_map.csv" ^
  --prefer-local-emoji ^
  --local-only ^
  --text-sentinel ^
  --sentinel-token "[EMOJI]" ^
  --text-field "sentence" ^
  --epochs 1 ^
  --batch-size 16 ^
  --num-workers 0 ^
  --resume ^
  --save-dir "emoji-finegrained-emotion\checkpoints"