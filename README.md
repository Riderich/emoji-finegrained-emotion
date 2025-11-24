# Emoji Fine-Grained Emotion Recognition (CN)

本仓库用于实现“视觉-语义-文本”三模态融合的中文细粒度Emoji情绪识别项目。包含数据目录结构、环境配置、基础脚本与最小代码骨架。

## 目录结构
- `data/emoji_images/` 多平台Emoji图像（原图与增强样本）
- `data/text_pairs/` 文本-Emoji配对数据（CSV/JSON）
- `data/annotations/` 中文评分与标签（ICC与Pearson统计结果）
- `data/semantics/` Emoji语义向量与映射文件
- `src/` 代码目录（数据、模型、训练、评估）
- `scripts/` Windows运行脚本（训练/评估）
- `reports/` 错误分析与可视化产出

## 快速开始（Windows）
1) 创建并激活虚拟环境
```
python -m venv .venv
.\.venv\Scripts\activate
```
2) 安装依赖
```
pip install -r requirements.txt
```
3) 运行训练与评估脚本（占位示例）
```
scripts\run_train.bat
scripts\run_eval.bat
scripts\clean_data.bat
```

## 数据准备指南
- 获取多平台 Emoji 图像：
  - 先用 `OpenMoji` 抓取「表情与情绪」类别的面部 Emoji（作为开放基线）：
    - 运行：`scripts\prepare_data.bat`
    - 会下载图片到 `data/emoji_images/openmoji/` 并统一分辨率为 `72x72`，随后在 `data/emoji_images/openmoji_aug/` 内生成增强样本（旋转、亮度、对比度）。
 - 其他平台（如小红书/微博/微信/Facebook）需按平台许可与来源获取，放置到 `data/emoji_images/<platform>/` 并遵循命名规则。
 
## B站数据爬取（BV获取 + 评论抓取 + 清洗）
- 目的：扩大训练数据规模，优先获取包含 `[表情名]` 的评论样本。
- 一键运行（Windows）：
  - `scripts\crawl_bilibili.bat`
- 一键运行（Linux）：
  - 赋权后运行：
    - `chmod +x emoji-finegrained-emotion/scripts/crawl_bilibili.sh`
    - `emoji-finegrained-emotion/scripts/crawl_bilibili.sh`
- 步骤说明：
  - 步骤1（获取BV，二选一）：
    - 热门榜（更贴近“视频热门度”）：
      - `python -m src.data.fetch_bilibili_bv --mode popular --max-pages 5 --min-reply 50 --out emoji-finegrained-emotion\data\vendor\crawl\seed_bvids.csv`
    - 关键词搜索（聚焦特定话题）：
      - `python -m src.data.fetch_bilibili_bv --mode search --keywords-file emoji-finegrained-emotion\data\vendor\keywords_bilibili.txt --max-pages 5 --order pubdate --out emoji-finegrained-emotion\data\vendor\crawl\seed_bvids.csv`
  - 步骤2（抓取评论）：`python -m src.data.crawl_bilibili --root emoji-finegrained-emotion --bvids-file emoji-finegrained-emotion\data\vendor\crawl\seed_bvids.csv --emoji-map emoji-finegrained-emotion\data\vendor\bilibili_emojiall_map.json --per-bvid-output-dir emoji-finegrained-emotion\data\vendor\crawl\by_bvid`
  - 步骤3（清洗列并合并）：`python -m src.data.clean_crawled_csv --src-dir emoji-finegrained-emotion\data\vendor\crawl\by_bvid --out-dir emoji-finegrained-emotion\data\vendor\crawl\cleaned --combined-out emoji-finegrained-emotion\data\vendor\crawl\cleaned_combined.csv`
- 说明：
  - 支持热门模式：`--mode popular`，可配合 `--min-reply`（按总评论数过滤，保留更具讨论度的视频）。
  - 支持关键词模式：`--mode search`，关键词文件示例 `data/vendor/keywords_bilibili.txt`（UTF-8，一行一个），可自行扩充。
  - 两模式均可传入 `--sessdata <你的SESSDATA>` 提升接口稳定性。
  - 请求间歇默认 `0.8s`（含随机抖动），可按网络状况调整；热门与评论统计查询会较密集，建议保留节流配置。
 
## 数据清洗（CSV）
- 目标：
  - 仅保留列：`bvid, emoji_alt, emoji_name, message, sentence, mid, uname, ctime_iso`
  - 去除由同一用户（优先 `mid`，缺失用 `uname`）重复发布的完全相同内容（优先 `message`，缺失用 `sentence`）
- 运行：
  - `scripts\clean_data.bat`
  - 或自定义：
    - `python -m src.data.clean_crawled_csv --src-dir emoji-finegrained-emotion\data\vendor\crawl\by_bvid --out-dir emoji-finegrained-emotion\data\vendor\crawl\cleaned --combined-out emoji-finegrained-emotion\data\vendor\crawl\cleaned_combined.csv`
- 输出：
  - 清洗后各文件在 `data/vendor/crawl/cleaned/`
  - 可选合并输出 `cleaned_combined.csv`
- 文本-Emoji配对数据：
  - 模板示例位于 `src/data/templates/text_pairs_template.csv`，可按列填写并放入 `data/text_pairs/`。
- 中文评分与标签：
  - 在 `data/annotations/` 中创建评分文件（个体与均值），后续由分析脚本计算 ICC/Pearson 并输出标签修订结果。

更多安装细节与故障排查请见 `env_setup.md`。
