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
1) 创建并激活虚拟环境（推荐使用 Conda）
```
conda create -n your_env_name python=3.9
conda activate your_env_name
```
   或使用 venv：
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
- **文本-Emoji配对数据准备：**
  - **原始数据统计与筛选：**
    - 运行 `data/vendor/crawl/emoji_count.py` 脚本，该脚本会：
      - 统计 `cleaned_combined.csv` 中各 Emoji 的数量。
      - 将占比小于1%的 Emoji 类别合并为“others”。
      - 生成饼图可视化 Emoji 分布（图例为英文，Emoji 名称为中文）。
      - 输出完整的 Emoji 计数到 `emoji_counts.csv`。
      - 筛选出计数大于300条的 Emoji 数据，并保存为 `cleaned_combined_count_gt300.csv`。
  - **训练数据文件生成：**
    - 将 `cleaned_combined.csv` 复制到 `data/text_pairs/train_pairs_1.csv` 作为训练文件1。
    - 将 `cleaned_combined_count_gt300.csv` 复制到 `data/text_pairs/train_pairs_2.csv` 作为训练文件2。
  - **训练代码路径更新：** 训练代码（`src/train/contrastive/run_pretrain.py`）已配置为使用 `data/text_pairs/train_pairs_2.csv` 作为默认训练数据源。
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
- **注意：** 文本-Emoji配对数据的最终准备（包括筛选和文件复制）请参考“数据准备指南”中的详细说明。

更多安装细节与故障排查请见 `env_setup.md`。

## 检索训练与评估（Top-1 / Top-5）
- 训练（Windows）：
  - 运行 `scripts\run_pretrain.bat`
  - 脚本会使用预处理后的 `data/text_pairs/train_pairs_2.csv` 数据集，端到端训练文本-图片对齐空间。
  - **核心策略：**
    - **多模态对比学习：** 文本和图像特征在共享语义空间中对齐。
    - **损失函数：** 结合了交叉熵（CE）分类损失和Triplet排序损失。Triplet损失的正样本对使用当前批次中的图像特征，确保梯度回传到图像编码器。CE损失权重相对较高，Triplet权重也已增强。
    - **原型库：** 每轮训练都会重建全局原型库。
    - **温度参数（τ）：** 引入可学习的温度参数，并约束其值在 `0.03` 到 `0.20` 之间，以优化对比学习的性能和稳定性。
    - **文本输入：** 使用 `message` 字段作为文本输入，并进行分词、截断和填充至96个token。
    - **学习率调度：** 采用线性预热（Linear Warmup）结合余弦退火（Cosine Annealing）调度策略。
    - **混合精度训练（AMP）：** 目前已禁用，以避免潜在的梯度NaN问题。
- 训练（Linux）：
  - 赋权后运行：
    - `chmod +x scripts/run_pretrain.sh`
    - `scripts/run_pretrain.sh --env your_conda_env`
  - 说明：Linux 脚本仅负责调用训练入口（`python -m src.train.contrastive.run_pretrain`）；训练超参数与数据路径已在 Python 内置，需调整请直接编辑 `src/train/contrastive/run_pretrain.py` 中的默认参数块（含批大小、轮数、数据目录、AMP与调度等）。
- 训练日志：
  - 打印 `loss`、`Top-1`、`Top-5`、温度 `tau` 与各参数组学习率。
  - 按 `Top-1` 更新 `checkpoints/pretrain_best.pt`；每轮保存 `pretrain_last.pt`。
  - **新增：** 训练结束后会生成详细的 `eval_summary.txt` 评估日志，包含整体指标、每类表现、平均排名、常见混淆及Top-1预测分布。
- 评估（Windows/Linux 通用）：
  - 执行 `python -m src.eval.main`
  - 程序会自动：
    - 使用 ModelScope/HF 本地缓存加载中文BERT分词器；
    - 构建数据集（严格本地图片）；
    - 加载 `pretrain_best.pt`（或 `pretrain_last.pt`）；
    - 构建全局原型库并计算 Top-1 / Top-5。
    - **新增：** 评估阶段采用平衡采样策略，每个表情类别抽取固定数量（例如，35条）的样本进行测试，确保评估结果的公平性和可靠性。
- 注意：若 `data/vendor/crawl/cleaned/` 为空或无本地图片映射，评估会提示无法构建原型库；请先运行数据准备与清洗脚本并确保 `bilibili_emojiall_map.json` 与 `bilibili_image_name_map.csv` 可用。
