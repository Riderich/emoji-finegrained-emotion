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
```

## 数据准备指南
- 获取多平台 Emoji 图像：
  - 先用 `OpenMoji` 抓取「表情与情绪」类别的面部 Emoji（作为开放基线）：
    - 运行：`scripts\prepare_data.bat`
    - 会下载图片到 `data/emoji_images/openmoji/` 并统一分辨率为 `72x72`，随后在 `data/emoji_images/openmoji_aug/` 内生成增强样本（旋转、亮度、对比度）。
  - 其他平台（如小红书/微博/微信/Facebook）需按平台许可与来源获取，放置到 `data/emoji_images/<platform>/` 并遵循命名规则。
- 文本-Emoji配对数据：
  - 模板示例位于 `src/data/templates/text_pairs_template.csv`，可按列填写并放入 `data/text_pairs/`。
- 中文评分与标签：
  - 在 `data/annotations/` 中创建评分文件（个体与均值），后续由分析脚本计算 ICC/Pearson 并输出标签修订结果。

更多安装细节与故障排查请见 `env_setup.md`。