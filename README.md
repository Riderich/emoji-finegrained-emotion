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

更多安装细节与故障排查请见 `env_setup.md`。