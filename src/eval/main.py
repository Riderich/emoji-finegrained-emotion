"""
检索评估入口（Top-1/Top-5）。

中文说明：该评估脚本会在项目数据集上构建全局原型库，并加载训练好的模型
（若存在 `checkpoints/pretrain_best.pt` 或 `pretrain_last.pt`），计算 Top-1/Top-5。
"""

import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

from src.train.contrastive.dataset import TextEmojiDataset
from src.train.contrastive.model import TextEmojiContrastive
from src.train.contrastive.train_loop import build_emoji_prototypes, eval_topk_accuracy


def _resolve_repo_root() -> Path:
    """解析项目根路径。

    中文说明：从当前文件 `src/eval/main.py` 向上两级到 `src/`，再到项目根。
    """
    return Path(__file__).resolve().parents[2]


def _load_tokenizer(modelscope_id: str, hf_fallback: str) -> AutoTokenizer:
    """加载分词器，优先使用 ModelScope 快照（本地）。

    中文说明：严格 local_files_only；若 ModelScope 两次失败，则回退到本地 HF 缓存名。
    """
    repo_root = _resolve_repo_root()
    ms_cache_dir = repo_root / 'data' / 'modelscope_cache'
    os.makedirs(ms_cache_dir, exist_ok=True)
    os.environ.setdefault('MODELSCOPE_CACHE', str(ms_cache_dir))
    try:
        text_model_dir = snapshot_download(modelscope_id, cache_dir=str(ms_cache_dir))
    except Exception as e1:
        print(f"[warn] ModelScope 主ID下载失败，尝试备用ID。错误: {e1}")
        try:
            text_model_dir = snapshot_download('iic/nlp_bert_backbone_base_std', cache_dir=str(ms_cache_dir))
        except Exception as e2:
            print(f"[error] ModelScope 备用ID也失败，回退到本地HF缓存。错误: {e2}")
            text_model_dir = hf_fallback
    return AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)


def _build_dataset(tokenizer: AutoTokenizer) -> TextEmojiDataset:
    """构建用于评估的数据集（本地图片、无URL回退）。

    中文说明：与训练保持一致的路径与行为（prefer_local/local_only/text_field等）。
    """
    repo_root = _resolve_repo_root()
    # 统一Windows路径分隔符（Python会自动兼容）
    csv_dir = repo_root / 'data' / 'vendor' / 'crawl' / 'cleaned'
    extra_combined = repo_root / 'data' / 'vendor' / 'crawl' / 'combined_emoji_mapped_more3.csv'
    emoji_map = repo_root / 'data' / 'vendor' / 'bilibili_emojiall_map.json'
    name_image_map = repo_root / 'data' / 'vendor' / 'bilibili_image_name_map.csv'
    cache_dir = repo_root / 'data' / 'vendor' / 'emote_cache'
    # 收集CSV：目录+额外文件（存在时）
    csvs = []
    if csv_dir.is_dir():
        import glob
        csvs.extend(glob.glob(str(csv_dir / '*.csv')))  # 目录下所有CSV
    if extra_combined.exists():
        csvs.append(str(extra_combined))
    if not csvs:
        print('[warn] 评估数据未找到任何CSV，请先准备数据。')
    ds = TextEmojiDataset(
        csv_paths=csvs,
        image_cache_dir=str(cache_dir),
        text_tokenizer=tokenizer,
        max_len=96,
        emoji_map_json=str(emoji_map),
        name_image_map_csv=str(name_image_map),
        prefer_local=True,
        local_only=True,
        use_sentinel=False,
        sentinel_token='[EMOJI]',
        text_field='sentence'
    )
    return ds


def _load_model_and_ckpt(device: torch.device) -> TextEmojiContrastive:
    """创建模型并加载检查点（若存在）。

    中文说明：优先加载 `pretrain_best.pt`，否则 `pretrain_last.pt`；若均不存在则随机初始化。
    """
    repo_root = _resolve_repo_root()
    model = TextEmojiContrastive(proj_dim=512, init_tau=0.07).to(device)
    ckpt_best = repo_root / 'checkpoints' / 'pretrain_best.pt'
    ckpt_last = repo_root / 'checkpoints' / 'pretrain_last.pt'
    ckpt_path = None
    if ckpt_best.exists():
        ckpt_path = ckpt_best
    elif ckpt_last.exists():
        ckpt_path = ckpt_last
    if ckpt_path is not None:
        try:
            print(f"[info] 加载检查点：{ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            state = ckpt.get('model', None)
            if state is None:
                # 兼容直接保存的state_dict
                state = ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[warn] 加载state_dict忽略键：missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[warn] 检查点加载失败，改用随机初始化。错误: {e}")
    else:
        print("[warn] 未找到检查点，使用随机初始化模型进行评估（指标可能较低）")
    return model


def main() -> None:
    # 设备与AMP配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enable_amp = device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

    # 文本分词器与数据集
    tokenizer = _load_tokenizer('damo/nlp_bert_backbone_base_std', 'bert-base-chinese')
    ds = _build_dataset(tokenizer)
    if len(getattr(ds, 'emoji_paths', [])) == 0:
        print('[error] 评估失败：数据集中未找到任何本地图片，无法构建类别原型。请先运行数据准备脚本。')
        return
    # 评估时DataLoader仅需较小batch（取决于显存）
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 模型与原型库
    model = _load_model_and_ckpt(device)
    prototypes = build_emoji_prototypes(model, ds, device, use_amp=enable_amp)

    # 计算 Top-1 / Top-5
    metrics = eval_topk_accuracy(model, dl, device, prototypes, use_amp=enable_amp, top_k=(1, 5))
    print(f"[Eval] Top-1={metrics.get('top1', 0.0):.4f} Top-5={metrics.get('top5', 0.0):.4f}")


if __name__ == "__main__":
    main()
