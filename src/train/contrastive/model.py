import os
from pathlib import Path
import torch
import torch.nn as nn
import math  # 中文说明：用于计算温度初值的对数
from transformers import AutoTokenizer, AutoModel
import timm
from modelscope.hub.snapshot_download import snapshot_download


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        # 简单MLP投影：编码器输出 → 对齐空间；并做L2归一化
        # 中文说明：使用 GELU 替代 ReLU，避免 "Dead ReLU" 导致输出为0进而引发 Normalization 梯度爆炸
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        z = self.net(x)
        # 中文说明：增加 eps 防止零向量导致除零异常
        return nn.functional.normalize(z, dim=-1, eps=1e-6)


class MeanPooler(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        # 均值池化：对有效token取均值，稳定且通用
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class TextEmojiContrastive(nn.Module):
    def __init__(self, proj_dim=256, init_tau: float = 0.07):
        super().__init__()
        # 中文说明：将 ModelScope 缓存目录重定位到项目路径，便于管理与迁移
        # 修正：从当前文件回溯四级到项目根（contrastive→train→src→项目根）
        repo_root = Path(__file__).resolve().parents[4]  # 项目根：emoji-finegrained-emotion/
        ms_cache_dir = repo_root / 'data' / 'modelscope_cache'
        os.makedirs(ms_cache_dir, exist_ok=True)
        # 兼容 ModelScope 的环境变量，用于其它代码遵循同一缓存目录
        os.environ.setdefault('MODELSCOPE_CACHE', str(ms_cache_dir))

        # 文本端模型（ModelScope）：优先使用 ModelScope Hub 下载到本地目录，避免访问 Hugging Face
        # 说明：snapshot_download 会将模型缓存到本地（~/.cache/modelscope），返回可用的目录路径
        # 中文说明：修正 ModelScope 模型ID，并加入备用ID
        # 首选：damo/nlp_bert_backbone_base_std（中文BERT Base骨干）
        # 备用：iic/nlp_bert_backbone_base_std（同款模型在 iic 组织下的镜像）
        ms_primary_id = 'damo/nlp_bert_backbone_base_std'
        ms_backup_id = 'iic/nlp_bert_backbone_base_std'
        try:
            # 尝试下载并返回本地快照目录（首次会联网，之后离线加载）
            # 中文说明：显式指定下载缓存目录到项目路径
            text_model_dir = snapshot_download(ms_primary_id, cache_dir=str(ms_cache_dir))
        except Exception as e1:
            print(f"[warn] ModelScope 主ID下载失败，尝试备用ID。错误: {e1}")
            try:
                # 中文说明：备用ID也使用项目内缓存目录
                text_model_dir = snapshot_download(ms_backup_id, cache_dir=str(ms_cache_dir))
            except Exception as e2:
                # 最后回退：尝试本地已有的 HuggingFace 缓存（不会联网）
                print(f"[error] ModelScope 备用ID也失败，回退到本地HF缓存。错误: {e2}")
                text_model_dir = 'bert-base-chinese'  # 若本地已缓存HF模型则仍可用；未缓存会抛错

        # 文本端：从本地目录加载分词器与模型，禁止联网（local_files_only=True）
        # 中文说明：严格仅从本地加载（local_files_only=True），确保在无法联网时也能工作
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
        try:
            self.text_encoder = AutoModel.from_pretrained(text_model_dir, local_files_only=True)
        except Exception as e:
            # 中文说明：若本地严格加载失败（例如未缓存），仅在确有网络的环境下放宽限制
            print(f"[warn] 本地加载文本模型失败，尝试放宽限制（可能联网）。错误: {e}")
            self.text_encoder = AutoModel.from_pretrained(text_model_dir)
        self.text_hidden = self.text_encoder.config.hidden_size   # 768 维
        self.text_pooler = MeanPooler()                           # 使用均值池化替代CLS

        # 图片端：加载 EfficientNet-Lite0（timm），返回全局特征
        # 中文说明：避免 timm 访问 HuggingFace，优先尝试从 ModelScope 下载并离线加载本地权重
        def _find_timm_checkpoint(root_dir: str):
            """在 ModelScope 快照目录中查找可用的 timm 权重文件。

            兼容常见文件名：pytorch_model.bin / *.pth / *.pt；若均未找到则返回 None。
            """
            if not root_dir or not os.path.exists(root_dir):
                return None
            candidates = []
            # 直接常见文件名
            for fn in [
                'pytorch_model.bin',        # HF常见命名（timm权重通常可直接 torch.load）
                'model.pth', 'model.pt'
            ]:
                p = os.path.join(root_dir, fn)
                if os.path.isfile(p):
                    candidates.append(p)
            # 递归查找 *.pth / *.pt（防止权重位于子目录）
            for dirpath, _, filenames in os.walk(root_dir):
                for fn in filenames:
                    low = fn.lower()
                    if low.endswith('.pth') or low.endswith('.pt') or low == 'pytorch_model.bin':
                        candidates.append(os.path.join(dirpath, fn))
            # 选择最可能的一个（按优先级排序）
            pref = ['pytorch_model.bin', '.pth', '.pt']
            def _score(fp: str):
                n = os.path.basename(fp).lower()
                if n == 'pytorch_model.bin':
                    return 3
                if n.endswith('.pth'):
                    return 2
                if n.endswith('.pt'):
                    return 1
                return 0
            candidates = sorted(set(candidates), key=_score, reverse=True)
            return candidates[0] if candidates else None

        # 中文说明：尝试用 ModelScope 拉取 timm/efficientnet_lite0.ra_in1k 的本地快照（缓存到项目内）
        img_checkpoint = None
        env_cp = os.environ.get('EFFICIENTNET_LITE0_CP', '')
        if env_cp and os.path.exists(env_cp):
            img_checkpoint = env_cp
        else:
            try:
                ms_vision_dir = snapshot_download('timm/efficientnet_lite0.ra_in1k', cache_dir=str(ms_cache_dir))
                img_checkpoint = _find_timm_checkpoint(ms_vision_dir)
            except Exception as e:
                print(f"[warn] ModelScope 获取 EfficientNet-Lite0 权重失败，将使用随机初始化。错误: {e}")

        if img_checkpoint and os.path.exists(img_checkpoint):
            # 中文说明：不使用 timm 的 checkpoint_path，改为手动加载并 strict=False
            # 这样可忽略分类器等不匹配键，避免 'Unexpected key(s) in state_dict' 报错。
            self.img_encoder = timm.create_model(
                'efficientnet_lite0', pretrained=False,
                num_classes=0, global_pool='avg'
            )
            try:
                # 加载权重文件到CPU，兼容常见格式
                sd = torch.load(img_checkpoint, map_location='cpu')
                # 兼容嵌套字典：有些文件以 {'state_dict': ...} 或 {'model': ...} 形式保存
                if isinstance(sd, dict):
                    if 'state_dict' in sd:
                        sd = sd['state_dict']
                    elif 'model' in sd:
                        sd = sd['model']
                # 若包含 DataParallel 前缀 'module.'，统一去除
                if isinstance(sd, dict):
                    sd = {k.replace('module.', ''): v for k, v in sd.items()}
                    # 清理分类器相关权重，避免与 num_classes=0 的结构不匹配
                    sd.pop('classifier.weight', None)
                    sd.pop('classifier.bias', None)
                # 严格性放宽：忽略缺失与多余键
                missing, unexpected = self.img_encoder.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    print(f"[warn] 加载EfficientNet权重时忽略不匹配键：missing={len(missing)}, unexpected={len(unexpected)}")
                print(f"[info] 已从本地权重加载 EfficientNet-Lite0: {img_checkpoint}")
            except Exception as e:
                print(f"[warn] 本地EfficientNet权重加载失败，改用随机初始化。错误: {e}")
                self.img_encoder = timm.create_model(
                    'efficientnet_lite0', pretrained=False,
                    num_classes=0, global_pool='avg'
                )
        else:
            # 回退：随机初始化（不触发任何联网行为）
            self.img_encoder = timm.create_model(
                'efficientnet_lite0', pretrained=False,
                num_classes=0, global_pool='avg'
            )
            print("[warn] 未找到本地 EfficientNet-Lite0 权重，使用随机初始化（训练会较慢收敛）。")
        self.img_hidden = self.img_encoder.num_features           # 一般是 1280 维

        # 两端投影到同一对齐空间
        self.text_proj = ProjectionHead(self.text_hidden, proj_dim)
        self.img_proj = ProjectionHead(self.img_hidden, proj_dim)

        # 可学习温度参数；初始化为常用范围 0.05~0.07 的中值（默认0.07）
        # 中文说明：以 log_tau 作为可学习参数，前向时取 exp(log_tau) 保证温度>0
        # 典型对比学习中温度过小会导致梯度不稳定，过大会降低对比度；0.05~0.07通常较稳健
        init_tau = max(1e-6, float(init_tau))  # 保护：确保初值>0
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), dtype=torch.float32))

    def forward(self, batch):
        # 文本前向：获取最后隐状态并做均值池化
        t_out = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        t_feat = self.text_pooler(t_out.last_hidden_state, batch['attention_mask'])
        tz = self.text_proj(t_feat)                                # 文本投影向量

        # 图片前向：EfficientNet-Lite0 已做全局池化
        i_feat = self.img_encoder(batch['image'])
        iz = self.img_proj(i_feat)                                 # 图片投影向量

        tau = torch.exp(self.log_tau)                              # 温度标量（>0）
        return tz, iz, tau, t_feat, i_feat                         # 返回中间特征便于分析

    # ========= 便捷编码接口（检索任务使用） =========
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """仅编码文本，返回对齐空间中的单位向量。

        中文说明：用于“文本驱动表情检索”训练/评估阶段，避免不必要的图片前向。
        """
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_feat = self.text_pooler(t_out.last_hidden_state, attention_mask)
        tz = self.text_proj(t_feat)
        return tz

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """仅编码图片，返回对齐空间中的单位向量。

        中文说明：用于构建 76 类表情的原型库（每类一张图片），训练时作为全局候选集合。
        """
        i_feat = self.img_encoder(images)
        iz = self.img_proj(i_feat)
        return iz
