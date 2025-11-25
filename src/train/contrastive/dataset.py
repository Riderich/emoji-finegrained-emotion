import os
import csv
import io
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# 中文说明：引入 ImageNet 均值/方差，用于与 EfficientNet-Lite0 预训练权重对齐
try:
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except Exception:
    # 中文说明：若 timm 版本较旧没有常量，退回到标准 ImageNet 值
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
import os


# 中文说明：统一路径解析函数，兼容 Windows 与 Linux 分隔符，支持相对项目根解析
def resolve_project_path(p: str) -> str:
    """将任意输入路径规范化为当前操作系统可识别的绝对路径。

    - 兼容传入包含 '/' 或 '\\' 的分隔符；统一为 `os.sep` 后再规范化。
    - 若为绝对路径，直接规范化返回；若为相对路径，则相对项目根（repo 根）解析。
    - 容忍误带项目名前缀（如 'emoji-finegrained-emotion\\data...')，自动剥离一次。
    """
    if not p:
        return ''
    # 计算项目根目录：从当前文件向上三级（src/train/contrastive → 项目根）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    # 统一分隔符
    norm = p.replace('\\', os.sep).replace('/', os.sep)
    # 绝对路径直接返回规范化
    if os.path.isabs(norm):
        return os.path.normpath(norm)
    # 处理误带项目名前缀
    prefix = f"emoji-finegrained-emotion{os.sep}"
    if norm.lower().startswith(prefix):
        norm = norm[len(prefix):]
    # 相对项目根解析
    full = os.path.join(project_root, norm)
    return os.path.normpath(full)

# 中文说明：当映射中的本地路径为其他机器的绝对路径时，尝试重定位到当前仓库
# 逻辑：若路径中包含 data/emoji_images 段，则截取该段至末尾，拼接到本仓库根目录
def rebase_repo_image_path(p: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if not p:
        return ''
    norm = p.replace('\\', os.sep).replace('/', os.sep)
    marker1 = f"data{os.sep}emoji_images".lower()
    marker2 = f"emoji-finegrained-emotion{os.sep}data{os.sep}emoji_images".lower()
    low = norm.lower()
    idx = low.find(marker1)
    if idx < 0:
        idx = low.find(marker2)
    if idx >= 0:
        suffix = norm[idx:]
        return os.path.normpath(os.path.join(project_root, suffix))
    return norm


class TextEmojiDataset(Dataset):
    def __init__(self, csv_paths, image_cache_dir, text_tokenizer, max_len=96,
                 emoji_map_json=None, name_image_map_csv=None,
                 prefer_local=True, local_only=False,
                 use_sentinel=True, sentinel_token='[EMOJI]', text_field='sentence'):
        # 初始化数据集：加载多份CSV，构造文本-图片样本对
        # 中文说明：支持本地JSON映射，将CSV中的emoji字段映射到本地图片路径；
        # 当 local_only=True 时，严格只用本地图片，不进行 URL 回退；若本地找不到则记录缺失并丢弃样本。
        self.rows = []  # 中文说明：每行包含{text, image_path, emoji_id(初始化后填充)}

        # 统计信息：便于训练前打印数据质量
        self.stats = {
            'total_rows': 0,           # 总行数（含无效）
            'kept_rows': 0,            # 保留用于训练的样本数
            'local_image_rows': 0,     # 使用本地图片的样本数
            'remote_image_rows': 0,    # 需要远程下载的样本数（本文件已禁用URL逻辑，保持为0用于兼容打印）
            'dropped_rows': 0,         # 被丢弃的样本数（无文本或无图片）
            'missing_rows': 0          # 本地图片缺失的样本数（仅在local_only模式下统计）
        }
        self.local_only = local_only
        self.use_sentinel = use_sentinel            # 是否使用哨兵token模式（将占位符替换为统一标记）
        self.sentinel = sentinel_token               # 哨兵token内容，如：[EMOJI]
        self.text_field = text_field                 # 文本来源字段：'sentence' 或 'message'
        self.missing = []              # 缺失列表：记录未能映射/不存在的本地图片

        # 规范化传入的路径参数：确保为绝对路径且分隔符正确（跨平台一致）
        image_cache_dir = resolve_project_path(image_cache_dir)
        emoji_map_json = resolve_project_path(emoji_map_json) if emoji_map_json else None
        name_image_map_csv = resolve_project_path(name_image_map_csv) if name_image_map_csv else None

        # 加载本地emoji映射：支持按Unicode emoji字符或按平台名称（如"[OK]")两种键查询
        self.emoji_map_by_name = {}
        self.emoji_map_by_emoji = {}
        # 优先读取CSV映射（图片-表情名），以满足“统一照CSV对应关系”的需求
        if name_image_map_csv and os.path.exists(name_image_map_csv):
            try:
                with io.open(name_image_map_csv, 'r', encoding='utf-8') as cf:
                    reader = csv.reader(cf)
                    header = next(reader, None)
                    # 中文说明：期望列顺序 name,img_url,local_path,emoji,codepoints,detail_url,kind
                    for row in reader:
                        if len(row) < 3:
                            continue
                        name = (row[0] or '').strip()
                        local_path = (row[2] or '').strip()
                        emoji = (row[3] or '').strip() if len(row) > 3 else ''
                        # 仅记录存在的本地文件路径，保证训练阶段无需访问网络
                        # 中文说明：统一映射中的本地路径分隔符，并相对项目根解析
                        lp_norm = resolve_project_path(local_path) if local_path else ''
                        # 若当前机器不存在该路径，尝试依据仓库目录进行重定位
                        if lp_norm and not os.path.exists(lp_norm):
                            lp_rebased = rebase_repo_image_path(lp_norm)
                            if os.path.exists(lp_rebased):
                                lp_norm = lp_rebased
                        if name and lp_norm and os.path.exists(lp_norm):
                            self.emoji_map_by_name[name] = lp_norm
                            if emoji:
                                self.emoji_map_by_emoji[emoji] = lp_norm
            except Exception:
                # 若CSV解析异常，继续尝试JSON映射
                pass

        if emoji_map_json and os.path.exists(emoji_map_json):
            try:
                with io.open(emoji_map_json, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                # 中文说明：JSON为列表，每项包含 name / emoji / image_path 等字段
                for item in data:
                    # 兼容多种字段名：image_path / img_path / local_path
                    # 中文说明：抓取脚本生成的映射文件字段名为 local_path，这里统一兼容。
                    img_path = item.get('image_path') or item.get('img_path') or item.get('local_path')
                    name_key = item.get('name')
                    emoji_key = item.get('emoji')
                    # 中文说明：规范化 JSON 中的本地路径，避免 Linux 下反斜杠失效
                    if img_path and isinstance(img_path, str):
                        img_path = resolve_project_path(img_path)
                        # 若当前机器不存在该路径，尝试依据仓库目录进行重定位
                        if img_path and not os.path.exists(img_path):
                            img_rebased = rebase_repo_image_path(img_path)
                            if os.path.exists(img_rebased):
                                img_path = img_rebased
                        if name_key and isinstance(name_key, str):
                            # 若CSV已提供同名的本地路径，则以CSV为准（不覆盖）
                            if name_key not in self.emoji_map_by_name and os.path.exists(img_path):
                                self.emoji_map_by_name[name_key] = img_path
                        if emoji_key and isinstance(emoji_key, str):
                            if emoji_key not in self.emoji_map_by_emoji and os.path.exists(img_path):
                                self.emoji_map_by_emoji[emoji_key] = img_path
            except Exception:
                # 映射加载失败时，不影响后续URL回退逻辑
                pass

        # 读取CSV构造样本
        for p in csv_paths:
            p_resolved = resolve_project_path(p)
            if not os.path.exists(p_resolved):
                # 文件不存在则跳过，保证健壮性
                continue
            with io.open(p_resolved, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # 跳过表头
                for row in reader:
                    self.stats['total_rows'] += 1
                    # 约定字段顺序：
                    # bvid,rpid,emoji_alt,emoji_name,message,sentence,emote_url,mid,uname,ctime_iso
                    if len(row) < 7:
                        self.stats['dropped_rows'] += 1
                        continue  # 行不完整则跳过
                    emoji_alt = (row[2] or '').strip()
                    emoji_name = (row[3] or '').strip()
                    # 中文说明：文本字段可选 message 或 sentence（默认使用 sentence）
                    sentence_raw = (row[5] or '').strip()
                    message_raw = (row[4] or '').strip()
                    text_src = sentence_raw if self.text_field == 'sentence' else message_raw
                    emote_url = (row[6] or '').strip()

                    # 过滤空文本
                    if not text_src:
                        self.stats['dropped_rows'] += 1
                        continue

                    # 根据映射查找本地图片路径（按名称匹配，支持自动加方括号等规范化）
                    local_path = None
                    if prefer_local:
                        # 中文说明：构造候选键集合，优先匹配原始，再尝试加方括号形式
                        candidates = []
                        if emoji_alt:
                            candidates.append(emoji_alt)
                        if emoji_name and emoji_name not in candidates:
                            candidates.append(emoji_name)
                        # 若不带方括号，补充如 "黑洞" → "[黑洞]" 形式
                        def bracketize(s: str) -> str:
                            s = s.strip()
                            if not s:
                                return s
                            if s.startswith('[') and s.endswith(']'):
                                return s
                            return f"[{s}]"
                        for s in list(candidates):
                            bs = bracketize(s)
                            if bs not in candidates:
                                candidates.append(bs)
                        # 执行查找：命中即返回
                        for key in candidates:
                            if key in self.emoji_map_by_name:
                                local_path = self.emoji_map_by_name[key]
                                break

                    # 验证本地路径是否存在，不存在则视为缺失
                    if local_path and not os.path.exists(local_path):
                        local_path = None

                    # 仅本地模式：本地路径缺失则记录并丢弃；本文件不再包含任何URL回退逻辑
                    if self.local_only:
                        if not local_path:
                            self.stats['missing_rows'] += 1
                            self.stats['dropped_rows'] += 1
                            # 记录缺失信息，便于后续生成报告
                            self.missing.append({
                                'emoji_alt': emoji_alt,
                                'emoji_name': emoji_name,
                                'sentence': text_src,
                                'suggested_url': emote_url
                            })
                            continue
                    else:
                        # 非严格模式也不进行URL回退；保持与local_only一致的丢弃策略
                        if not local_path:
                            self.stats['missing_rows'] += 1
                            self.stats['dropped_rows'] += 1
                            self.missing.append({
                                'emoji_alt': emoji_alt,
                                'emoji_name': emoji_name,
                                'sentence': text_src,
                                'suggested_url': emote_url
                            })
                            continue

                    # 记录样本
                    if local_path:
                        # 中文说明：将文本中的表情占位符替换为统一哨兵token，保留位置信号但不泄露具体表情名
                        text_use = text_src
                        if self.use_sentinel:
                            import re
                            # 匹配方括号中的内容，例如：[笑哭]，[doge]
                            text_use = re.sub(r"\[[^\[\]]+\]", self.sentinel, text_use)
                            # 规范化空白，避免出现重复空格
                            text_use = re.sub(r"\s+", " ", text_use).strip()
                        self.rows.append({
                            'text': text_use,           # 哨兵处理后的文本
                            'image_path': local_path    # 本地图片路径
                        })
                        self.stats['kept_rows'] += 1
                        self.stats['local_image_rows'] += 1
                    # 本文件不添加任何包含 emote_url 的行，彻底移除URL下载相关路径

        # 缓存目录仅用于远程下载的图片
        self.cache_dir = image_cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_tokenizer = text_tokenizer

        # 建立 emoji_id 映射：将唯一图片路径映射到类别ID（0..N-1）
        # 中文说明：这是“文本驱动表情检索”的关键，训练时用emoji_id作为CE分类标签。
        from collections import OrderedDict
        unique_ordered = OrderedDict()  # 保持插入顺序，稳定ID分配
        for r in self.rows:
            p = r['image_path']
            if p not in unique_ordered:
                unique_ordered[p] = None
        self.emoji_paths = list(unique_ordered.keys())        # 类别原型的图片路径列表
        self.emoji_id_of_path = {p: i for i, p in enumerate(self.emoji_paths)}
        for r in self.rows:
            r['emoji_id'] = self.emoji_id_of_path[r['image_path']]  # 为每条样本填充类别ID

        # 强化图片增强：提升特征熵，避免表示崩溃
        # 中文说明：采用随机裁剪/翻转/颜色抖动/模糊/灰度扰动等增强；
        # 对于Emoji图像，增强幅度控制在合理范围，尽量不破坏语义，但提高外观多样性。
        self.img_tf = transforms.Compose([
            # 随机裁剪到 224×224，并允许一定缩放与宽高比变化（提升视角多样性）
            transforms.RandomResizedCrop(
                224,
                scale=(0.5, 1.0),          # 中文说明：裁剪尺度范围（保留50%~100%内容）
                ratio=(0.75, 1.33)         # 宽高比扰动，避免特征过于集中
            ),
            # 随机水平翻转（部分emoji翻转语义不变，增强鲁棒性）
            transforms.RandomHorizontalFlip(p=0.5),
            # 颜色抖动（亮度/对比度/饱和度/色相），增强光照与风格变化的适应性
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            # 轻度高斯模糊，模拟拍摄/压缩导致的细节损失
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # 小概率转为灰度，迫使网络降低对颜色的过拟合
            transforms.RandomGrayscale(p=0.05),
            # 转张量与 ImageNet 归一化（对齐 EfficientNet-Lite0 预训练分布）
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
        self.max_len = max_len

        # 文本动态增强设置（B站风格）：在 __getitem__ 中以一定概率应用
        # 中文说明：模拟口语化/打字习惯，如“哈哈哈”“啊呀吧嘛”，提升文本特征的鲁棒性。
        self.text_aug_prob = 0.3  # 30% 概率进行一次增强；可在后续需要时提升

    def __len__(self):
        # 返回样本总数
        return len(self.rows)

    # 已移除所有URL缓存/下载相关方法

    def __getitem__(self, idx):
        # 返回一个样本：文本token与图片张量
        row = self.rows[idx]
        text = row['text']
        # 动态文本增强：按概率对原文本做轻度扰动
        if self.text_aug_prob > 0.0:
            import random
            if random.random() < self.text_aug_prob:
                text = self._augment_text(text)
        # 中文说明：仅本地图片模式；样本在构造阶段已确保存在本地路径
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except Exception:
            # 本地图片损坏时回退为白色占位，防止训练中断
            img = Image.new('RGB', (224, 224), color=(255, 255, 255))

        toks = self.text_tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        img_t = self.img_tf(img)
        return {
            'input_ids': toks['input_ids'].squeeze(0),           # 文本token id
            'attention_mask': toks['attention_mask'].squeeze(0), # 注意力mask
            'image': img_t,                                      # 图片张量（训练检索时通常不会使用）
            'emoji_id': row['emoji_id']                          # 表情类别ID（CE分类标签）
        }

    # ========== 文本增强实现 ==========
    def _augment_text(self, text: str) -> str:
        """对文本进行轻度口语化增强（B站特色）。

        - 随机加入语气词："啊/呀/吧/嘛/哇/啦/呢"；
        - 随机拉长笑/草等字符："哈/草/嗷/嘻/喵"；
        - 小概率重复末尾字符或加入标点 "！/～"。
        """
        import random
        fillers = ["啊", "呀", "吧", "嘛", "哇", "啦", "呢"]
        laugh_chars = ["哈", "草", "嗷", "嘻", "喵"]
        s = text
        # 1) 随机插入语气词（句首或句尾）
        if random.random() < 0.5:
            if random.random() < 0.5:
                s = random.choice(fillers) + s
            else:
                s = s + random.choice(fillers)
        # 2) 随机拉长笑/草等字符（若出现则拉长）
        for ch in laugh_chars:
            if ch in s and random.random() < 0.5:
                n = random.randint(2, 5)
                s = s.replace(ch, ch * n)
        # 3) 小概率重复末尾字符或加入标点
        if len(s) > 0 and random.random() < 0.4:
            end = s[-1]
            if end.isalpha() or end.isdigit():
                s = s + end
            else:
                s = s + random.choice(["！", "～"])
        return s
