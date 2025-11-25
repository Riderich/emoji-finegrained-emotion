import os
from pathlib import Path
import glob
# 中文说明：移除命令行参数依赖，采用代码内常量实现“一键运行”
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from torch.optim import AdamW
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from modelscope.hub.snapshot_download import snapshot_download

# 相对导入：与项目结构保持一致
from src.train.contrastive.dataset import TextEmojiDataset
from src.train.contrastive.model import TextEmojiContrastive
from src.train.contrastive.train_loop import (
    build_emoji_prototypes,
    train_one_epoch_retrieval,
    eval_topk_accuracy,
)


class Args:
    """
    中文说明：将训练所需的所有配置改为代码内常量，避免命令行参数。
    约定当前工作目录为项目根目录：`emoji-finegrained-emotion/`。
    路径均使用项目相对路径，确保在Windows下可直接运行：`python -m src.train.contrastive.run_pretrain`。
    """
    # 训练用的CSV文件列表（可选）。留None表示按目录扫描
    csv = [r'/data6/zwang/emoji-finegrained-emotion/data/vendor/crawl/cleaned_combined.csv']  # 单一整合CSV
    # CSV目录（默认扫描 cleaned 目录） —— 注意使用项目相对路径（无项目名前缀）
    csv_dir = r'data\vendor\crawl\cleaned'
    # 额外的合并CSV（可选）
    extra_combined = r'data\vendor\crawl\combined_emoji_mapped_more3.csv'
    # 图片缓存目录（本数据集当前不启用URL下载，仍保留目录以便缺失报告）
    cache_dir = r'data\vendor\emote_cache'
    # 本地emoji映射JSON（name/emoji → local_path），已兼容 `local_path` 字段
    emoji_map = r'data\vendor\bilibili_emojiall_map.json'
    # 图片-表情名映射CSV（name → local_path，优先使用该CSV），来自抓取脚本的输出
    name_image_map = r'data\vendor\bilibili_image_name_map.csv'
    # 数据集行为控制
    prefer_local_emoji = True       # 优先使用本地emoji图片
    local_only = True               # 严格只用本地图片，缺失样本直接丢弃
    text_sentinel = False           # 默认保留表情名文本信号（建议），需要匿名时再启用哨兵
    sentinel_token = '[EMOJI]'
    text_field = 'sentence'         # 使用 `sentence` 字段；如需用 `message` 可改为 'message'
    # 训练参数
    epochs = 70                        # 中文说明：默认训练70轮，便于在50/70做策略决策
    batch_size = 160                   # 默认批大小提升，若显存不足可下调
    num_workers = 12                   # 提升并发加载线程（根据IO与CPU调整）
    resume = False
    val_ratio = 0.12                   # 中文说明：验证集比例≈12%，与最优trial相近
    save_dir = r'checkpoints'
    use_amp = True                     # 启用AMP混合精度以节省显存、加速训练
    # 文本模型来源：'modelscope' 或 'huggingface'
    text_model_source = 'modelscope'
    # 中文说明：修正为有效的ModelScope模型ID（中文BERT Base骨干）
    modelscope_id = 'damo/nlp_bert_backbone_base_std'
    hf_model_name = 'bert-base-chinese'
    # Optuna 调参配置
    tune = True                       # 是否启用调参流程；默认False，开启后运行study.optimize
    n_trials = 30                      # 调参次数提升，扩大探索范围
    trial_epochs = 7                   # 每个trial训练轮数略增，提升评估稳定性
    # 学习率与投影维度（固定配置，避免命令行参数）
    text_lr = 2e-5
    img_lr = 1e-4
    proj_lr = 1e-3
    # 中文说明：温度参数（log_tau）使用更高学习率以加快自适应
    temp_lr = 3e-3
    proj_dim = 512
    # 冻结轮数（检索任务按需取消冻结阶段，直接端到端训练）
    warmup_freeze_epochs = 0  # 中文说明：取消冻结阶段，端到端训练
    lr_eta_min = 0.0
    # 中文说明：按步（batch）预热比例（线性 warmup），建议 5%~10%
    warmup_ratio = 0.1
    # 温度初值与梯度调试
    init_temp = 0.05                   # 中文说明：固定温度τ=0.05，避免小数据下不稳定学习
    debug_grad = False


def collect_csvs(args):
    # 中文说明：统一路径解析，避免 Windows 下出现 "emoji-finegrained-emotion\emoji-finegrained-emotion" 重复前缀
    def resolve_path(p: str) -> str:
        """将传入路径规范化为当前操作系统的分隔符，并相对项目根解析为绝对路径。

        中文说明：兼容 Windows 和 Linux 输入的分隔符（'/' 与 '\\'），避免在 Linux 下
        出现 '\\' 被当作普通字符导致路径失效的问题。同时容忍误带项目名前缀。
        """
        # 计算项目根目录：从当前文件向上三级（src/train/contrastive → 项目根）
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if not p:
            return ''
        # 若为绝对路径，直接做规范化并返回
        if os.path.isabs(p):
            return os.path.normpath(p.replace('\\', os.sep).replace('/', os.sep))
        # 统一分隔符到当前操作系统的分隔符
        norm = p.replace('\\', os.sep).replace('/', os.sep)
        # 若误带项目名前缀，去除一次（兼容 bat 脚本/手工输入）
        prefix = f"emoji-finegrained-emotion{os.sep}"
        if norm.lower().startswith(prefix):
            norm = norm[len(prefix):]
        # 最终与项目根目录拼出绝对路径，并做进一步规范化
        full = os.path.join(project_root, norm)
        return os.path.normpath(full)

    # 收集CSV：优先使用显式列表；否则扫描目录，再附加合并CSV
    csvs = []
    if args.csv:
        # 中文说明：显式列表中的路径逐一规范化
        for p in args.csv:
            rp = resolve_path(p)
            csvs.append(rp)
    else:
        scan_dir = resolve_path(args.csv_dir)
        if os.path.isdir(scan_dir):
            csvs.extend(glob.glob(os.path.join(scan_dir, '*.csv')))
    extra_path = resolve_path(args.extra_combined)
    if extra_path and os.path.exists(extra_path):
        csvs.append(extra_path)

    # 去重与存在性过滤，并打印可用列表
    seen = set()
    result = []
    for p in csvs:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            result.append(p)
    print(f"[info] 解析后的CSV目录：{os.path.normpath(resolve_path(args.csv_dir))}")
    print(f"[info] 找到 {len(result)} 个CSV：\n  " + "\n  ".join(result))
    return result


def save_checkpoint(path, model, optimizer, epoch, best_metric):
    # 保存检查点：包含模型参数、优化器状态、当前epoch与最佳指标
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric
    }, path)


def load_checkpoint(path, model, optimizer):
    # 加载检查点：恢复模型与优化器状态
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt.get('model', {}))
    optimizer.load_state_dict(ckpt.get('optimizer', {}))
    epoch = ckpt.get('epoch', 0)
    best_metric = ckpt.get('best_metric', 0.0)
    return epoch, best_metric


def main():
    # 中文说明：一键运行版本。配置源自 Args 常量，不依赖命令行参数。
    args = Args()
    # 中文说明：支持通过环境变量覆盖训练数据路径与映射文件，便于服务器直接指定绝对路径
    # 1) EMOJI_CSV：指定单个“整合的完整CSV”（例如 cleaned_combined.csv）
    # 2) EMOJI_CSV_DIR：指定CSV目录（将扫描该目录下所有 .csv）
    # 3) EMOJI_MAP_JSON：指定本地 emoji 映射 JSON（name/emoji → local_path）
    # 4) EMOJI_NAME_IMAGE_MAP：指定 name → local_path 的映射 CSV
    env_csv = os.getenv('EMOJI_CSV', '').strip()
    env_csv_dir = os.getenv('EMOJI_CSV_DIR', '').strip()
    if env_csv:
        # 中文说明：若指定单一整合CSV，优先使用该文件，并关闭 extra_combined 以避免重复
        if os.path.exists(env_csv):
            args.csv = [env_csv]                # 仅读取该CSV文件
            args.extra_combined = ''            # 关闭额外合并CSV防止重复读取
            print(f"[info] 使用环境变量 EMOJI_CSV 指定的整合CSV：{env_csv}")
        else:
            print(f"[warn] EMOJI_CSV 指定的文件不存在：{env_csv}")
    elif env_csv_dir:
        # 中文说明：若指定目录，则扫描该目录下所有CSV
        if os.path.isdir(env_csv_dir):
            args.csv_dir = env_csv_dir
            print(f"[info] 使用环境变量 EMOJI_CSV_DIR 指定的目录：{env_csv_dir}")
        else:
            print(f"[warn] EMOJI_CSV_DIR 指定的目录不存在：{env_csv_dir}")

    # 中文说明：映射文件的环境变量覆盖（如果提供且存在则使用）
    env_map_json = os.getenv('EMOJI_MAP_JSON', '').strip()
    env_name_img_map = os.getenv('EMOJI_NAME_IMAGE_MAP', '').strip()
    if env_map_json and os.path.exists(env_map_json):
        args.emoji_map = env_map_json
        print(f"[info] 使用环境变量 EMOJI_MAP_JSON：{env_map_json}")
    if env_name_img_map and os.path.exists(env_name_img_map):
        args.name_image_map = env_name_img_map
        print(f"[info] 使用环境变量 EMOJI_NAME_IMAGE_MAP：{env_name_img_map}")
    warmup_freeze_epochs = args.warmup_freeze_epochs
    eta_min = args.lr_eta_min
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 中文说明：AMP仅在CUDA设备可用时启用，避免CPU环境下出错
    enable_amp = args.use_amp and (device.type == 'cuda')
    # 中文说明：启用cudnn benchmark优化卷积选择；对稳定输入尺寸场景有加速
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            # 在PyTorch>=2.0上可进一步提升GEMM精度与速度（对BERT前向也有益）
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

    # 文本tokenizer：优先使用 ModelScope 快照目录，避免访问 Hugging Face
    # 说明：选择 'modelscope' 时，通过 snapshot_download 获取本地目录并 local_files_only 加载
    if args.text_model_source == 'modelscope':
        # 中文说明：将 ModelScope 缓存目录重定位到项目 data/modelscope_cache
        # 修正：从当前文件回溯四级到项目根（contrastive→train→src→项目根）
        repo_root = Path(__file__).resolve().parents[4]
        ms_cache_dir = repo_root / 'data' / 'modelscope_cache'
        os.makedirs(ms_cache_dir, exist_ok=True)
        os.environ.setdefault('MODELSCOPE_CACHE', str(ms_cache_dir))
        try:
            # 中文说明：首先尝试主ID（damo），失败则尝试备用ID（iic）
            text_model_dir = snapshot_download(args.modelscope_id, cache_dir=str(ms_cache_dir))
        except Exception as e1:
            print(f"[warn] ModelScope 主ID下载失败，尝试备用ID。错误: {e1}")
            try:
                text_model_dir = snapshot_download('iic/nlp_bert_backbone_base_std', cache_dir=str(ms_cache_dir))
            except Exception as e2:
                # 最后回退：尝试使用本地HF缓存的模型名（不会联网）
                print(f"[error] ModelScope 备用ID也失败，回退到本地HF缓存。错误: {e2}")
                text_model_dir = args.hf_model_name
        # 严格仅从本地加载，防止无意联网
        tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)

    # 收集CSV并构造数据集
    csvs = collect_csvs(args)
    if not csvs:
        print('[warn] 未找到可用CSV，训练将提前结束。请检查数据路径：')
        print(f"       csv-dir={args.csv_dir}\n       extra-combined={args.extra_combined}")
        return

    ds = TextEmojiDataset(
        csv_paths=csvs,
        image_cache_dir=args.cache_dir,
        text_tokenizer=tokenizer,
        max_len=96,
        emoji_map_json=args.emoji_map,
        name_image_map_csv=args.name_image_map,
        prefer_local=args.prefer_local_emoji,
        local_only=args.local_only,
        use_sentinel=args.text_sentinel,
        sentinel_token=args.sentinel_token,
        text_field=args.text_field
    )

    # 中文说明：打印数据集统计，确认本地图片映射是否生效
    try:
        stats = ds.stats
        print(f"[info] 数据集统计：总行={stats['total_rows']} 保留={stats['kept_rows']} 丢弃={stats['dropped_rows']} 本地={stats['local_image_rows']} 远程={stats['remote_image_rows']} 缺失本地图片={stats.get('missing_rows', 0)}")
        # 中文说明：补充数据加载体检：
        # 1) 映射规模（name/emoji键数量）；2) 唯一图片数与重复率；3) 文本样例与长度分布近似
        try:
            name_keys = len(getattr(ds, 'emoji_map_by_name', {}) or {})
            emoji_keys = len(getattr(ds, 'emoji_map_by_emoji', {}) or {})
            print(f"[info] 映射规模：by_name={name_keys} by_emoji={emoji_keys}")
        except Exception:
            pass
        try:
            from collections import Counter
            paths = [r['image_path'] for r in ds.rows]
            c = Counter(paths)
            unique_images = len(c)
            dup_images = sum(1 for _, v in c.items() if v > 1)
            dup_ratio = (sum(v - 1 for v in c.values()) / max(1, len(paths)))
            print(f"[info] 图片唯一数={unique_images} 重复图片数={dup_images} 重复样本比例={dup_ratio:.4f}")
        except Exception:
            pass
        try:
            # 打印前5条文本样例（截断展示），确认哨兵/表情名处理是否符合预期
            sample_n = min(5, len(ds.rows))
            for i in range(sample_n):
                t = ds.rows[i]['text']
                p = ds.rows[i]['image_path']
                print(f"[sample#{i}] text='{t[:80]}' | image_path='{p}' | exists={os.path.exists(p)}")
        except Exception:
            pass
    except Exception:
        pass

    # 中文说明：若存在缺失的本地图片，生成报告文件，便于用户自行下载补齐
    try:
        if getattr(ds, 'missing', None) and len(ds.missing) > 0:
            os.makedirs(args.save_dir, exist_ok=True)
            report_path = os.path.join(args.save_dir, 'missing_local_images.csv')
            import csv
            with open(report_path, 'w', encoding='utf-8', newline='') as rf:
                writer = csv.DictWriter(rf, fieldnames=['emoji_alt', 'emoji_name', 'sentence', 'suggested_url'])
                writer.writeheader()
                for rec in ds.missing:
                    writer.writerow(rec)
            print(f"[warn] 检测到 {len(ds.missing)} 条样本缺少本地图片，已生成报告：{report_path}")
    except Exception:
        pass

    # 训练/验证集划分：按比例拆分
    # 中文说明：固定随机种子以确保在调参与常规训练时划分一致，便于比较
    val_len = int(len(ds) * args.val_ratio)
    train_len = len(ds) - val_len
    rng = torch.Generator().manual_seed(42)
    if val_len > 0 and train_len > 0:
        train_ds, val_ds = random_split(ds, [train_len, val_len], generator=rng)
    else:
        train_ds, val_ds = ds, None

    # DataLoader
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False  # 多worker下保持线程常驻减少创建开销
    )
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
    else:
        val_dl = None

    # 中文说明：快速检查一个batch的形状与取值范围，排除数据管道错误
    try:
        b = next(iter(train_dl))
        ids = b['input_ids']
        mask = b['attention_mask']
        img = b['image']
        print(f"[sanity] batch shapes: input_ids={tuple(ids.shape)} attention_mask={tuple(mask.shape)} image={tuple(img.shape)}")
        # 简要统计：文本长度均值/最大值；图片像素均值/方差（已归一化后）
        try:
            lens = mask.sum(dim=1).float()
            print(f"[sanity] text length: mean={lens.mean().item():.1f} max={lens.max().item():.0f}")
        except Exception:
            pass
        try:
            print(f"[sanity] image norm stats: mean={img.mean().item():.4f} std={img.std(unbiased=False).item():.4f}")
        except Exception:
            pass
    except Exception:
        pass

    # 模型
    model = TextEmojiContrastive(proj_dim=args.proj_dim, init_tau=args.init_temp).to(device)  # 中文说明：投影维度=512；温度初始固定为0.05
    # ===== 固定温度τ：不参与训练，避免学习失控 =====
    import math
    try:
        model.log_tau.data = torch.tensor(math.log(0.05), dtype=torch.float32, device=model.log_tau.device)
    except Exception:
        model.log_tau.data = torch.tensor(math.log(0.05), dtype=torch.float32)
    model.log_tau.requires_grad = False  # 中文说明：冻结温度参数，移出优化器
    # 中文说明：冷启动阶段冻结编码器，先训练投影头以稳定对齐；后续再解冻
    # 冻结轮数改为可配置：warmup_freeze_epochs（默认1，可通过CLI覆盖）

    # 参数分组：不同模块不同学习率（已移除 log_tau 参数组）
    param_groups = [
        {'params': model.text_encoder.parameters(), 'lr': args.text_lr, 'weight_decay': 0.01},   # 文本编码器
        {'params': model.img_encoder.parameters(),  'lr': args.img_lr,  'weight_decay': 0.01},   # 图片编码器
        {'params': list(model.text_proj.parameters()) + list(model.img_proj.parameters()),
         'lr': args.proj_lr, 'weight_decay': 0.0}                                                # 投影头
    ]
    optimizer = AdamW(param_groups)
    # 中文说明：改用“按步”的顺序调度器：线性 warmup（按 warmup_ratio）→ 余弦退火
    # 计算总步数（近似为每轮的batch数乘以轮数）
    import math
    steps_per_epoch = max(1, len(train_dl))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)  # 中文说明：线性从极低LR升至基准LR
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, int((total_steps - warmup_steps) * 1.5)),  # 中文说明：扩展余弦总步数×1.5，避免前期LR过快衰减
        eta_min=eta_min
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # 检查点路径
    last_path = os.path.join(args.save_dir, 'pretrain_last.pt')
    best_path = os.path.join(args.save_dir, 'pretrain_best.pt')

    # 断点续训：若启用且存在文件，则加载恢复
    start_epoch = 0
    best_metric = 0.0
    if args.resume and os.path.exists(last_path):
        start_epoch, best_metric = load_checkpoint(last_path, model, optimizer)
        print(f"[info] 从检查点恢复：epoch={start_epoch}, best_metric={best_metric:.4f}")

    # 如果启用Optuna调参，则运行study并根据最佳trial进行一次完整训练；否则直接常规训练
    if args.tune:
        def objective(trial: optuna.Trial):
            # 中文说明：每个trial采样一组超参数，并在较少轮次下训练评估 Top-1/Top-5
            # 采样空间可根据A6000与数据规模调整
            # 中文说明：扩展搜索空间（批大小、学习率、权重衰减、投影维度、验证比例），更充分探索
            bsz = trial.suggest_categorical('batch_size', [128, 160, 192, 224, 256])
            n_workers = trial.suggest_categorical('num_workers', [8, 12, 16])
            text_lr = trial.suggest_float('text_lr', 5e-6, 2e-4, log=True)
            img_lr = trial.suggest_float('img_lr', 1e-5, 1e-3, log=True)
            proj_lr = trial.suggest_float('proj_lr', 1e-4, 1e-2, log=True)
            enc_wd = trial.suggest_float('enc_weight_decay', 1e-5, 5e-2, log=True)
            head_wd = trial.suggest_float('head_weight_decay', 1e-10, 1e-2, log=True)
            proj_dim = trial.suggest_categorical('proj_dim', [256, 512, 768])
            w_ep = trial.suggest_categorical('warmup_freeze_epochs', [0, 1, 2])
            val_r = trial.suggest_float('val_ratio', 0.1, 0.3)

            # 重新划分数据集（保持同一随机种子，便于trial间公平对比）
            v_len = int(len(ds) * val_r)
            t_len = len(ds) - v_len
            rng2 = torch.Generator().manual_seed(42)
            if v_len > 0 and t_len > 0:
                t_ds, v_ds = random_split(ds, [t_len, v_len], generator=rng2)
            else:
                t_ds, v_ds = ds, None

            # 调参阶段关闭persistent_workers，避免多trial线程常驻导致资源占用
            t_dl = DataLoader(
                t_ds, batch_size=bsz, shuffle=True, num_workers=n_workers,
                pin_memory=True, persistent_workers=False
            )
            v_dl = None
            if v_ds is not None:
                v_dl = DataLoader(
                    v_ds, batch_size=bsz, shuffle=False, num_workers=n_workers,
                    pin_memory=True, persistent_workers=False
                )

            # 为每个trial新建模型与优化器
            # 中文说明：按trial采样的投影维度构建模型
            m = TextEmojiContrastive(proj_dim=proj_dim, init_tau=args.init_temp).to(device)
            # ===== 冻结温度τ（trial阶段也不学习）=====
            try:
                m.log_tau.data = torch.tensor(math.log(0.05), dtype=torch.float32, device=m.log_tau.device)
            except Exception:
                m.log_tau.data = torch.tensor(math.log(0.05), dtype=torch.float32)
            m.log_tau.requires_grad = False
            pg = [
                {'params': m.text_encoder.parameters(), 'lr': text_lr, 'weight_decay': enc_wd},
                {'params': m.img_encoder.parameters(), 'lr': img_lr, 'weight_decay': enc_wd},
                {'params': list(m.text_proj.parameters()) + list(m.img_proj.parameters()),
                 'lr': proj_lr, 'weight_decay': head_wd}
            ]
            opt = AdamW(pg)
            # 中文说明：trial 阶段同样采用按步 warmup+cosine 调度器，确保与主训练一致
            steps_per_epoch_trial = max(1, len(t_dl))
            total_steps_trial = steps_per_epoch_trial * args.trial_epochs
            warmup_steps_trial = max(1, int(total_steps_trial * args.warmup_ratio))
            from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
            warmup_t = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps_trial)
            cosine_t = CosineAnnealingLR(
                opt,
                T_max=max(1, int((total_steps_trial - warmup_steps_trial) * 1.5)),  # 中文说明：扩展余弦总步数×1.5
                eta_min=eta_min
            )
            trial_scheduler = SequentialLR(opt, schedulers=[warmup_t, cosine_t], milestones=[warmup_steps_trial])

            best_acc = 0.0  # 追踪 Top-1 最优
            # 仅训练较少轮次，加速搜索
            for ep in range(args.trial_epochs):
                # 中文说明：检索任务取消冻结阶段，直接端到端训练
                for p in m.text_encoder.parameters():
                    p.requires_grad = True
                for p in m.img_encoder.parameters():
                    p.requires_grad = True
                try:
                    # 中文说明：每轮先构建一次全局原型库（使用数据集唯一图片 → 向量）
                    prototypes = build_emoji_prototypes(m, ds, device, use_amp=enable_amp)
                    # 中文说明：按步调度器在训练函数内部每个batch推进，无需此处调用 step()
                    loss_ep = train_one_epoch_retrieval(
                        m, t_dl, opt, device, prototypes,
                        use_amp=enable_amp, scheduler=trial_scheduler,
                        debug_grad=args.debug_grad,
                        mix_weights=(1.0, 0.3), tri_margin=0.2, top_k=10
                    )
                    # 评估 Top-1/Top-5
                    metrics = eval_topk_accuracy(
                        m, v_dl if v_dl is not None else t_dl, device, prototypes,
                        use_amp=enable_amp, top_k=(1, 5)
                    )
                    acc_ep = float(metrics.get('top1', 0.0))
                except RuntimeError as e:
                    # 中文说明：捕获OOM等异常，清空显存并剪枝该trial，避免整个study中断
                    if 'out of memory' in str(e).lower():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        raise optuna.TrialPruned()
                    else:
                        raise
                best_acc = max(best_acc, acc_ep)
                # 可选：中位数剪枝，加速调参
                trial.report(best_acc, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            # 释放显存（特别是在GPU上多trial时有益）
            try:
                del m, opt, t_dl, v_dl
                torch.cuda.empty_cache()
            except Exception:
                pass
            return best_acc

        # 创建study并运行
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        )
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
        print(f"[optuna] 最优trial: value={study.best_value:.4f} params={study.best_params}")

        # 使用最佳超参数进行一次完整训练保存权重
        best = study.best_params
        # 重新构建DataLoader与优化器（持久worker重新开启）
        train_dl = DataLoader(
            train_ds,
            batch_size=best['batch_size'],
            shuffle=True,
            num_workers=best['num_workers'],
            pin_memory=True,
            persistent_workers=True if best['num_workers'] > 0 else False
        )
        if val_dl is not None:
            val_dl = DataLoader(
                val_ds,
                batch_size=best['batch_size'],
                shuffle=False,
                num_workers=best['num_workers'],
                pin_memory=True,
                persistent_workers=True if best['num_workers'] > 0 else False
            )
        # 使用最佳投影维度重建模型（中文说明：让最终训练与trial设定一致）
        if 'proj_dim' in best:
            model = TextEmojiContrastive(proj_dim=best['proj_dim'], init_tau=args.init_temp).to(device)
        # ===== 确保最终训练也冻结温度τ =====
        try:
            model.log_tau.data = torch.tensor(math.log(0.05), dtype=torch.float32, device=model.log_tau.device)
        except Exception:
            model.log_tau.data = torch.tensor(math.log(0.05), dtype=torch.float32)
        model.log_tau.requires_grad = False
        # 根据最佳权重衰减与学习率重建优化器
        enc_wd_best = best.get('enc_weight_decay', 0.01)
        head_wd_best = best.get('head_weight_decay', 0.0)
        # 中文说明：重建优化器参数分组，保持 log_tau 独立参数组（零权重衰减）与较高学习率，避免“静默杀死”
        optimizer = AdamW([
            {'params': model.text_encoder.parameters(), 'lr': best['text_lr'], 'weight_decay': enc_wd_best},
            {'params': model.img_encoder.parameters(),  'lr': best['img_lr'],  'weight_decay': enc_wd_best},
            {'params': list(model.text_proj.parameters()) + list(model.img_proj.parameters()),
             'lr': best['proj_lr'], 'weight_decay': head_wd_best}
        ])
        # 中文说明：重建按步 warmup+cosine 调度器（与最佳超参对应的DataLoader步数）
        steps_per_epoch = max(1, len(train_dl))
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int((total_steps - warmup_steps) * 1.5)),  # 中文说明：扩展余弦总步数×1.5
            eta_min=eta_min
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        # 若trial包含冻结轮数则应用到完整训练
        if 'warmup_freeze_epochs' in best:
            warmup_freeze_epochs = int(best['warmup_freeze_epochs'])

        # 完整训练并保存best/last与最终权重（按 Top-1 衡量）
        last_acc = 0.0  # 中文说明：记录最后一次Top-1，用于70轮决策
        for epoch in range(start_epoch, args.epochs):
            # 中文说明：检索任务取消冻结阶段，始终端到端训练
            for p in model.text_encoder.parameters():
                p.requires_grad = True
            for p in model.img_encoder.parameters():
                p.requires_grad = True
            # 中文说明：打印当前可训练参数规模（解冻后约应为≈20M），用于确认 requires_grad 生效
            try:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[info] 可训练参数: {trainable_params/1e6:.1f}M")
            except Exception:
                pass
            # 中文说明：每轮重建一次全局原型库（注入图片增强多样性）
            prototypes = build_emoji_prototypes(model, ds, device, use_amp=enable_amp)
            # 中文说明：按步调度器在训练函数内部推进，无需此处调用 step()
            loss = train_one_epoch_retrieval(
                model, train_dl, optimizer, device, prototypes,
                use_amp=enable_amp, scheduler=scheduler, debug_grad=args.debug_grad,
                mix_weights=(1.0, 0.3), tri_margin=0.2, top_k=10
            )
            # 评估 Top-1/Top-5
            metrics = eval_topk_accuracy(
                model, val_dl if val_dl is not None else train_dl, device, prototypes,
                use_amp=enable_amp, top_k=(1, 5)
            )
            acc = float(metrics.get('top1', 0.0))
            last_acc = acc  # 中文说明：更新最后一次Top-1
            # 中文说明：打印当前温度（tau）以便观察是否在合理范围（0.01~0.1）并逐步学习
            try:
                current_tau = float(torch.exp(model.log_tau).item())
            except Exception:
                current_tau = float('nan')
            print(f"epoch={epoch} loss={loss:.4f} top1={acc:.4f} top5={metrics.get('top5', 0.0):.4f} tau={current_tau:.4f} lr={[g['lr'] for g in optimizer.param_groups]}")
            # ===== Checkpoint 1：第50轮决策 =====
            if (epoch + 1) == 50:
                if acc > 0.25 and loss < 3.5:
                    print("[checkpoint-50] 指标良好：继续训练到70。")
                elif acc < 0.20 or loss > 4.0:
                    print("[checkpoint-50] 指标低/损失高：提前终止训练，请检查文本增强或温度设置。")
                    break
                else:
                    print("[checkpoint-50] 指标一般：继续到70，但不建议额外续训。")
            save_checkpoint(last_path, model, optimizer, epoch + 1, best_metric)
            if acc >= best_metric:
                best_metric = acc
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_metric)
                print(f"[info] 更新best检查点：Top-1={best_metric:.4f}")

        # ===== Checkpoint 2：第70轮后的决策与可选微调 =====
        final_acc = last_acc
        if final_acc > 0.35:
            more = 20 if final_acc > 0.40 else 10
            print(f"[checkpoint-70] Top-1={final_acc:.4f}，计划额外微调 {more} epoch（监控过拟合）。")
            drop_patience = 0
            for e2 in range(args.epochs, args.epochs + more):
                prototypes = build_emoji_prototypes(model, ds, device, use_amp=enable_amp)
                loss2 = train_one_epoch_retrieval(
                    model, train_dl, optimizer, device, prototypes,
                    use_amp=enable_amp, scheduler=None, debug_grad=args.debug_grad,
                    mix_weights=(1.0, 0.3), tri_margin=0.2, top_k=10
                )
                metrics2 = eval_topk_accuracy(
                    model, val_dl if val_dl is not None else train_dl, device, prototypes,
                    use_amp=enable_amp, top_k=(1, 5)
                )
                acc2 = float(metrics2.get('top1', 0.0))
                print(f"epoch={e2} loss={loss2:.4f} top1={acc2:.4f} top5={metrics2.get('top5', 0.0):.4f} lr={[g['lr'] for g in optimizer.param_groups]}")
                save_checkpoint(last_path, model, optimizer, e2, best_metric)
                if acc2 >= best_metric:
                    best_metric = acc2
                    save_checkpoint(best_path, model, optimizer, e2, best_metric)
                    print(f"[info] 更新best检查点：Top-1={best_metric:.4f}")
                    drop_patience = 0
                else:
                    if (best_metric - acc2) > 0.02:
                        print(f"[early-stop] 验证Top-1较最佳下降>2%：{best_metric:.4f}→{acc2:.4f}，提前停止微调。")
                        break
                    drop_patience += 1
                    if drop_patience >= 2:
                        print("[early-stop] 验证Top-1连续下降，提前停止微调。")
                        break
        elif 0.28 <= final_acc <= 0.35:
            print(f"[checkpoint-70] Top-1={final_acc:.4f}，不建议继续（数据潜力基本耗尽）。")
        else:
            print(f"[checkpoint-70] Top-1={final_acc:.4f}，停止训练。")

        os.makedirs(args.save_dir, exist_ok=True)
        raw_model_path = os.path.join(args.save_dir, 'text_emoji_clip_roberta_efficientnetlite0.pt')
        torch.save(model.state_dict(), raw_model_path)
        print(f"[info] 预训练完成，模型权重已保存：{raw_model_path}")
    else:
        # 常规训练：检索任务（Top-1/Top-5）
        last_acc = 0.0  # 中文说明：记录最后一次Top-1，用于70轮决策
        for epoch in range(start_epoch, args.epochs):
            # 中文说明：每轮重建一次全局原型库，用于CE分类与Triplet排序
            prototypes = build_emoji_prototypes(model, ds, device, use_amp=enable_amp)
            # 中文说明：训练阶段启用AMP（若use_amp=True）；调度器在函数内推进
            loss = train_one_epoch_retrieval(
                model, train_dl, optimizer, device, prototypes,
                use_amp=enable_amp, scheduler=scheduler, debug_grad=args.debug_grad,
                mix_weights=(1.0, 0.3), tri_margin=0.2, top_k=10
            )
            # 评估阶段：Top-1/Top-5
            metrics = eval_topk_accuracy(
                model, val_dl if val_dl is not None else train_dl, device, prototypes,
                use_amp=enable_amp, top_k=(1, 5)
            )
            acc = float(metrics.get('top1', 0.0))
            last_acc = acc  # 中文说明：更新最后一次Top-1
            try:
                current_tau = float(torch.exp(model.log_tau).item())
            except Exception:
                current_tau = float('nan')
            print(f"epoch={epoch} loss={loss:.4f} top1={acc:.4f} top5={metrics.get('top5', 0.0):.4f} tau={current_tau:.4f} lr={[g['lr'] for g in optimizer.param_groups]}")
            # ===== Checkpoint 1：第50轮决策 =====
            if (epoch + 1) == 50:
                if acc > 0.25 and loss < 3.5:
                    print("[checkpoint-50] 指标良好：继续训练到70。")
                elif acc < 0.20 or loss > 4.0:
                    print("[checkpoint-50] 指标低/损失高：提前终止训练，请检查文本增强或温度设置。")
                    break
                else:
                    print("[checkpoint-50] 指标一般：继续到70，但不建议额外续训。")

            # 保存last检查点（每轮保存），避免中断丢失进度
            save_checkpoint(last_path, model, optimizer, epoch + 1, best_metric)

            # 若指标更好则更新best检查点（按Top-1）
            if acc >= best_metric:
                best_metric = acc
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_metric)
                print(f"[info] 更新best检查点：Top-1={best_metric:.4f}")

        # ===== Checkpoint 2：第70轮后的决策与可选微调 =====
        final_acc = last_acc
        if final_acc > 0.35:
            more = 20 if final_acc > 0.40 else 10
            print(f"[checkpoint-70] Top-1={final_acc:.4f}，计划额外微调 {more} epoch（监控过拟合）。")
            drop_patience = 0
            for e2 in range(args.epochs, args.epochs + more):
                prototypes = build_emoji_prototypes(model, ds, device, use_amp=enable_amp)
                loss2 = train_one_epoch_retrieval(
                    model, train_dl, optimizer, device, prototypes,
                    use_amp=enable_amp, scheduler=None, debug_grad=args.debug_grad,
                    mix_weights=(1.0, 0.3), tri_margin=0.2, top_k=10
                )
                metrics2 = eval_topk_accuracy(
                    model, val_dl if val_dl is not None else train_dl, device, prototypes,
                    use_amp=enable_amp, top_k=(1, 5)
                )
                acc2 = float(metrics2.get('top1', 0.0))
                print(f"epoch={e2} loss={loss2:.4f} top1={acc2:.4f} top5={metrics2.get('top5', 0.0):.4f} lr={[g['lr'] for g in optimizer.param_groups]}")
                save_checkpoint(last_path, model, optimizer, e2, best_metric)
                if acc2 >= best_metric:
                    best_metric = acc2
                    save_checkpoint(best_path, model, optimizer, e2, best_metric)
                    print(f"[info] 更新best检查点：Top-1={best_metric:.4f}")
                    drop_patience = 0
                else:
                    if (best_metric - acc2) > 0.02:
                        print(f"[early-stop] 验证Top-1较最佳下降>2%：{best_metric:.4f}→{acc2:.4f}，提前停止微调。")
                        break
                    drop_patience += 1
                    if drop_patience >= 2:
                        print("[early-stop] 验证Top-1连续下降，提前停止微调。")
                        break
        elif 0.28 <= final_acc <= 0.35:
            print(f"[checkpoint-70] Top-1={final_acc:.4f}，不建议继续（数据潜力基本耗尽）。")
        else:
            print(f"[checkpoint-70] Top-1={final_acc:.4f}，停止训练。")

        # 完成后额外保存纯模型权重（便于微调加载）
        os.makedirs(args.save_dir, exist_ok=True)
        raw_model_path = os.path.join(args.save_dir, 'text_emoji_clip_roberta_efficientnetlite0.pt')
        torch.save(model.state_dict(), raw_model_path)
        print(f"[info] 预训练完成，模型权重已保存：{raw_model_path}")


if __name__ == '__main__':
    main()
