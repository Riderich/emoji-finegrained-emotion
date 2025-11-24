import os
from pathlib import Path
import glob
import argparse
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
from src.train.contrastive.train_loop import train_one_epoch, eval_retrieval


class Args:
    """
    中文说明：将训练所需的所有配置改为代码内常量，避免命令行参数。
    约定当前工作目录为项目根目录：`emoji-finegrained-emotion/`。
    路径均使用项目相对路径，确保在Windows下可直接运行：`python -m src.train.contrastive.run_pretrain`。
    """
    # 训练用的CSV文件列表（可选）。留None表示按目录扫描
    csv = None
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
    epochs = 40                        # 默认训练轮次提升（可按需再增至60）
    batch_size = 160                   # 默认批大小提升，若显存不足可下调
    num_workers = 12                   # 提升并发加载线程（根据IO与CPU调整）
    resume = False
    val_ratio = 0.1
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
    # 中文说明：支持命令行覆盖默认参数，兼容现有 bat 脚本
    default = Args()
    parser = argparse.ArgumentParser(description='Text-Emoji 对比学习预训练')
    # 数据路径相关参数
    parser.add_argument('--csv', nargs='*', default=None, help='显式CSV列表；不传则按目录扫描')
    parser.add_argument('--csv-dir', default=default.csv_dir, help='CSV目录（相对项目根或绝对路径）')
    parser.add_argument('--extra-combined', default=default.extra_combined, help='附加合并CSV路径')
    parser.add_argument('--cache-dir', default=default.cache_dir, help='图片缓存目录')
    parser.add_argument('--emoji-map', default=default.emoji_map, help='本地emoji映射JSON')
    parser.add_argument('--name-image-map', default=default.name_image_map, help='图片-表情名映射CSV')
    parser.add_argument('--prefer-local-emoji', action='store_true', default=default.prefer_local_emoji, help='优先使用本地emoji图片')
    parser.add_argument('--local-only', action='store_true', default=default.local_only, help='严格只用本地图片')
    parser.add_argument('--text-sentinel', action='store_true', default=default.text_sentinel, help='启用文本哨兵替换')
    parser.add_argument('--sentinel-token', default=default.sentinel_token, help='哨兵token文本')
    parser.add_argument('--text-field', default=default.text_field, help='文本字段名')
    # 训练与运行参数
    parser.add_argument('--epochs', type=int, default=default.epochs, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=default.batch_size, help='批大小')
    parser.add_argument('--num-workers', type=int, default=default.num_workers, help='DataLoader并发数')
    parser.add_argument('--resume', action='store_true', default=default.resume, help='是否断点续训')
    parser.add_argument('--val-ratio', type=float, default=default.val_ratio, help='验证集比例')
    parser.add_argument('--save-dir', default=default.save_dir, help='检查点保存目录')
    parser.add_argument('--use-amp', action='store_true', default=default.use_amp, help='启用混合精度')
    # 学习率与投影维度（新增可配置项）
    parser.add_argument('--text-lr', type=float, default=2e-5, help='文本编码器学习率')
    parser.add_argument('--img-lr', type=float, default=1e-4, help='图片编码器学习率')
    parser.add_argument('--proj-lr', type=float, default=1e-3, help='投影头与温度学习率')
    parser.add_argument('--proj-dim', type=int, default=512, help='文本/图片投影维度')  # 中文说明：默认提升到512维
    # 文本模型来源
    parser.add_argument('--text-model-source', choices=['modelscope', 'huggingface'], default=default.text_model_source, help='文本模型来源')
    parser.add_argument('--modelscope-id', default=default.modelscope_id, help='ModelScope模型ID')
    parser.add_argument('--hf-model-name', default=default.hf_model_name, help='HuggingFace模型名')
    # Optuna 调参
    parser.add_argument('--tune', action='store_true', default=default.tune, help='是否启用Optuna调参')
    parser.add_argument('--n-trials', type=int, default=default.n_trials, help='调参trial次数')
    parser.add_argument('--trial-epochs', type=int, default=default.trial_epochs, help='每个trial训练轮数')

    cli = parser.parse_args()
    # 将命令行参数应用到 args 对象（保留默认作为后备）
    args = default
    args.csv = cli.csv
    args.csv_dir = cli.csv_dir
    args.extra_combined = cli.extra_combined
    args.cache_dir = cli.cache_dir
    args.emoji_map = cli.emoji_map
    args.name_image_map = cli.name_image_map
    args.prefer_local_emoji = cli.prefer_local_emoji
    args.local_only = cli.local_only
    args.text_sentinel = cli.text_sentinel
    args.sentinel_token = cli.sentinel_token
    args.text_field = cli.text_field
    args.epochs = cli.epochs
    args.batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.resume = cli.resume
    args.val_ratio = cli.val_ratio
    args.save_dir = cli.save_dir
    args.use_amp = cli.use_amp
    args.text_model_source = cli.text_model_source
    args.modelscope_id = cli.modelscope_id
    args.hf_model_name = cli.hf_model_name
    args.tune = cli.tune
    args.n_trials = cli.n_trials
    args.trial_epochs = cli.trial_epochs
    # 新增：学习率与投影维度
    args.text_lr = cli.text_lr
    args.img_lr = cli.img_lr
    args.proj_lr = cli.proj_lr
    args.proj_dim = cli.proj_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # 模型
    model = TextEmojiContrastive(proj_dim=args.proj_dim).to(device)  # 中文说明：提升默认投影维度至512
    # 中文说明：冷启动阶段冻结编码器，先训练投影头以稳定对齐；后续再解冻
    warmup_freeze_epochs = 1

    # 参数分组：不同模块不同学习率
    # 中文说明：使用可配置学习率，默认略提速；必要时通过命令行覆盖
    param_groups = [
        {'params': model.text_encoder.parameters(), 'lr': args.text_lr, 'weight_decay': 0.01},   # 文本编码器
        {'params': model.img_encoder.parameters(),  'lr': args.img_lr,  'weight_decay': 0.01},   # 图片编码器
        {'params': list(model.text_proj.parameters()) + list(model.img_proj.parameters()) + [model.log_tau],
         'lr': args.proj_lr, 'weight_decay': 0.0}                                               # 投影头与温度
    ]
    optimizer = AdamW(param_groups)
    # 中文说明：加入余弦退火学习率调度，帮助稳定训练并提升检索性能
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

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
            # 中文说明：每个trial采样一组超参数，并在较少轮次下训练评估R@1
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
            m = TextEmojiContrastive(proj_dim=proj_dim).to(device)
            pg = [
                {'params': m.text_encoder.parameters(), 'lr': text_lr, 'weight_decay': enc_wd},
                {'params': m.img_encoder.parameters(), 'lr': img_lr, 'weight_decay': enc_wd},
                {'params': list(m.text_proj.parameters()) + list(m.img_proj.parameters()) + [m.log_tau],
                 'lr': proj_lr, 'weight_decay': head_wd}
            ]
            opt = AdamW(pg)

            best_acc = 0.0
            # 仅训练较少轮次，加速搜索
            for ep in range(args.trial_epochs):
                # 中文说明：前 w_ep 轮冻结编码器，稳定投影层训练
                enc_trainable = ep >= w_ep
                for p in m.text_encoder.parameters():
                    p.requires_grad = enc_trainable
                for p in m.img_encoder.parameters():
                    p.requires_grad = enc_trainable
                try:
                    loss_ep = train_one_epoch(m, t_dl, opt, device, use_amp=args.use_amp)
                    acc_ep = eval_retrieval(m, v_dl if v_dl is not None else t_dl, device, use_amp=args.use_amp)
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
            model = TextEmojiContrastive(proj_dim=best['proj_dim']).to(device)
        # 根据最佳权重衰减与学习率重建优化器
        enc_wd_best = best.get('enc_weight_decay', 0.01)
        head_wd_best = best.get('head_weight_decay', 0.0)
        optimizer = AdamW([
            {'params': model.text_encoder.parameters(), 'lr': best['text_lr'], 'weight_decay': enc_wd_best},
            {'params': model.img_encoder.parameters(), 'lr': best['img_lr'], 'weight_decay': enc_wd_best},
            {'params': list(model.text_proj.parameters()) + list(model.img_proj.parameters()) + [model.log_tau],
             'lr': best['proj_lr'], 'weight_decay': head_wd_best}
        ])
        # 中文说明：重建学习率调度器，使其绑定新的优化器
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        # 若trial包含冻结轮数则应用到完整训练
        if 'warmup_freeze_epochs' in best:
            warmup_freeze_epochs = int(best['warmup_freeze_epochs'])

        # 完整训练并保存best/last与最终权重
        for epoch in range(start_epoch, args.epochs):
            # 冻结/解冻策略：前 warmup_freeze_epochs 轮冻结编码器参数
            enc_trainable = epoch >= warmup_freeze_epochs
            for p in model.text_encoder.parameters():
                p.requires_grad = enc_trainable
            for p in model.img_encoder.parameters():
                p.requires_grad = enc_trainable
            if epoch == 0:
                print(f"[info] 冻结编码器进行投影头预训练：epochs={warmup_freeze_epochs}")
            if epoch == warmup_freeze_epochs:
                print("[info] 解冻编码器，开始端到端训练")
            loss = train_one_epoch(model, train_dl, optimizer, device, use_amp=args.use_amp)
            scheduler.step()
            acc = eval_retrieval(model, val_dl if val_dl is not None else train_dl, device, use_amp=args.use_amp)
            print(f"epoch={epoch} loss={loss:.4f} R@1={acc:.4f} lr={[g['lr'] for g in optimizer.param_groups]}")
            save_checkpoint(last_path, model, optimizer, epoch + 1, best_metric)
            if acc >= best_metric:
                best_metric = acc
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_metric)
                print(f"[info] 更新best检查点：R@1={best_metric:.4f}")

        os.makedirs(args.save_dir, exist_ok=True)
        raw_model_path = os.path.join(args.save_dir, 'text_emoji_clip_roberta_efficientnetlite0.pt')
        torch.save(model.state_dict(), raw_model_path)
        print(f"[info] 预训练完成，模型权重已保存：{raw_model_path}")
    else:
        # 训练若干epoch，并做简单检索评估（R@1）
        for epoch in range(start_epoch, args.epochs):
            # 中文说明：训练阶段启用AMP（若use_amp=True）
            loss = train_one_epoch(model, train_dl, optimizer, device, use_amp=args.use_amp)
            if val_dl is not None:
                # 评估阶段也可使用autocast减小显存占用
                acc = eval_retrieval(model, val_dl, device, use_amp=args.use_amp)
            else:
                acc = eval_retrieval(model, train_dl, device, use_amp=args.use_amp)  # 若无验证集，用训练集近似评估
            print(f"epoch={epoch} loss={loss:.4f} R@1={acc:.4f}")

            # 保存last检查点（每轮保存），避免中断丢失进度
            save_checkpoint(last_path, model, optimizer, epoch + 1, best_metric)

            # 若指标更好则更新best检查点
            if acc >= best_metric:
                best_metric = acc
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_metric)
                print(f"[info] 更新best检查点：R@1={best_metric:.4f}")

        # 完成后额外保存纯模型权重（便于微调加载）
        os.makedirs(args.save_dir, exist_ok=True)
        raw_model_path = os.path.join(args.save_dir, 'text_emoji_clip_roberta_efficientnetlite0.pt')
        torch.save(model.state_dict(), raw_model_path)
        print(f"[info] 预训练完成，模型权重已保存：{raw_model_path}")


if __name__ == '__main__':
    main()
