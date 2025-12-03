import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import Image


# ========== 原双向 InfoNCE（保留以兼容旧流程） ==========


def info_nce_loss(tz, iz, tau):
    # 计算双向InfoNCE对比损失：文本→图片、图片→文本
    sim = tz @ iz.t() / tau                      # 相似度矩阵 (B,B)
    labels = torch.arange(tz.size(0), device=tz.device)
    loss_t2i = F.cross_entropy(sim, labels)      # 文本查询 → 图片
    loss_i2t = F.cross_entropy(sim.t(), labels)  # 图片查询 → 文本
    return (loss_t2i + loss_i2t) / 2.0


def train_one_epoch(model, dataloader, optimizer, device, use_amp=False, scheduler=None, debug_grad=False):
    # 单轮训练：遍历数据、计算损失并反传
    # 中文说明：改为“按步（batch）”进行学习率调度，并在前若干步打印LR与梯度，便于诊断warmup效果
    model.train()
    total = 0.0
    scaler = GradScaler(enabled=use_amp)  # 中文说明：仅使用 enabled 参数初始化，避免传入非法的位置参数
    printed_debug = False  # 中文说明：仅在每轮的首个batch打印梯度，避免日志过长
    global_step = 0        # 中文说明：用于控制前若干步的调试打印
    for batch in tqdm(dataloader, desc='train'):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        if use_amp:
            # 混合精度前向与损失计算
            with autocast('cuda', dtype=torch.float16):
                tz, iz, tau, _, _ = model(batch)
                loss = info_nce_loss(tz, iz, tau)
            # 反传与更新使用GradScaler防止溢出
            scaler.scale(loss).backward()
            # 中文说明：调试模式下在 step 前打印梯度的平均绝对值；需先反缩放以得到真实梯度
            if debug_grad and not printed_debug:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        try:
                            pass
                            # print(f"[grad] {name}: {param.grad.abs().mean().item():.6f}")  # 中文说明：应用户要求，注释掉[grad]打印
                        except Exception:
                            pass
                # 中文说明：额外打印温度的梯度与当前值，诊断其是否被权重衰减/数值问题抑制
                try:
                    tau_val = float(torch.exp(model.log_tau).item())
                    tau_grad = model.log_tau.grad
                    tau_grad_mean = float(tau_grad.abs().mean().item()) if tau_grad is not None else 0.0
                    print(f"[tau] tau={tau_val:.5f} |dL/dlog_tau|≈{tau_grad_mean:.8f}")
                except Exception:
                    pass
                printed_debug = True
            scaler.step(optimizer)
            scaler.update()
        else:
            tz, iz, tau, _, _ = model(batch)
            loss = info_nce_loss(tz, iz, tau)
            loss.backward()
            # 中文说明：调试模式下打印一次梯度的平均绝对值，验证梯度是否正常流动
            if debug_grad and not printed_debug:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        try:
                            # 检查梯度是否含有 NaN
                            if torch.isnan(param.grad).any():
                                print(f"[grad-nan] {name} 梯度含NaN!")
                            # print(f"[grad] {name}: {param.grad.abs().mean().item():.6f}")  # 中文说明：应用户要求，注释掉[grad]打印
                        except Exception:
                            pass
                # 中文说明：额外打印温度的梯度与当前值，诊断其是否被权重衰减/数值问题抑制
                try:
                    tau_val = float(torch.exp(model.log_tau).item())
                    tau_grad = model.log_tau.grad
                    tau_grad_mean = float(tau_grad.abs().mean().item()) if tau_grad is not None else 0.0
                    print(f"[tau] tau={tau_val:.5f} |dL/dlog_tau|≈{tau_grad_mean:.8f}")
                except Exception:
                    pass
                printed_debug = True
            
            # 中文说明：增加梯度裁剪，防止BERT微调时梯度爆炸导致NaN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

        # 中文说明：训练步末尾打印损失，观察是否按预期下降（例如 4.38→3.5→2.8）
        try:
            print(f"Step {global_step}: Loss={loss.item():.4f}")
        except Exception:
            pass

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                # 兼容旧版本PyTorch的调度器行为
                pass
        # 中文说明：在前100步打印当前学习率，确认warmup阶段是否线性上升
        if debug_grad and global_step < 100:
            try:
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                print(f"[lr] step={global_step} lr={lrs}")
            except Exception:
                pass
        # 中文说明：在首个调试步额外打印图像特征的“表示坍塌”指标：
        # 1) 余弦相似度矩阵的非对角均值/方差；2) 特征矩阵的秩
        if debug_grad and global_step == 0:
            try:
                # iz 已经是投影后且 L2 归一化的特征
                sim = (iz @ iz.T).detach()
                n = sim.size(0)
                # 取非对角元素
                off_diag = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
                mean_od = float(off_diag.mean().item()) if off_diag.numel() > 0 else float('nan')
                std_od = float(off_diag.std(unbiased=False).item()) if off_diag.numel() > 0 else float('nan')
                # 特征矩阵秩（以数值稳定容差自动估计）
                rank = int(torch.linalg.matrix_rank(iz, tol=None).item())
                print(f"[collapse] off-diag cos mean={mean_od:.4f} std={std_od:.4f} | rank(iz)={rank}")
            except Exception:
                pass
        total += loss.item()
        global_step += 1
    return total / max(1, len(dataloader))


@torch.no_grad()
def eval_retrieval(model, dataloader, device, use_amp=False):
    # 简单检索评估：计算文本→图片的R@1（旧）
    # 中文说明：为兼容旧日志而保留；新评估在 eval_topk_accuracy 中实现。
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc='eval'):
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_amp:
            # 中文说明：显式指定 device_type='cuda'，避免 PyTorch 2.x 中 autocast 缺失参数导致的运行时错误
            with autocast('cuda', dtype=torch.float16):
                tz, iz, tau, _, _ = model(batch)
        else:
            tz, iz, tau, _, _ = model(batch)
        sim = tz @ iz.t() / tau
        pred = sim.argmax(dim=1)
        gt = torch.arange(tz.size(0), device=tz.device)
        correct += (pred == gt).sum().item()
        total += tz.size(0)
    return correct / max(1, total)


# ========== 检索任务：全局原型库 + CE 分类 + Triplet 排序 ==========
def build_emoji_prototypes(model, dataset, device, use_amp=False, mode: str = 'train'):
    """从数据集的唯一图片路径构建原型库（每类一张图片 → 向量）。

    返回：原型张量 `P`，形状 [C, D]，C 为类别数（emoji唯一图片数），D 为投影维度。
    中文说明：每轮可重建一次以注入图像增强的多样性（增强由 dataset.img_tf 控制）。
    """
    model.eval()
    imgs = []
    tf = None
    try:
        tf = dataset.get_image_transform('eval' if mode == 'eval' else 'train')
    except Exception:
        tf = getattr(dataset, 'img_tf_eval', None) if mode == 'eval' else getattr(dataset, 'img_tf_train', None)
        if tf is None:
            tf = getattr(dataset, 'img_tf', None)
    for p in dataset.emoji_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        imgs.append(tf(img))
    P = torch.stack(imgs, dim=0).to(device)  # [C, 3, 224, 224]
    with torch.no_grad():
        if use_amp:
            with autocast('cuda', dtype=torch.float16):
                proto = model.encode_images(P)  # [C, D]
        else:
            proto = model.encode_images(P)
            
    # 中文说明：检查生成的原型是否包含NaN，若有则发出警告（可能是图像编码器梯度爆炸或输入异常）
    if torch.isnan(proto).any():
        print(f"[Warning] build_emoji_prototypes: 生成的原型包含 NaN！ (use_amp={use_amp})")
        # 尝试打印出哪些图片产生了NaN（前5个）
        nan_indices = torch.where(torch.isnan(proto).any(dim=1))[0]
        print(f"  NaN原型索引 (前5个): {nan_indices[:5].tolist()}")
        
    return proto  # 单位向量（投影层已做 L2 归一化）


def ce_triplet_loss(tz: torch.Tensor, prototypes: torch.Tensor, targets: torch.Tensor,
                    tau: torch.Tensor, top_k: int = 10, margin: float = 0.2,
                    w_ce: float = 1.0, w_tri: float = 0.3, debug_print: bool = False,
                    pos_features: torch.Tensor = None, log_basic: bool = False) -> torch.Tensor:
    """混合损失：CrossEntropy 分类 + Triplet 排序。

    - CE：文本向量对所有类别原型做相似度，监督为 emoji_id（76 选 1）。
    - Triplet：在 Top-K 候选中进行半难负样本的排序约束；温度 tau 仅用于 Triplet 的相似度缩放。
    
    参数:
    - pos_features: [B, D] 当前batch的实时图片特征。若提供，Triplet的正样本对将使用 (tz, pos_features)，
      从而允许梯度回传到图片编码器；若为None，则使用 prototypes[targets]（无图片梯度）。
    """
    # 中文说明：标签清洗与规范化 —— 确保 targets 为 1D Long 的类别索引
    # 若上游误传 one-hot 或浮点型标签，这里统一转换，避免 CE/索引运算失效
    try:
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)  # one-hot → 索引
        if targets.dtype != torch.long:
            targets = targets.long()         # 转为 Long 索引类型
        num_classes = prototypes.size(0)
        if (targets < 0).any() or (targets >= num_classes).any():
            print("[label-error] 发现越界类别ID，已clip到合法范围。")
            targets = targets.clamp(0, num_classes - 1)
    except Exception:
        pass  # 清洗过程异常时不阻断训练

    # CE 部分：对所有类别计算相似度并监督分类
    # 注意：logits 计算仍使用 prototypes（作为分类器的权重），因此 CE 仅更新文本端
    logits = tz @ prototypes.t()  # [B, C]
    ce = F.cross_entropy(logits, targets)

    # Triplet 部分：选取 Top-K 错误类别作为负样本（半难负）
    with torch.no_grad():
        # 排除正确类别的分数，选择最高的错误类别作为候选
        masked = logits.clone()
        # 中文说明：FP16 的最小可表示有限值为 -65504，直接写入 -1e9 会在混合精度下溢出。
        # 这里改为使用 -inf（负无穷），softmax 对 -inf 的概率为 0，语义等价且数值安全。
        # 同时确保索引张量与 logits 在同一设备，避免 CPU/GPU 交叉索引导致的错误。
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        masked[batch_idx, targets] = float('-inf')
        # 中文说明：诊断遮罩是否生效（仅在 debug_print=True 时打印一次）
        try:
            if debug_print:
                diag_mean = logits[batch_idx, targets].detach().float().mean().item()
                masked_diag = masked[batch_idx, targets].detach()
                ok_ratio = (~torch.isfinite(masked_diag)).float().mean().item()
                print(f"[mask-check] diag_mean={diag_mean:.4f} masked_ok_ratio={ok_ratio:.2f}")
        except Exception:
            pass
        topk_scores, topk_idx = masked.topk(k=min(top_k, prototypes.size(0)-1), dim=1)
    
    # 正样本特征：优先使用实时计算的 pos_features（带梯度），否则使用原型（无梯度）
    if pos_features is not None:
        pos = pos_features
    else:
        pos = prototypes[targets]                  # [B, D]

    # 取每个样本的一条负样本（Top-K中得分最高的那个）
    neg = prototypes[topk_idx[:, 0]]               # [B, D]
    # 采用余弦相似度的 margin 形式：鼓励 sim(a,p) ≥ sim(a,n) + margin
    # 中文说明：仅 Triplet 使用 tau 进行缩放，以调节“难度”。
    sim_pos = (tz * pos).sum(dim=1) / tau.squeeze()   # [B] (优化：使用 sum(dim=1) 替代 diag，更高效)
    sim_neg = (tz * neg).sum(dim=1) / tau.squeeze()   # [B]
    tri = F.relu(margin + sim_neg - sim_pos).mean()

    total_loss = w_ce * ce + w_tri * tri
    # 中文说明：调试模式下，打印一次必要的诊断信息，验证损失与梯度是否有效：
    # 1) logits 的前 5x5 数值（检查是否恒定或异常）
    # 2) ∂L/∂logits 的平均绝对值（若为0或极小，说明损失对参数不敏感）
    # 3) 正样本/ hardest neg样本的相似度均值，以及 Triplet Loss 数值（判断是否在学习）
    # 4) tz 与 tau 的梯度范数/均值（确保反向传播链路通畅）
    if debug_print:
        try:
            # -- logits 取样打印（前 5 行、前 5 列）
            try:
                lg = logits[:5, :5].detach().float().cpu().numpy()
                print(f"[logits] 前5x5: {lg}", flush=True)  # 中文说明：加 flush=True，确保在tqdm进度条下也立即输出
            except Exception:
                pass
            # -- logits 的梯度（使用 autograd.grad 直接计算）
            grad_logits = torch.autograd.grad(total_loss, logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if grad_logits is not None:
                gl_mean = float(grad_logits.abs().mean().item())
                print(f"[grad|logits] mean_abs={gl_mean:.8f}", flush=True)
            else:
                print("[grad|logits] None（可能logits未参与图或未启用requires_grad）", flush=True)

            # 中文说明：确保 tz 支持求导（encode_text 输出通常带有 requires_grad=True）
            grad_tz = torch.autograd.grad(total_loss, tz, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if grad_tz is not None:
                gn_l2 = float(grad_tz.norm().item())
                gn_mean = float(grad_tz.abs().mean().item())
                print(f"[grad|tz] L2={gn_l2:.6f} mean_abs={gn_mean:.6f} shape={tuple(grad_tz.shape)}", flush=True)
            else:
                print("[grad|tz] None（可能tz未参与图或requires_grad=False）", flush=True)
        except Exception as e:
            print(f"[grad|tz] 计算失败：{e}")
        try:
            # 中文说明：对温度 tau 的梯度（若 learn_tau=True 时尤为关键）；此处仅诊断打印
            grad_tau = torch.autograd.grad(total_loss, tau, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if grad_tau is not None:
                gt_mean = float(grad_tau.abs().mean().item())
                print(f"[grad|tau] mean_abs={gt_mean:.8f} tau={float(tau.item()):.5f}", flush=True)
            else:
                print("[grad|tau] None（tau可能为常量或未参与图）", flush=True)
        except Exception as e:
            print(f"[grad|tau] 计算失败：{e}")
    # 中文说明：核心指标打印（不关注 tau 的梯度）；若启用 log_basic，则每步打印正/负样本相似度与 Triplet Loss。
    if log_basic:
        try:
            pos_sim = sim_pos.detach().float()
            neg_sim = sim_neg.detach().float()
            print(f"pos_sim={float(pos_sim.mean().item()):.3f}, neg_sim={float(neg_sim.mean().item()):.3f}")
            print(f"triplet={float(tri.item()):.4f}")
        except Exception:
            pass
    return total_loss


def train_one_epoch_retrieval(model, dataloader, optimizer, device, prototypes,
                              use_amp=False, scheduler=None, debug_grad=False,
                              mix_weights=(1.0, 0.3), tri_margin=0.2, top_k=10):
    """检索训练：使用全局原型库进行 CE+Triplet 混合训练。

    参数：
    - prototypes: [C, D] 类别原型向量；可每轮重建以注入图像增强
    - mix_weights: (w_ce, w_tri) 两个损失的权重
    - tri_margin: Triplet 的 margin
    - top_k: Triplet 负样本选择的候选数量（半难负采样）
    """
    model.train()
    total = 0.0
    scaler = GradScaler(enabled=use_amp)  # 中文说明：仅使用 enabled 参数初始化，避免传入非法的位置参数
    # 中文说明：仅在每轮的首个batch打印梯度，避免日志过长
    printed_debug = False
    global_step = 0
    
    # 中文说明：[诊断] 检查首个batch的数据完整性
    first_batch_checked = False
    
    for batch in tqdm(dataloader, desc='train(retr)'):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # [诊断] 打印首个batch的数据概况
        if not first_batch_checked:
            try:
                print(f"\n[Data Check] Batch Size: {len(batch['input_ids'])}")
                print(f"[Data Check] input_ids shape: {batch['input_ids'].shape}")
                print(f"[Data Check] input_ids range: min={batch['input_ids'].min().item()}, max={batch['input_ids'].max().item()}")
                # 中文说明：标签形状/类型检查，避免 one-hot 或浮点导致 CE 梯度无效
                try:
                    y = batch['emoji_id']
                    shape = tuple(y.shape)
                    dtype = str(y.dtype)
                    is_onehot = (y.dim() == 2 and torch.all((y == 0) | (y == 1)))
                    print(f"[Data Check] emoji_id sample: {y[:5].tolist()} shape={shape} dtype={dtype} onehot={bool(is_onehot)}")
                except Exception:
                    print(f"[Data Check] emoji_id sample: {batch['emoji_id'][:5].tolist()}")
                # 检查是否存在异常值
                if torch.isnan(batch['input_ids']).any() or torch.isinf(batch['input_ids']).any():
                     print("[Data Check] CRITICAL: Input IDs contain NaN or Inf!")
            except Exception as e:
                print(f"[Data Check] Error: {e}")
            first_batch_checked = True

        optimizer.zero_grad()
        # 前向仅编码文本
        if use_amp:
            with autocast('cuda', dtype=torch.float16):
                tz = model.encode_text(batch['input_ids'], batch['attention_mask'])
                tau = torch.exp(model.log_tau)
                pos_feats = model.encode_images(batch['image'])
                # 中文说明：仅在本轮的第一个batch打印梯度诊断，避免日志过多
                loss = ce_triplet_loss(
                    tz, prototypes, batch['emoji_id'], tau,
                    top_k=top_k, margin=tri_margin,
                    w_ce=mix_weights[0], w_tri=mix_weights[1],
                    debug_print=(not printed_debug),  # 中文说明：仅首批次开启遮罩校验等调试打印
                    pos_features=pos_feats,
                    log_basic=True  # 中文说明：启用核心指标打印（pos/neg相似度与triplet）
                )
            scaler.scale(loss).backward()
            if debug_grad and not printed_debug:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        try:
                            # 检查梯度是否含有 NaN 或 Inf
                            is_nan = torch.isnan(param.grad).any()
                            is_inf = torch.isinf(param.grad).any()
                            
                            if is_nan:
                                print(f"[grad-error] {name} 梯度含NaN!", flush=True)
                            if is_inf:
                                print(f"[grad-error] {name} 梯度含Inf!", flush=True)
                                
                            # 仅打印前几层和特定层的梯度，或异常层
                            if is_nan or is_inf or "text_encoder.embeddings" in name or "text_proj" in name:
                                val = param.grad.abs().mean().item()
                                # print(f"[grad] {name}: {val:.6f}", flush=True)  # 中文说明：应用户要求，注释掉[grad]打印
                        except Exception:
                            pass
                # 中文说明：额外打印温度的梯度与当前值，诊断其是否被权重衰减/数值问题抑制
                try:
                    tau_val = float(torch.exp(model.log_tau).item())
                    tau_grad = model.log_tau.grad
                    tau_grad_mean = float(tau_grad.abs().mean().item()) if tau_grad is not None else 0.0
                    print(f"[tau] tau={tau_val:.5f} |dL/dlog_tau|≈{tau_grad_mean:.8f}")
                except Exception:
                    pass
                printed_debug = True
            scaler.step(optimizer)
            scaler.update()
        else:
            tz = model.encode_text(batch['input_ids'], batch['attention_mask'])
            tau = torch.exp(model.log_tau)
            pos_feats = model.encode_images(batch['image'])
            # 中文说明：仅在本轮的第一个batch打印梯度诊断，避免日志过多
            loss = ce_triplet_loss(
                tz, prototypes, batch['emoji_id'], tau,
                top_k=top_k, margin=tri_margin,
                w_ce=mix_weights[0], w_tri=mix_weights[1],
                debug_print=(not printed_debug),  # 中文说明：仅首批次开启遮罩校验等调试打印
                pos_features=pos_feats,
                log_basic=True  # 中文说明：启用核心指标打印（pos/neg相似度与triplet）
            )
            loss.backward()
            if debug_grad and not printed_debug:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        try:
                            if torch.isnan(param.grad).any():
                                print(f"[grad-nan] {name} 梯度含NaN!")
                            # print(f"[grad] {name}: {param.grad.abs().mean().item():.6f}")  # 中文说明：应用户要求，注释掉[grad]打印
                        except Exception:
                            pass
                # 中文说明：额外打印温度的梯度与当前值，诊断其是否被权重衰减/数值问题抑制
                try:
                    tau_val = float(torch.exp(model.log_tau).item())
                    tau_grad = model.log_tau.grad
                    tau_grad_mean = float(tau_grad.abs().mean().item()) if tau_grad is not None else 0.0
                    print(f"[tau] tau={tau_val:.5f} |dL/dlog_tau|≈{tau_grad_mean:.8f}")
                except Exception:
                    pass
                printed_debug = True
            
            # 中文说明：增加梯度裁剪，防止BERT微调时梯度爆炸导致NaN
            # 在裁剪前处理 Inf/NaN，避免 scaler 报错（若不使用AMP，则 clip_grad_norm_ 会处理）
            
            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 中文说明：在 step 前再次检查梯度是否有效，若仍有NaN/Inf则跳过更新
            grad_invalid = False
            for param in model.parameters():
                if param.grad is not None:
                     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        grad_invalid = True
                        break
            
            if grad_invalid:
                print(f"[grad-skip] Step {global_step}: 检测到梯度NaN/Inf，跳过本次参数更新！", flush=True)
                optimizer.zero_grad()
            else:
                optimizer.step()

        # ===== 温度约束：每步更新后对 log_tau 进行裁剪，避免温度学习到异常值 =====
        try:
            import math
            min_tau = 0.03
            max_tau = 0.20
            with torch.no_grad():
                model.log_tau.data.clamp_(math.log(min_tau), math.log(max_tau))
        except Exception:
            pass

        # 中文说明：训练步末尾打印损失，观察是否按预期下降（例如 4.38→3.5→2.8）
        try:
            print(f"Step {global_step}: Loss={loss.item():.4f}")
        except Exception:
            pass

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass
        if debug_grad and global_step < 100:
            try:
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                print(f"[lr] step={global_step} lr={lrs}")
            except Exception:
                pass
        # 中文说明：在首个调试步额外打印图像特征的“表示坍塌”指标：
        # 1) 余弦相似度矩阵的非对角均值/方差；2) 特征矩阵的秩
        if debug_grad and global_step == 0:
            try:
                # 中文说明：由于此函数未计算图片特征 iz，这里改为检查文本特征 tz 的“表示坍塌”指标
                sim = (tz @ tz.T).detach()
                n = sim.size(0)
                # 取非对角元素
                off_diag = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
                mean_od = float(off_diag.mean().item()) if off_diag.numel() > 0 else float('nan')
                std_od = float(off_diag.std(unbiased=False).item()) if off_diag.numel() > 0 else float('nan')
                # 特征矩阵秩（以数值稳定容差自动估计）
                rank = int(torch.linalg.matrix_rank(tz, tol=None).item())
                print(f"[collapse] off-diag cos mean={mean_od:.4f} std={std_od:.4f} | rank(tz)={rank}")
            except Exception:
                pass
        total += loss.item()
        global_step += 1
    return total / max(1, len(dataloader))


@torch.no_grad()
def eval_topk_accuracy(model, dataloader, device, prototypes, use_amp=False, top_k=(1, 5), proto_ids=None, class_names=None):
    """评估 Top-1 / Top-5 准确率：文本查询原型库。

    参数：
    - prototypes: [C, D] 类别原型向量；评估前已构建
    - top_k: 需要统计的 Top-K 列表，例如 (1,5)
    返回：dict，如 { 'top1': 0.35, 'top5': 0.62 }
    """
    model.eval()
    # 中文说明：按类别统计容器（Top-1），帮助定位识别好/差的具体表情
    num_classes = prototypes.size(0)
    per_class_total = [0] * int(num_classes)  # 每类样本数
    per_class_hit = [0] * int(num_classes)    # 每类Top-1命中数
    per_class_hit_topk = {k: [0] * int(num_classes) for k in top_k}
    per_class_rank_sum = [0] * int(num_classes)
    per_class_rank_cnt = [0] * int(num_classes)
    confusion = [[0] * int(num_classes) for _ in range(int(num_classes))]
    total = 0
    hit = {k: 0 for k in top_k}
    for batch in tqdm(dataloader, desc='eval(top-k)'):
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_amp:
            # 中文说明：torch.amp.autocast 需要显式声明 device_type（'cuda' 或 'cpu'）。
            # 在启用AMP时，我们仅在CUDA设备上使用FP16，因此传入 'cuda'。
            with autocast('cuda', dtype=torch.float16):
                tz = model.encode_text(batch['input_ids'], batch['attention_mask'])
        else:
            tz = model.encode_text(batch['input_ids'], batch['attention_mask'])
        # 与原型库计算相似度并排序
        logits = tz @ prototypes.t()  # [B, C]
        # 正确标签
        targets = batch['emoji_id']   # [B] 或 [B, C]（容错 one-hot）
        # 中文说明：标签清洗 —— one-hot → 索引；统一为 Long 类型
        try:
            if targets.dim() == 2:
                targets = targets.argmax(dim=1)
            if targets.dtype != torch.long:
                targets = targets.long()
        except Exception:
            pass
        # 计算 Top-K 命中
        sorted_idx = logits.argsort(dim=1, descending=True)
        for k in top_k:
            topk = sorted_idx[:, :k]
            topk_any = topk.eq(targets.unsqueeze(1)).any(dim=1)
            hit[k] += (topk_any.sum().item())
            try:
                t_cpu = targets.detach().cpu().tolist()
                any_cpu = topk_any.detach().cpu().tolist()
                for idx, ok in enumerate(any_cpu):
                    if ok:
                        per_class_hit_topk[k][int(t_cpu[idx])] += 1
            except Exception:
                pass
        # 中文说明：统计每个类别的Top-1表现（更直观查看哪类识别好/差）
        pred_top1 = sorted_idx[:, 0]
        t_cpu = targets.detach().cpu().tolist()
        p_cpu = pred_top1.detach().cpu().tolist()
        s_cpu = sorted_idx.detach().cpu().tolist()
        for bi, (t, p) in enumerate(zip(t_cpu, p_cpu)):
            ti = int(t)
            pi = int(p)
            per_class_total[ti] += 1
            if pi == ti:
                per_class_hit[ti] += 1
            confusion[ti][pi] += 1
            try:
                row = s_cpu[bi]
                rank_pos = 0
                for j, cid in enumerate(row):
                    if int(cid) == ti:
                        rank_pos = j
                        break
                per_class_rank_sum[ti] += (rank_pos + 1)
                per_class_rank_cnt[ti] += 1
            except Exception:
                pass
        total += tz.size(0)
    # 计算并打印Top-K，聚焦Top-1趋势（中文说明：运行脚本层建议打印为 Epoch x: Top-1=...）
    results = {f'top{k}': hit[k] / max(1, total) for k in top_k}
    try:
        top1 = results.get('top1', None)
        top5 = results.get('top5', None)
        if top1 is not None:
            if top5 is not None:
                print(f"[eval] Top-1={top1:.4f} | Top-5={top5:.4f}")
            else:
                print(f"[eval] Top-1={top1:.4f}")
    except Exception:
        pass
    
    # 中文说明：整理并打印“按类别”的Top-1准确率排名（前5最佳/后5薄弱），通过表情名字展示
    try:
        acc_per_class = []
        for i in range(int(num_classes)):
            n = per_class_total[i]
            acc = float(per_class_hit[i]) / n if n > 0 else float('nan')
            acc_per_class.append(acc)

        valid_indices = [i for i in range(int(num_classes)) if per_class_total[i] > 0]
        if len(valid_indices) > 0:
            sorted_desc = sorted(valid_indices, key=lambda i: acc_per_class[i], reverse=True)
            sorted_asc = sorted(valid_indices, key=lambda i: acc_per_class[i])
            show_n = min(5, len(valid_indices))
            best = sorted_desc[:show_n]
            worst = sorted_asc[:show_n]

            def name_of(i):
                try:
                    if class_names is not None and i < len(class_names) and class_names[i]:
                        return str(class_names[i])
                except Exception:
                    pass
                return f"id={i}"

            print("[per-class] Top-1类别表现（前5优秀）:")
            for i in best:
                acc = acc_per_class[i]
                n = per_class_total[i]
                print(f"  ✓ {name_of(i)}: Top-1={acc:.3f} (n={n})")

            print("[per-class] Top-1类别表现（后5薄弱）:")
            for i in worst:
                acc = acc_per_class[i]
                n = per_class_total[i]
            print(f"  ✗ {name_of(i)}: Top-1={acc:.3f} (n={n})")
    except Exception as e:
        print(f"[per-class] 统计失败：{e}")
    try:
        results['per_class_total'] = per_class_total
        results['per_class_hit'] = per_class_hit
        results['per_class_hit_topk'] = per_class_hit_topk
        results['per_class_rank_avg'] = [
            (float(per_class_rank_sum[i]) / per_class_rank_cnt[i]) if per_class_rank_cnt[i] > 0 else float('nan')
            for i in range(int(num_classes))
        ]
        results['confusion'] = confusion
        if class_names is not None:
            results['class_names'] = class_names
    except Exception:
        pass
    # ========== 分桶总结：按样本量区间汇总可读性指标 ==========
    try:
        def _summarize_bins(per_class_total, per_class_hit, class_names):
            bins = [
                ('≥ 500 条', 500, float('inf')),
                ('200–500 条', 200, 500),
                ('50–200 条', 50, 200),
                ('20–50 条', 20, 50),
                ('< 20 条', 0, 20),
            ]
            out = []
            for label, lo, hi in bins:
                idxs = [i for i, n in enumerate(per_class_total) if (n >= lo and (hi == float('inf') or n < hi))]
                acc_vals = []
                for i in idxs:
                    n = per_class_total[i]
                    h = per_class_hit[i]
                    if n > 0:
                        acc_vals.append(h / n)
                avg = (sum(acc_vals) / len(acc_vals)) if len(acc_vals) > 0 else 0.0
                # 典型例子：按样本量降序取前3个
                sorted_by_n = sorted(idxs, key=lambda i: per_class_total[i], reverse=True)
                examples = []
                for i in sorted_by_n[:3]:
                    try:
                        nm = None
                        if class_names is not None and i < len(class_names) and class_names[i]:
                            nm = str(class_names[i])
                        else:
                            nm = f"id={i}"
                        examples.append(f"{nm} ({per_class_total[i]} 条)")
                    except Exception:
                        examples.append(f"id={i} ({per_class_total[i]} 条)")
                out.append({'label': label, 'class_count': len(idxs), 'avg_top1': avg, 'examples': examples})
            return out
        bins_summary = _summarize_bins(per_class_total, per_class_hit, class_names)
        print("[summary] 每类样本量区间,类数量,平均 Top-1,典型例子")
        for item in bins_summary:
            ex_str = "、".join(item['examples']) if item['examples'] else "-"
            print(f" {item['label']},{item['class_count']},{item['avg_top1']*100:.1f}%,{ex_str}")
        results['bins'] = bins_summary
    except Exception:
        pass
    return results
