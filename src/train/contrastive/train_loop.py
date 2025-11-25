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
    scaler = GradScaler('cuda', enabled=use_amp)  # 根据use_amp决定是否启用缩放器
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
                            print(f"[grad] {name}: {param.grad.abs().mean().item():.6f}")
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
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        try:
                            print(f"[grad] {name}: {param.grad.abs().mean().item():.6f}")
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
            optimizer.step()
        # 中文说明：按“步”进行调度。若使用Linear+Cosine的顺序调度器，将在此处逐步推进。
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
            with autocast(dtype=torch.float16):
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
def build_emoji_prototypes(model, dataset, device, use_amp=False):
    """从数据集的唯一图片路径构建原型库（每类一张图片 → 向量）。

    返回：原型张量 `P`，形状 [C, D]，C 为类别数（emoji唯一图片数），D 为投影维度。
    中文说明：每轮可重建一次以注入图像增强的多样性（增强由 dataset.img_tf 控制）。
    """
    model.eval()
    imgs = []
    for p in dataset.emoji_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        imgs.append(dataset.img_tf(img))
    P = torch.stack(imgs, dim=0).to(device)  # [C, 3, 224, 224]
    with torch.no_grad():
        if use_amp:
            with autocast('cuda', dtype=torch.float16):
                proto = model.encode_images(P)  # [C, D]
        else:
            proto = model.encode_images(P)
    return proto  # 单位向量（投影层已做 L2 归一化）


def ce_triplet_loss(tz: torch.Tensor, prototypes: torch.Tensor, targets: torch.Tensor,
                    tau: torch.Tensor, top_k: int = 10, margin: float = 0.2,
                    w_ce: float = 1.0, w_tri: float = 0.3) -> torch.Tensor:
    """混合损失：CrossEntropy 分类 + Triplet 排序。

    - CE：文本向量对所有类别原型做相似度，监督为 emoji_id（76 选 1）。
    - Triplet：在 Top-K 候选中进行半难负样本的排序约束；温度 tau 仅用于 Triplet 的相似度缩放。
    """
    # CE 部分：对所有类别计算相似度并监督分类
    logits = tz @ prototypes.t()  # [B, C]，不使用 tau（仅 Triplet 使用）
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
        topk_scores, topk_idx = masked.topk(k=min(top_k, prototypes.size(0)-1), dim=1)
    pos = prototypes[targets]                      # [B, D]
    # 取每个样本的一条负样本（Top-K中得分最高的那个）
    neg = prototypes[topk_idx[:, 0]]               # [B, D]
    # 采用余弦相似度的 margin 形式：鼓励 sim(a,p) ≥ sim(a,n) + margin
    # 中文说明：仅 Triplet 使用 tau 进行缩放，以调节“难度”。
    sim_pos = (tz @ pos.t()).diag() / tau.squeeze()   # [B]
    sim_neg = (tz @ neg.t()).diag() / tau.squeeze()   # [B]
    tri = F.relu(margin + sim_neg - sim_pos).mean()

    return w_ce * ce + w_tri * tri


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
    scaler = GradScaler('cuda', enabled=use_amp)
    printed_debug = False
    global_step = 0
    for batch in tqdm(dataloader, desc='train(retr)'):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        # 前向仅编码文本
        if use_amp:
            with autocast('cuda', dtype=torch.float16):
                tz = model.encode_text(batch['input_ids'], batch['attention_mask'])
                tau = torch.exp(model.log_tau)
                loss = ce_triplet_loss(tz, prototypes, batch['emoji_id'], tau,
                                       top_k=top_k, margin=tri_margin,
                                       w_ce=mix_weights[0], w_tri=mix_weights[1])
            scaler.scale(loss).backward()
            if debug_grad and not printed_debug:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        try:
                            print(f"[grad] {name}: {param.grad.abs().mean().item():.6f}")
                        except Exception:
                            pass
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
            loss = ce_triplet_loss(tz, prototypes, batch['emoji_id'], tau,
                                   top_k=top_k, margin=tri_margin,
                                   w_ce=mix_weights[0], w_tri=mix_weights[1])
            loss.backward()
            if debug_grad and not printed_debug:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        try:
                            print(f"[grad] {name}: {param.grad.abs().mean().item():.6f}")
                        except Exception:
                            pass
                try:
                    tau_val = float(torch.exp(model.log_tau).item())
                    tau_grad = model.log_tau.grad
                    tau_grad_mean = float(tau_grad.abs().mean().item()) if tau_grad is not None else 0.0
                    print(f"[tau] tau={tau_val:.5f} |dL/dlog_tau|≈{tau_grad_mean:.8f}")
                except Exception:
                    pass
                printed_debug = True
            optimizer.step()

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
        total += loss.item()
        global_step += 1
    return total / max(1, len(dataloader))


@torch.no_grad()
def eval_topk_accuracy(model, dataloader, device, prototypes, use_amp=False, top_k=(1, 5)):
    """评估 Top-1 / Top-5 准确率：文本查询原型库。

    参数：
    - prototypes: [C, D] 类别原型向量；评估前已构建
    - top_k: 需要统计的 Top-K 列表，例如 (1,5)
    返回：dict，如 { 'top1': 0.35, 'top5': 0.62 }
    """
    model.eval()
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
        targets = batch['emoji_id']   # [B]
        # 计算 Top-K 命中
        sorted_idx = logits.argsort(dim=1, descending=True)
        for k in top_k:
            topk = sorted_idx[:, :k]
            hit[k] += (topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item())
        total += tz.size(0)
    return {f'top{k}': hit[k] / max(1, total) for k in top_k}
