import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def info_nce_loss(tz, iz, tau):
    # 计算双向InfoNCE对比损失：文本→图片、图片→文本
    sim = tz @ iz.t() / tau                      # 相似度矩阵 (B,B)
    labels = torch.arange(tz.size(0), device=tz.device)
    loss_t2i = F.cross_entropy(sim, labels)      # 文本查询 → 图片
    loss_i2t = F.cross_entropy(sim.t(), labels)  # 图片查询 → 文本
    return (loss_t2i + loss_i2t) / 2.0


def train_one_epoch(model, dataloader, optimizer, device, use_amp=False):
    # 单轮训练：遍历数据、计算损失并反传
    # 中文说明：当 use_amp=True 时，启用混合精度以减少显存占用并提升速度
    model.train()
    total = 0.0
    scaler = GradScaler(enabled=use_amp)  # 根据use_amp决定是否启用缩放器
    for batch in tqdm(dataloader, desc='train'):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        if use_amp:
            # 混合精度前向与损失计算
            with autocast(dtype=torch.float16):
                tz, iz, tau, _, _ = model(batch)
                loss = info_nce_loss(tz, iz, tau)
            # 反传与更新使用GradScaler防止溢出
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            tz, iz, tau, _, _ = model(batch)
            loss = info_nce_loss(tz, iz, tau)
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / max(1, len(dataloader))


@torch.no_grad()
def eval_retrieval(model, dataloader, device, use_amp=False):
    # 简单检索评估：计算文本→图片的R@1
    # 中文说明：评估阶段也可使用autocast减少显存占用，加速推理
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
        pred = sim.argmax(dim=1)                 # 每个文本预测最相似图片索引
        gt = torch.arange(tz.size(0), device=tz.device)
        correct += (pred == gt).sum().item()
        total += tz.size(0)
    return correct / max(1, total)