#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从零实现的 DDPM（Denoising Diffusion Probabilistic Models）在 CIFAR-10 上训练。

核心：自定义 UNet（时间嵌入 + 残差块 + 注意力）+ 线性 beta 噪声调度 + epsilon 预测。
模式：
  test      - 冒烟测试（前向/反向/采样）
  train     - 完整训练
  sample    - 采样生成图 + 渐进过程图
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# ---------------------------------------------------------------------------
# 噪声调度（线性 beta，DDPM 标准）
# ---------------------------------------------------------------------------
def linear_beta_schedule(T=1000, beta1=1e-4, beta2=2e-2):
    return torch.linspace(beta1, beta2, T, dtype=torch.float32)

class DiffusionSchedule:
    def __init__(self, T=1000, beta1=1e-4, beta2=2e-2, device='cuda'):
        self.T = T
        self.device = device
        self.beta = linear_beta_schedule(T, beta1, beta2).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]])
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        # posterior mean coefs: mean = coef1*x0 + coef2*x
        # 正确公式（DDPM 附录 A）:
        #   coef1 = beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)
        #   coef2 = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        #   posterior_var = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        denom = (1.0 - self.alpha_bar)
        self.posterior_mean_coef1 = self.beta * torch.sqrt(self.alpha_bar_prev) / denom
        self.posterior_mean_coef2 = torch.sqrt(self.alpha) * (1.0 - self.alpha_bar_prev) / denom
        self.posterior_variance = self.beta * (1.0 - self.alpha_bar_prev) / denom

    def q_sample(self, x0, t, noise=None):
        """前向：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def p_mean_variance(self, model, x, t, clip_denoised=True):
        """预测均值与方差（模型预测 epsilon）"""
        eps = model(x, t)
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x0 = (x - sqrt_1mab * eps) / sqrt_ab
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        mean = coef1 * x0 + coef2 * x
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        log_var = torch.log(var.clamp(min=1e-20))
        return mean, var, log_var, x0

    @torch.no_grad()
    def sample(self, model, shape, device='cuda', prog_timesteps=None):
        """反向采样，完整 T 步。返回 (x0_final, prog_images, prog_labels)"""
        model.eval()
        T = self.T
        n = shape[0]
        x = torch.randn(shape, device=device)
        prog_images = []
        prog_labels = []
        record = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1] + (prog_timesteps or [])), reverse=True)
        for t in range(T - 1, -1, -1):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            mean, var, log_var, x0 = self.p_mean_variance(model, x, tt)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
            if t in record:
                img = (x.clamp(-1, 1) + 1) / 2
                prog_images.append(img.clone())
                prog_labels.append(t)
        model.train()
        return x, prog_images, prog_labels

# ---------------------------------------------------------------------------
# UNet 组件
# ---------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / half)
        args = time.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.time_mlp = nn.Linear(tdim, cout)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, t):
        h = self.norm1(x); h = F.silu(h); h = self.conv1(h)
        t_emb = self.time_mlp(F.silu(t)).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        h = self.norm2(h); h = F.silu(h); h = self.dropout(h); h = self.conv2(h)
        return h + self.skip(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3 * self.heads, C // self.heads, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv.chunk(3, dim=0)
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(out)

class Downsample(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
    def forward(self, x):
        return F.silu(self.conv(x))

class Upsample(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return F.silu(self.conv(x))

class UNet(nn.Module):
    """小型 UNet：适用于 32x32（CIFAR-10）。

    结构：in_conv -> [Res,Res,Attn,Down]*levels -> Mid(Res,Attn,Res) -> [Up,Res,Res,Attn]*levels -> out
    """
    def __init__(self, in_channels=3, base=64, ch_mult=(1, 2, 4), attn_res=(16, 8), tdim=256, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.base = base
        self.ch_mult = ch_mult
        self.attn_res = attn_res
        self.tdim = tdim
        chs = [base * m for m in ch_mult]
        n_levels = len(chs)
        self.time_mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.time_emb = SinusoidalPositionEmbeddings(base)
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        # 下采样路径：每层 [Res(c,c), Res(c,c), Attn?], 层间 Downsample
        self.downs = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i, c in enumerate(chs):
            blk = nn.ModuleList([
                ResBlock(c, c, tdim, dropout),
                ResBlock(c, c, tdim, dropout),
            ])
            self.downs.append(blk)
            self.down_attns.append(Attention(c) if (32 // (2 ** i)) in attn_res else nn.Identity())
            if i < n_levels - 1:
                self.down_samples.append(Downsample(c, chs[i + 1]))
            else:
                self.down_samples.append(nn.Identity())

        # 瓶颈
        self.mid1 = ResBlock(chs[-1], chs[-1], tdim, dropout)
        self.mid_attn = Attention(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], tdim, dropout)

        # 上采样路径：每层 [Up, Res(2c,c), Res(c,c), Attn?]
        self.ups = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_upsamples = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            c = chs[i]
            up_in = chs[i + 1] if i < n_levels - 1 else c  # 上一层通道
            self.ups.append(nn.ModuleList([
                Upsample(up_in, c) if i < n_levels - 1 else nn.Identity(),
                ResBlock(c * 2, c, tdim, dropout),
                ResBlock(c, c, tdim, dropout),
            ]))
            self.up_attns.append(Attention(c) if (32 // (2 ** i)) in attn_res else nn.Identity())

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x, t):
        x = self.in_conv(x)
        t_emb = self.time_mlp(self.time_emb(t))
        skips = []
        # 下采样
        for i in range(len(self.downs)):
            for blk in self.downs[i]:
                x = blk(x, t_emb)
            x = self.down_attns[i](x)
            skips.append(x)
            x = self.down_samples[i](x)
        # 瓶颈
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)
        # 上采样（ups[0]=最深/瓶颈侧，ups[-1]=最浅/输出侧；跳过 skip 需倒序对应）
        for i in range(len(self.ups)):
            dlvl = len(self.ups) - 1 - i  # ups[i] 构建于 level (n-1-i)，对应 skip[dlvl]
            x = self.ups[i][0](x)
            x = torch.cat([x, skips[dlvl]], dim=1)
            x = self.ups[i][1](x, t_emb)
            x = self.ups[i][2](x, t_emb)
            x = self.up_attns[i](x)
        x = self.out_norm(x)
        x = F.silu(x)
        return self.out_conv(x)

# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
    def update(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
            else:
                self.shadow[k] = v.clone()
    def apply(self):
        for k, v in self.shadow.items():
            self.model.state_dict()[k].copy_(v)

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_cifar(data_root, batch_size=128, num_workers=2):
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf)
    test = datasets.CIFAR10(root=data_root, train=False, download=False, transform=tf_eval)
    return train, test

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] device={device}', flush=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    train_ds, _ = load_cifar(args.data_root, args.batch_size)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f'[train] data: {len(train_ds)} images, batches/epoch={len(loader)}', flush=True)
    model = UNet(base=args.base).to(device)
    ema = EMA(model, args.ema_decay) if args.ema else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[train] UNet params={n_params/1e6:.2f}M', flush=True)
    sched = DiffusionSchedule(T=args.T, beta1=args.beta1, beta2=args.beta2, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    history = {'epoch': [], 'loss': []}
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0; nb = 0
        t0 = time.time()
        for x0, _ in loader:
            x0 = x0.to(device, non_blocking=True)
            b = x0.size(0)
            t = torch.randint(0, args.T, (b,), device=device, dtype=torch.long)
            x_t, noise = sched.q_sample(x0, t)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                eps_pred = model(x_t, t)
                loss = F.mse_loss(eps_pred.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update()
            ep_loss += loss.item(); nb += 1
        avg = ep_loss / max(nb, 1)
        history['epoch'].append(epoch)
        history['loss'].append(avg)
        dt = time.time() - t0
        print(f'[train] epoch {epoch}/{args.epochs} loss={avg:.5f} time={dt:.1f}s', flush=True)
        if epoch % args.save_every == 0:
            ckpt = {'model': model.state_dict(), 'ema': ema.shadow if ema else None, 'epoch': epoch, 'loss': avg}
            torch.save(ckpt, os.path.join(args.out_dir, 'ddpm_last.pt'))
            with open(os.path.join(args.out_dir, 'train_history.json'), 'w') as f:
                json.dump(history, f)
    ckpt = {'model': model.state_dict(), 'ema': ema.shadow if ema else None, 'epoch': args.epochs, 'loss': history['loss'][-1]}
    torch.save(ckpt, os.path.join(args.out_dir, 'ddpm_final.pt'))
    with open(os.path.join(args.out_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f)
    print(f'[train] DONE final loss={history["loss"][-1]:.5f}', flush=True)

# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    use_ema = args.ema and ckpt.get('ema') is not None
    sd = ckpt['ema'] if use_ema else ckpt['model']
    model.load_state_dict(sd)
    model.eval()
    print(f'[sample] loaded {args.ckpt} (ema={use_ema})', flush=True)
    sched = DiffusionSchedule(T=args.T, beta1=args.beta1, beta2=args.beta2, device=device)
    n = args.n_samples
    grid = int(math.ceil(n ** 0.5))
    x0, prog, plabels = sched.sample(model, (n, 3, 32, 32), device=device)
    gen = (x0.clamp(-1, 1) + 1) / 2
    save_image(gen, os.path.join(args.out_dir, 'samples.png'), nrow=grid, padding=2)
    rows = []
    for img in prog:
        rows.append(img[:min(n, 8)].clone())
    if rows:
        prog_img = torch.cat(rows, dim=0)
        save_image(prog_img, os.path.join(args.out_dir, 'progression.png'), nrow=min(n, 8), padding=2)
    info = {'prog_timesteps': plabels, 'n_samples': n, 'nrow': grid}
    with open(os.path.join(args.out_dir, 'sample_info.json'), 'w') as f:
        json.dump(info, f)
    print(f'[sample] saved samples.png ({n} images, grid {grid}x{grid})', flush=True)

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='train', choices=['test', 'train', 'sample'])
    p.add_argument('--data-root', default='data')
    p.add_argument('--out-dir', default='output_diffusion')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--beta1', type=float, default=1e-4)
    p.add_argument('--beta2', type=float, default=2e-2)
    p.add_argument('--base', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--ema', action='store_true', default=False)
    p.add_argument('--ema-decay', type=float, default=0.9999)
    p.add_argument('--n-samples', type=int, default=64)
    args = p.parse_args()
    if args.mode == 'test':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(base=args.base).to(device)
        sched = DiffusionSchedule(T=args.T, device=device)
        x = torch.randn(4, 3, 32, 32, device=device)
        t = torch.randint(0, args.T, (4,), device=device)
        eps = model(x, t)
        print(f'[test] forward ok, eps shape={eps.shape}', flush=True)
        loss = F.mse_loss(eps, torch.randn_like(eps))
        loss.backward()
        print(f'[test] backward ok, loss={loss.item():.4f}', flush=True)
        x0, prog, plabels = sched.sample(model, (2, 3, 32, 32), device=device)
        print(f'[test] sample ok, prog timesteps={plabels[:5]}...', flush=True)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        if not args.ckpt:
            print('[sample] need --ckpt', flush=True)
            return
        sample(args)

if __name__ == '__main__':
    main()
