#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""计算 FID（Frechet Inception Distance）。

用 CIFAR-10 分类器（ResNet-18，ImageNet 预训练 + 微调）作为特征提取器。
FID 计算参考实现（Heusel et al. 2017）：
  d^2 = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2*sqrt(Sigma1 Sigma2))
特征使用 avgpool 后 512 维向量。
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn

def load_cifar(root, batch=128):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    te = datasets.CIFAR10(root=root, train=False, download=False, transform=tf)
    return te

def extract_features(model, loader, device):
    model.eval()
    feats = []
    def hook_fn(module, inp, out):
        feats.append(out.detach().cpu().squeeze())
    handle = None
    for name, mod in model.named_modules():
        if name == 'avgpool':
            handle = mod.register_forward_hook(hook_fn)
            break
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            model(x)
    if handle:
        handle.remove()
    feats = torch.cat([f.view(f.size(0), -1) for f in feats], dim=0)
    return feats.numpy()

def calc_fid(mu1, sigma1, mu2, sigma2):
    # 特征维度可能很大，使用矩阵平方根近似
    diff = mu1 - mu2
    # Tr(Sigma1 + Sigma2 - 2*sqrt(Sigma1 Sigma2))
    # 用对称平方根：sqrt(Sigma1 Sigma2) ~ 需要 SVD
    covmean = _sqrtm(sigma1 @ sigma2)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def _sqrtm(A):
    # 对称矩阵平方根（对称化，避免数值误差）
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)  # 数值稳定
    return (vecs * np.sqrt(vals)) @ vecs.T

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data')
    p.add_argument('--clf-ckpt', default='output_diffusion/fid_clf.pt')
    p.add_argument('--samples', default='output_diffusion/samples.png')
    p.add_argument('--out-dir', default='output_diffusion')
    p.add_argument('--n-real', type=int, default=10000)
    p.add_argument('--n-gen', type=int, default=5000)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    device = torch.device(args.device)
    # 1) 加载分类器
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(args.clf_ckpt, map_location=device))
    model = model.to(device)
    # 2) 真实分布特征（测试集，截取 n_real）
    te = load_cifar(args.data_root)
    te_loader = DataLoader(te, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    feats_real = extract_features(model, te_loader, device)
    if len(feats_real) > args.n_real:
        feats_real = feats_real[:args.n_real]
    # 3) 生成分布特征：从 samples.png 读回（归一化 [0,1] -> [-1,1]，然后按 clf 的 normalize）
    #    samples.png 是 [0,1] 的 32x32 图，CIFAR 生成。需要转回 [-1,1] 再 normalize。
    from PIL import Image
    from torchvision.transforms import ToTensor
    img = Image.open(args.samples).convert('RGB')
    # 读 grid -> 需要知道 nrow；这里简单做法：把图切回单张（用已知 nrow 来自 sample_info）
    info_path = os.path.join(args.out_dir, 'sample_info.json')
    nrow = 8
    if os.path.exists(info_path):
        with open(info_path) as f:
            nrow = json.load(f).get('nrow', 8)
    # 计算每个 cell 32x32（padding=2 在 save_image 中）
    w, h = img.size
    cell = 32
    pad = 2
    grid_w = (w - pad) // (cell + pad)
    grid_h = (h - pad) // (cell + pad)
    gen_imgs = []
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    to_t = transforms.ToTensor()
    for gy in range(grid_h):
        for gx in range(grid_w):
            x0 = pad + gx * (cell + pad)
            y0 = pad + gy * (cell + pad)
            crop = img.crop((x0, y0, x0 + cell, y0 + cell))
            arr = np.array(crop).astype(np.float32) / 255.0
            # [0,1] -> [-1,1]
            arr = arr * 2 - 1
            arr = (arr - np.array([0.4914, 0.4822, 0.4465])) / np.array([0.2470, 0.2435, 0.2616])
            t = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
            gen_imgs.append(t)
    gen_tensor = torch.cat(gen_imgs, dim=0)[:args.n_gen]
    gen_loader = DataLoader(gen_tensor, batch_size=128, shuffle=False)
    feats_gen = extract_features(model, gen_loader, device)
    # 4) FID
    mu1, sigma1 = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu2, sigma2 = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)
    fid = calc_fid(mu1, sigma1, mu2, sigma2)
    print(f'[fid] FID={fid:.2f} (real n={len(feats_real)}, gen n={len(feats_gen)})', flush=True)
    # 保存
    res = {'fid': fid, 'n_real': len(feats_real), 'n_gen': len(feats_gen),
           'mu_real': mu1.tolist(), 'mu_gen': mu2.tolist()}
    with open(os.path.join(args.out_dir, 'fid_result.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False)

if __name__ == '__main__':
    main()
