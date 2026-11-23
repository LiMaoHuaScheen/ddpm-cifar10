#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练一个 CIFAR-10 分类器（ResNet-18，ImageNet 预训练 + 微调），
作为 FID 的特征提取器（Inception 权重不可达时的替代方案）。
输出：FID 特征提取器权重 + 测试集特征（作为参考分布）。
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def load_cifar(root, batch=128):
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tr = datasets.CIFAR10(root=root, train=True, download=False, transform=tf)
    te = datasets.CIFAR10(root=root, train=False, download=False, transform=tf_eval)
    return tr, te

def extract_features(model, loader, device, layer='avgpool'):
    """提取特征：将模型在指定层后的输出作为特征（global pooling -> feature vector）"""
    model.eval()
    feats = []
    def hook_fn(module, inp, out):
        feats.append(out.detach().cpu().squeeze())
    # 注册 hook 到 avgpool（输出 512 维）
    handle = None
    for name, mod in model.named_modules():
        if name == 'avgpool':
            handle = mod.register_forward_hook(hook_fn)
            break
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)
    if handle:
        handle.remove()
    feats = torch.cat([f.view(f.size(0), -1) for f in feats], dim=0)
    return feats.numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data')
    p.add_argument('--out-dir', default='output_diffusion')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    device = torch.device(args.device)
    tr, te = load_cifar(args.data_root, args.batch_size)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 替换最后一层为 10 类
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.out_dir, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        ep_loss = 0.0; nb = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        # 验证
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
        acc = correct / total
        print(f'[clf] epoch {epoch}/{args.epochs} loss={ep_loss/nb:.4f} test_acc={acc:.4f} t={time.time():.0f}', flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'fid_clf.pt'))
    # 提取特征（用最佳权重）
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'fid_clf.pt'), map_location=device))
    feats_tr = extract_features(model, tr_loader, device)
    feats_te = extract_features(model, te_loader, device)
    np.save(os.path.join(args.out_dir, 'ref_features_train.npy'), feats_tr)
    np.save(os.path.join(args.out_dir, 'ref_features_test.npy'), feats_te)
    print(f'[clf] DONE best_acc={best:.4f} feats_train={feats_tr.shape} feats_test={feats_te.shape}', flush=True)

if __name__ == '__main__':
    main()
