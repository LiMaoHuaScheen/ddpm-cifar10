#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DDPM 训练曲线 + 采样可视化。

读取 output_diffusion/train_history.json（loss 曲线），生成：
  - training_loss.png   (loss vs epoch)
  - samples_preview.png (最终采样图，放大到 128 便于查看)
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_loss(hist, out_dir):
    ep = hist['epoch']; loss = hist['loss']
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, loss, 'b-o', markersize=3, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (epsilon)')
    ax.set_title('DDPM Training Loss (CIFAR-10, 1000 steps, UNet 11.85M)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_loss.png'), dpi=150)
    plt.close(fig)
    print(f'[plot] saved training_loss.png (final loss={loss[-1]:.5f})', flush=True)

def plot_samples(samples_png, out_dir):
    img = Image.open(samples_png)
    w, h = img.size
    scale = 4
    big = img.resize((w * scale, h * scale), Image.NEAREST)
    big.save(os.path.join(out_dir, 'samples_preview.png'))
    print(f'[plot] saved samples_preview.png ({w}x{h} -> {w*scale}x{h*scale})', flush=True)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='output_diffusion')
    args = p.parse_args()
    hist_path = os.path.join(args.out_dir, 'train_history.json')
    with open(hist_path) as f:
        hist = json.load(f)
    plot_loss(hist, args.out_dir)
    sp = os.path.join(args.out_dir, 'samples.png')
    if os.path.exists(sp):
        plot_samples(sp, args.out_dir)

if __name__ == '__main__':
    main()
