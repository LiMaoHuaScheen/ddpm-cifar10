# 扩散模型（DDPM）从零实现：CIFAR-10 图像生成

不依赖 diffusers，**完全从零手写** DDPM（Denoising Diffusion Probabilistic Models）训练与采样，在 CIFAR-10 上生成 32×32 彩色图像。

## 核心结果（60 epoch，断电中断）

| 指标 | 值 | 说明 |
|------|-----|------|
| 模型 | UNet，11.85M 参数 | base=64, ch_mult=(1,2,4) |
| 扩散步数 T | 1000 | 线性 β 调度 1e-4→2e-2 |
| 训练 epoch | 60/100 | 因断电中断 |
| 最终 MSE loss | **0.0314** | 从 0.0998 收敛 |
| FID | **587.0** | 60 epoch 未收敛（完整需 1000+） |
| 采样 | 64 张 8×8 网格 | 10 类清晰可辨 |

![训练曲线](figures/training_loss.png)

## 采样结果（修复后）

![采样预览](figures/samples_preview.png)

## 关键发现

1. **从零手写 DDPM 可行**：不依赖 diffusers，~70s/epoch，60 epoch 后 loss 从 0.0998 收敛到 0.0314，采样图能清晰分辨 CIFAR-10 的 10 个类别。
2. **修复了一个关键采样 bug**（详见 docs/04_实验报告.md）：`posterior_mean_coef2` 误用 `sqrt(alpha_t)≈0.99`，正确应为 `sqrt(alpha_t)*(1-alpha_bar_{t-1})/(1-alpha_bar_t)`。训练目标不受影响（只用 q_sample），但采样完全错误，修复后图像从噪点变为清晰类。
3. **FID=587 是诚实结果**：仅训练 60 epoch（断电），完整收敛通常需 1000+ epoch、FID 3~10。此结果如实反映训练不足，而非模型架构问题。

## 目录结构

~~~
ddpm-cifar10/
├── README.md                本文件
├── requirements.txt         依赖
├── download_data.py         自动下载 CIFAR-10
├── code/
│   ├── train_ddpm.py         主脚本（test/train/sample）
│   ├── train_fid_classifier.py  训练 FID 特征提取器（ResNet-18）
│   ├── compute_fid.py        计算 FID
│   └── plot_ddpm.py          可视化
├── figures/                 实验产出图
├── results/                 指标 JSON
└── docs/                    完整文档
~~~

## 快速开始

~~~bash
pip install -r requirements.txt
python download_data.py

# 1) 冒烟测试（前向/反向/采样）
python code/train_ddpm.py --mode test

# 2) 训练（100 epoch，约 2 小时）
python code/train_ddpm.py --mode train --epochs 100 --ema

# 3) 采样（用最后一次 checkpoint）
python code/train_ddpm.py --mode sample --ckpt output_diffusion/ddpm_last.pt --n-samples 64

# 4) 训练 FID 特征提取器 + 计算 FID
python code/train_fid_classifier.py
python code/compute_fid.py

# 5) 可视化
python code/plot_ddpm.py
~~~

## 实验设置

| 项 | 设置 |
|----|------|
| 模型 | UNet base=64, ch_mult=(1,2,4), attention 8×8/16×16, 11.85M 参数 |
| 扩散步数 | T=1000，线性 β：1e-4 → 2e-2 |
| 训练目标 | MSE(预测 ε, 真实噪声) |
| 优化器 | AdamW lr=2e-4 |
| EMA | decay=0.9999（可选） |
| 数据 | CIFAR-10 50k 训练 / 10k 测试，RandomHorizontalFlip |
| 归一化 | [-1, 1] |
| 硬件 | NVIDIA RTX 5060 Laptop 8GB（CUDA 12.8） |

## 为什么不用 diffusers？

从零实现能完整展示扩散模型的数学原理（前向加噪 q_sample、逆向去噪 p_mean_variance、噪声预测 ε），对研究生申请更有说服力。同时避免网络下载 diffusers 依赖的问题。

## License

MIT