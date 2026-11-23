#!/usr/bin/env python3
"""下载 CIFAR-10 数据集（使用可用的国内镜像）。"""
import os
import tarfile
import urllib.request

MIRROR = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
DATA_DIR = "data"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    tarpath = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    if os.path.exists(os.path.join(DATA_DIR, "cifar-10-batches-py")):
        print("[download] CIFAR-10 already extracted.", flush=True)
        return
    if not os.path.exists(tarpath):
        print(f"[download] downloading {MIRROR} ...", flush=True)
        urllib.request.urlretrieve(MIRROR, tarpath)
        print("[download] done.", flush=True)
    else:
        print(f"[download] using cached {tarpath}", flush=True)
    print("[download] extracting ...", flush=True)
    with tarfile.open(tarpath, "r:gz") as tf:
        tf.extractall(DATA_DIR)
    print("[download] extracted to data/cifar-10-batches-py", flush=True)

if __name__ == "__main__":
    main()