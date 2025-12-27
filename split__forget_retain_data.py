import numpy as np
from torchvision import datasets
import os

DATA_DIR = "/home/theo/Desktop/Evaluate_Mia_Through_Unlearning/data"
os.makedirs(DATA_DIR, exist_ok=True)

train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
n = len(train)
k = int(round(0.1 * n))
rng = np.random.default_rng(42)
idx = rng.choice(n, size=k, replace=False).astype(np.int64)

out = os.path.join(DATA_DIR, "forget_idx_cifar.npy")
np.save(out, idx)

print(f"Saved {len(idx)} indices to {out} (n_train={n})")
print(f"min={idx.min()}, max={idx.max()}, dtype={idx.dtype}")