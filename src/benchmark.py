"""Training + evaluation + TIMING harness, loader-free.

Why no DataLoader in the timed loop: the models here are tiny, so DataLoader
worker IPC would dominate the clock and we'd measure loading infrastructure,
not the representation's compute cost. So we preload each split into device
tensors once (amortized setup, timed separately), then iterate index batches
for pure model-on-representation timing.

Reports, per representation:
  - preprocess time (mesh -> rep), from the cache-build pass (in run_baseline)
  - load-to-device time (amortized setup)
  - per-epoch train time + total train time
  - inference time per sample (warmup-excluded)
  - accuracy

Timing controls (same for every rep): same device, seeds, batch, epochs;
MPS/CUDA synchronized before stopping the clock; inference warmup excluded.
"""
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def load_to_device(ds, device):
    """Read cached arrays in-process (no worker IPC) and move to device.

    Single-process is faster than DataLoader here: items are tiny cached
    arrays, so DataLoader worker spawn/IPC would dominate the clock."""
    xs = [ds[i][0] for i in range(len(ds))]
    ys = [ds[i][1] for i in range(len(ds))]
    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    return X, Y


def train_model(model, X, Y, batch, epochs, lr, device, verbose=True):
    """Returns (per_epoch_times_sec, per_epoch_acc, total_time_sec)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = X.size(0)
    per_t, per_acc = [], []
    model.train()
    _sync(device)
    wall = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        t0 = time.perf_counter()
        correct, n = 0, 0
        for i in range(0, N - batch + 1, batch):
            idx = perm[i:i + batch]
            logits = model(X[idx])
            loss = F.cross_entropy(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == Y[idx]).sum().item()
            n += idx.size(0)
        _sync(device)
        per_t.append(time.perf_counter() - t0)
        per_acc.append(correct / max(n, 1))
        if verbose:
            print(f"  epoch {ep+1:2d}/{epochs}: {per_t[-1]:6.2f}s  "
                  f"train_acc={per_acc[-1]:.3f}", flush=True)
    return per_t, per_acc, time.perf_counter() - wall


@torch.no_grad()
def evaluate(model, X, Y, batch, device, warmup=5):
    """Returns (accuracy, total_infer_sec, n_samples). Warmup excluded."""
    model.eval()
    N = X.size(0)
    steps = max(1, N // batch)
    for i in range(min(warmup, steps)):
        model(X[i * batch:(i + 1) * batch])
    _sync(device)
    t0 = time.perf_counter()
    correct = 0
    for i in range(0, N, batch):
        logits = model(X[i:i + batch])
        correct += (logits.argmax(1) == Y[i:i + batch]).sum().item()
    _sync(device)
    return correct / N, time.perf_counter() - t0, N
