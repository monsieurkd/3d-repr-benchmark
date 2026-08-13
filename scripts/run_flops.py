"""Count theoretical FLOPs for each of the 4 representations.

This is the *cheap* evidence the blog leans on: a single forward pass per
representation under torch.utils.flop_counter.FlopCounterMode, no training,
no GPU. FLOPs are counted on CPU, which is fine because a FLOP count is a
property of the network + input shape, not the device it runs on.

The point of the exercise is to compare the FLOPs ordering against the
wall-clock ordering measured on real (Apple Silicon) hardware and show where
they disagree. That disagreement is the proof that FLOPs mislead here.

Output: results/flops.json
"""
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

# Repo is imported via src/ as a package, mirroring run_baseline.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import PointNet, VoxNet, MultiViewCNN  # noqa: E402

NUM_CLASSES = 40  # matches the *_all_c40_e15.json runs
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def count_flops(model, x, num_warmup=3):
    """One counted forward pass. Warm up first so lazy initialisation (cuDNN-style
    autotuning, allocator caches) does not perturb the count. Under no_grad/eval
    so dropout/BN behave as inference and only real compute is counted."""
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
        fcm = FlopCounterMode(display=False)
        with fcm:
            _ = model(x)
    # fcm.get_total_flops() counts multiply-accumulate as 1 FLOP. The convention
    # in the 3D literature is to report MACs; we expose that convention plainly
    # so the blog can be honest about the number it prints.
    return fcm.get_total_flops()


def main():
    torch.manual_seed(0)
    # All counting on CPU: FLOP counts are shape-determined, and FlopCounterMode
    # is CPU/deterministic-friendly.
    reps = [
        {
            "rep": "point", "model": "PointNet", "shape": [1, 1024, 3],
            "build": lambda: PointNet(NUM_CLASSES),
        },
        {
            "rep": "voxel", "model": "VoxNet", "shape": [1, 1, 32, 32, 32],
            "build": lambda: VoxNet(NUM_CLASSES, res=32),
        },
        {
            # Sparse voxel is approximated as a point set through PointNet
            # (spconv, the production sparse-3D-conv library, is unavailable on
            # MPS). So it shares PointNet's architecture and therefore its FLOPs.
            "rep": "sparse", "model": "PointNet (sparse point set)", "shape": [1, 1024, 3],
            "build": lambda: PointNet(NUM_CLASSES),
        },
        {
            "rep": "mview", "model": "MultiViewCNN", "shape": [1, 8, 64, 64],
            "build": lambda: MultiViewCNN(NUM_CLASSES, in_channels=8, img_size=64),
        },
    ]

    out = {"num_classes": NUM_CLASSES, "device": "cpu", "convention": "MACs (1 multiply-accumulate = 1 FLOP)", "runs": []}
    for r in reps:
        model = r["build"]().cpu()
        x = torch.zeros(*r["shape"]).cpu()
        params = sum(p.numel() for p in model.parameters())
        flops = count_flops(model, x)
        out["runs"].append({
            "rep": r["rep"],
            "model": r["model"],
            "input_shape": r["shape"],
            "params": params,
            "macs_per_sample": flops,
        })
        print(f"{r['rep']:<8} {r['model']:<28} shape={str(r['shape']):<20} "
              f"params={params:>9,}  MACs/sample={flops:>14,}")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "flops.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
