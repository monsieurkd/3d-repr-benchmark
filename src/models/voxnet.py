"""VoxNet (Maturana & Scherer 2015) — small 3D-conv net for dense voxel grids.

Input:  (B, 1, R, R, R) occupancy grid (R = 32 in config).
Output: (B, num_classes) logits.

This is the natural counterpart to PointNet: same task, but the cost of 3D
convolution over a *dense* cube is what we expect to be slow — and what the
sparse variant (rep C) and the new format (rep D) aim to beat.
"""
import torch
import torch.nn as nn


class VoxNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, res: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3), nn.BatchNorm3d(32), nn.ReLU(),  # R-2
            nn.MaxPool3d(2),                                              # //2
            nn.Conv3d(32, 32, 3),        nn.BatchNorm3d(32), nn.ReLU(),   # -2
            nn.MaxPool3d(2),                                              # //2
        )
        # Probe the body with a dummy batch to get the exact flattened size,
        # so we can use a concrete Linear (no LazyLinear) and stay robust to
        # the voxel resolution. batch=2 avoids BatchNorm-on-N=1 issues.
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, res, res, res)
            flat = self.body(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):               # x: (B, 1, R, R, R)
        return self.head(self.body(x))
