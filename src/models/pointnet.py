"""PointNet (Qi et al. 2017) — the canonical point-cloud classifier.

Input:  (B, N, 3) point cloud.
Output: (B, num_classes) logits.

The max-pool over points is the key trick: it makes the net invariant to point
ordering and permutation — which is why an *unordered* representation (a point
cloud) can be classified at all.
"""
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        # Per-point feature extractor (1x1 convs = shared MLP over points).
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1),          nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1),         nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1),       nn.BatchNorm1d(1024), nn.ReLU(),
        )
        # Classifier on the global (max-pooled) descriptor.
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):          # x: (B, N, 3)
        x = x.transpose(1, 2)      # (B, 3, N) for Conv1d
        f = self.feat(x)           # (B, 1024, N)
        g = f.max(dim=2).values    # (B, 1024) — symmetric global feature
        return self.head(g)
