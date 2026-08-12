"""Multi-view 2D CNN (MVCNN-style) for rep D.

Input:  (B, n_views, H, W) silhouette stack (views treated as channels).
Output: (B, num_classes) logits.

The speed bet: 2D convolution on a small image stack is extremely GPU-friendly,
so both training and inference should undercut the 3D-conv (rep B) and the
point-MLP (rep A) on this hardware.
"""
import torch
import torch.nn as nn


class MultiViewCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 8, img_size: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),  # //2
            nn.Conv2d(32, 64, 3, padding=1),           nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),  # //2
            nn.Conv2d(64, 128, 3, padding=1),          nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # //2
        )
        # Probe body with a dummy batch to size the Linear layer concretely.
        with torch.no_grad():
            d = torch.zeros(2, in_channels, img_size, img_size)
            flat = self.body(d).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                 # x: (B, n_views, H, W)
        return self.head(self.body(x))
