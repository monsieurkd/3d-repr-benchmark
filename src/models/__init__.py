"""Model zoo for the 3D-representation benchmark.

One small, idiomatic net per representation. All take a batch tensor and return
logits (B, num_classes). Kept deliberately small so train/infer time reflects the
REPRESENTATION's cost, not a heavy backbone.
"""
from .pointnet import PointNet
from .voxnet import VoxNet
from .multiview import MultiViewCNN

__all__ = ["PointNet", "VoxNet", "MultiViewCNN"]
