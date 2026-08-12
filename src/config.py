"""Central config for the 3D-representation benchmark.

Single source of truth for paths, device, seeds, and representation parameters.
Keeping these here is what makes a fair comparison: every representation reads
the same knobs.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
MODELNET_DIR = ROOT / "data" / "ModelNet40"          # extracted meshes
PROCESSED = ROOT / "results" / "processed"           # cached arrays
RESULTS = ROOT / "results"

# Device: MPS if available (Apple Silicon), else CUDA, else CPU.
# IMPORTANT for a timing benchmark: the SAME device is used for every rep.
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

# Representation parameters
NUM_POINTS = 1024        # points sampled per mesh (rep A / rep C)
VOX_RES = 32             # voxel grid edge (rep B dense / rep C sparse)
MV_VIEWS = 8             # number of rendered views (rep D multi-view)
MV_IMG_SIZE = 64         # per-view image edge in pixels (rep D)

# Training knobs (held constant across reps)
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

# 10-class dev subset (subset of ModelNet40). Used for fast iteration;
# final numbers run on all 40 classes.
DEV_CLASSES = [
    "airplane", "bathtub", "chair", "desk", "dresser",
    "lamp", "sofa", "table", "toilet", "xbox",
]
